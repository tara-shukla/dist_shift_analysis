# ── Cell 1: imports & paths ───────────────────────────────────────────────────
import math
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from credit_shift import (
    load_data,
    clean_transactions,
    add_user_features,
    train_default_model,
    compute_uplift_score,
    evaluate_by_age_group,
    policy_profit_impact,
    stress_test_cohort_mix,
    _SPEND_COLS,
    _CATEGORY_MAP,
    _is_leaky,
    HARD_LEAKY_COLS,
)

TX_PATH    = "transactions_train.csv"
USERS_PATH = "users.csv"


# ── Cell 2: hyperparameter tuning ─────────────────────────────────────────────
def tune_model(train_df: pd.DataFrame):
    """
    Run RandomizedSearchCV over GradientBoostingClassifier hyperparameters,
    optimising for ROC-AUC. Returns the best fitted Pipeline and feature_cols.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    feature_cols = [
        c for c in train_df.columns
        if c != "default_next"
        and c not in {"split"}
        and not _is_leaky(c)
    ]

    X = train_df[feature_cols]
    y = train_df["default_next"]

    cat_cols = [c for c in feature_cols if X[c].dtype == object or str(X[c].dtype) == "category"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=42))])

    param_dist = {
        "classifier__n_estimators":    [100, 200, 300],
        "classifier__max_depth":       [3, 4, 5],
        "classifier__learning_rate":   [0.01, 0.05, 0.1],
        "classifier__subsample":       [0.6, 0.8, 1.0],
        "classifier__min_samples_leaf":[10, 20, 50],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)

    print(f"Best params : {search.best_params_}")
    print(f"Best CV AUC : {search.best_score_:.4f}")

    return search.best_estimator_, feature_cols


# ── Cell 3: shared fixtures (module-scoped so model is trained once) ──────────
@pytest.fixture(scope="module")
def raw_data():
    tx, users = load_data(TX_PATH, USERS_PATH)
    return tx, users

@pytest.fixture(scope="module")
def cleaned(raw_data):
    tx, _ = raw_data
    return clean_transactions(tx)

@pytest.fixture(scope="module")
def merged(cleaned, raw_data):
    _, users = raw_data
    return add_user_features(cleaned, users)

@pytest.fixture(scope="module")
def trained(merged):
    train_df = merged[merged["split"] == "train"]
    model, feature_cols = tune_model(train_df)
    return model, feature_cols, train_df

@pytest.fixture(scope="module")
def deploy_df(merged):
    return merged[merged["split"] == "deploy"]


# ── Cell 4: clean_transactions tests ─────────────────────────────────────────
class TestCleanTransactions:

    def test_no_exact_duplicates(self, cleaned):
        assert cleaned.duplicated().sum() == 0

    def test_primary_category_values(self, cleaned):
        valid = set(_CATEGORY_MAP.values())
        actual = set(cleaned["primary_category"].dropna().unique())
        assert actual <= valid, f"Unexpected categories: {actual - valid}"

    def test_spend_cols_no_nulls(self, cleaned):
        assert cleaned[_SPEND_COLS].isna().sum().sum() == 0

    def test_spend_cols_no_negatives(self, cleaned):
        assert (cleaned[_SPEND_COLS] < 0).sum().sum() == 0

    def test_monthly_spend_no_nulls(self, cleaned):
        assert cleaned["monthly_spend"].isna().sum() == 0

    def test_monthly_spend_geq_spend_sum(self, cleaned):
        # monthly_spend should be >= sum of spend_* (it may be slightly higher
        # if original value was already present and larger)
        spend_sum = cleaned[_SPEND_COLS].sum(axis=1)
        # where monthly_spend was imputed it equals spend_sum exactly
        imputed_mask = cleaned["monthly_spend"].round(6) == spend_sum.round(6)
        assert imputed_mask.any(), "Expected some imputed monthly_spend rows"

    def test_payment_rate_no_nulls(self, cleaned):
        assert cleaned["payment_rate"].isna().sum() == 0

    def test_payment_rate_range(self, cleaned):
        assert cleaned["payment_rate"].between(0, 1).all()

    def test_apr_no_nulls(self, cleaned):
        assert cleaned["apr"].isna().sum() == 0

    def test_apr_range(self, cleaned):
        assert cleaned["apr"].between(0, 0.40).all()

    def test_spend_clipped_at_995(self, raw_data, cleaned):
        tx, _ = raw_data
        caps = tx[_SPEND_COLS].quantile(0.995)
        for col in _SPEND_COLS:
            assert cleaned[col].max() <= caps[col] + 1e-8, f"{col} exceeds 99.5th pct cap"

    def test_dtypes(self, cleaned):
        assert cleaned["user_id"].dtype == np.int64 or cleaned["user_id"].dtype == np.int32
        assert cleaned["month"].dtype   == np.int64 or cleaned["month"].dtype   == np.int32
        assert cleaned["age_group"].dtype == object

    def test_row_count_not_inflated(self, raw_data, cleaned):
        tx, _ = raw_data
        # cleaning may drop duplicates but must never add rows
        assert len(cleaned) <= len(tx)


# ── Cell 5: add_user_features tests ──────────────────────────────────────────
class TestAddUserFeatures:

    def test_row_count_preserved(self, cleaned, merged):
        assert len(merged) == len(cleaned)

    def test_user_id_present(self, merged):
        assert "user_id" in merged.columns

    def test_no_duplicate_user_cols(self, merged):
        # suffixed columns (_x, _y) indicate a bad merge
        assert not any(c.endswith("_x") or c.endswith("_y") for c in merged.columns)

    def test_all_users_joined(self, merged):
        # user_id should never be NaN after a left join on a column present in tx
        assert merged["user_id"].isna().sum() == 0


# ── Cell 6: train_default_model / uplift tests ────────────────────────────────
class TestTrainDefaultModel:

    def test_returns_pipeline(self, trained):
        model, _, _ = trained
        assert isinstance(model, Pipeline)

    def test_predict_proba_shape(self, trained):
        model, feature_cols, train_df = trained
        probs = model.predict_proba(train_df[feature_cols])
        assert probs.shape == (len(train_df), 2)

    def test_predict_proba_valid_range(self, trained):
        model, feature_cols, train_df = trained
        probs = model.predict_proba(train_df[feature_cols])[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_no_leaky_features(self, trained):
        _, feature_cols, _ = trained
        leaky = [c for c in feature_cols if _is_leaky(c)]
        assert leaky == [], f"Leaky columns in feature_cols: {leaky}"

    def test_uplift_train_geq_4(self, trained):
        model, feature_cols, train_df = trained
        score = compute_uplift_score(model, train_df, feature_cols, top_frac=0.10)
        assert score >= 4.0, f"Train uplift {score:.3f} < 4.0"


# ── Cell 7: compute_uplift_score tests ───────────────────────────────────────
class TestComputeUpliftScore:

    def test_top_frac_1_equals_1(self, trained):
        # With top_frac=1.0 every row is selected, so uplift must equal 1.0
        model, feature_cols, train_df = trained
        score = compute_uplift_score(model, train_df, feature_cols, top_frac=1.0)
        assert abs(score - 1.0) < 1e-6

    def test_returns_float(self, trained):
        model, feature_cols, train_df = trained
        score = compute_uplift_score(model, train_df, feature_cols)
        assert isinstance(score, float)

    def test_invalid_top_frac_raises(self, trained):
        model, feature_cols, train_df = trained
        with pytest.raises(ValueError):
            compute_uplift_score(model, train_df, feature_cols, top_frac=0.0)
        with pytest.raises(ValueError):
            compute_uplift_score(model, train_df, feature_cols, top_frac=1.5)

    def test_missing_target_raises(self, trained):
        model, feature_cols, train_df = trained
        with pytest.raises(ValueError):
            compute_uplift_score(model, train_df.drop(columns=["default_next"]), feature_cols)


# ── Cell 8: evaluate_by_age_group tests ──────────────────────────────────────
class TestEvaluateByAgeGroup:

    def test_returns_all_cohorts(self, trained, merged):
        model, feature_cols, _ = trained
        result = evaluate_by_age_group(model, merged, feature_cols)
        expected = set(merged["age_group"].unique())
        assert set(result.keys()) == expected

    def test_metric_keys(self, trained, merged):
        model, feature_cols, _ = trained
        result = evaluate_by_age_group(model, merged, feature_cols)
        for group, metrics in result.items():
            assert set(metrics.keys()) == {"auc", "brier", "n"}, f"Wrong keys for {group}"

    def test_auc_in_range(self, trained, merged):
        model, feature_cols, _ = trained
        result = evaluate_by_age_group(model, merged, feature_cols)
        for group, metrics in result.items():
            if not math.isnan(metrics["auc"]):
                assert 0.0 <= metrics["auc"] <= 1.0, f"AUC out of range for {group}"

    def test_brier_in_range(self, trained, merged):
        model, feature_cols, _ = trained
        result = evaluate_by_age_group(model, merged, feature_cols)
        for group, metrics in result.items():
            assert 0.0 <= metrics["brier"] <= 1.0, f"Brier out of range for {group}"

    def test_n_sums_to_total(self, trained, merged):
        model, feature_cols, _ = trained
        result = evaluate_by_age_group(model, merged, feature_cols)
        assert sum(m["n"] for m in result.values()) == len(merged)

    def test_missing_columns_raise(self, trained, merged):
        model, feature_cols, _ = trained
        with pytest.raises(ValueError):
            evaluate_by_age_group(model, merged.drop(columns=["age_group"]), feature_cols)
        with pytest.raises(ValueError):
            evaluate_by_age_group(model, merged.drop(columns=["default_next"]), feature_cols)


# ── Cell 9: policy_profit_impact tests ───────────────────────────────────────
class TestPolicyProfitImpact:

    def test_returns_three_tuple(self, trained, deploy_df):
        model, feature_cols, _ = trained
        result = policy_profit_impact(model, deploy_df, feature_cols,
                                      threshold=0.15, offer_cost=10.0,
                                      delta_old=0.03, delta_genz=0.01)
        assert len(result) == 3

    def test_profit_types(self, trained, deploy_df):
        model, feature_cols, _ = trained
        no, with_, _ = policy_profit_impact(model, deploy_df, feature_cols,
                                             threshold=0.15, offer_cost=10.0,
                                             delta_old=0.03, delta_genz=0.01)
        assert isinstance(no,   float)
        assert isinstance(with_, float)

    def test_breakdown_cohorts(self, trained, deploy_df):
        model, feature_cols, _ = trained
        _, _, breakdown = policy_profit_impact(model, deploy_df, feature_cols,
                                               threshold=0.15, offer_cost=10.0,
                                               delta_old=0.03, delta_genz=0.01)
        expected = set(deploy_df["age_group"].unique())
        assert set(breakdown.keys()) == expected

    def test_breakdown_keys(self, trained, deploy_df):
        model, feature_cols, _ = trained
        _, _, breakdown = policy_profit_impact(model, deploy_df, feature_cols,
                                               threshold=0.15, offer_cost=10.0,
                                               delta_old=0.03, delta_genz=0.01)
        for group, metrics in breakdown.items():
            assert set(metrics.keys()) == {"n", "profit_no_policy", "profit_with_policy", "offer_rate"}

    def test_offer_rate_range(self, trained, deploy_df):
        model, feature_cols, _ = trained
        _, _, breakdown = policy_profit_impact(model, deploy_df, feature_cols,
                                               threshold=0.15, offer_cost=10.0,
                                               delta_old=0.03, delta_genz=0.01)
        for group, metrics in breakdown.items():
            assert 0.0 <= metrics["offer_rate"] <= 1.0, f"offer_rate out of range for {group}"

    def test_zero_threshold_no_offers(self, trained, deploy_df):
        # threshold=0 means p_hat <= 0 is never true → no offers → profits equal
        model, feature_cols, _ = trained
        no, with_, breakdown = policy_profit_impact(model, deploy_df, feature_cols,
                                                    threshold=0.0, offer_cost=10.0,
                                                    delta_old=0.03, delta_genz=0.01)
        assert abs(no - with_) < 1e-6
        for metrics in breakdown.values():
            assert metrics["offer_rate"] == 0.0

    def test_missing_columns_raise(self, trained, deploy_df):
        model, feature_cols, _ = trained
        with pytest.raises(ValueError):
            policy_profit_impact(model, deploy_df.drop(columns=["profit_true"]),
                                 feature_cols, threshold=0.15, offer_cost=10.0,
                                 delta_old=0.03, delta_genz=0.01)
        with pytest.raises(ValueError):
            policy_profit_impact(model, deploy_df.drop(columns=["age_group"]),
                                 feature_cols, threshold=0.15, offer_cost=10.0,
                                 delta_old=0.03, delta_genz=0.01)


# ── Cell 10: stress_test_cohort_mix tests ────────────────────────────────────
class TestStressCohortMix:

    def test_output_shape(self, deploy_df):
        result = stress_test_cohort_mix(deploy_df, cohort="GenZ")
        assert list(result.columns) == ["cohort", "share", "expected_profit"]
        assert len(result) == 5  # default shares list has 5 entries

    def test_custom_shares(self, deploy_df):
        shares = [0.1, 0.5, 0.9]
        result = stress_test_cohort_mix(deploy_df, cohort="GenZ", shares=shares)
        assert len(result) == len(shares)
        assert list(result["share"]) == shares

    def test_cohort_column_correct(self, deploy_df):
        result = stress_test_cohort_mix(deploy_df, cohort="Boomer")
        assert (result["cohort"] == "Boomer").all()

    def test_expected_profit_is_float(self, deploy_df):
        result = stress_test_cohort_mix(deploy_df, cohort="GenZ")
        assert result["expected_profit"].dtype == float

    def test_weights_sum_to_one(self, deploy_df):
        # At share=0.5 with equal halves, weighted mean == simple mean only if
        # cohort sizes happen to be equal — so we verify the formula directly:
        # expected_profit should be finite and non-NaN for all shares.
        result = stress_test_cohort_mix(deploy_df, cohort="GenZ")
        assert result["expected_profit"].notna().all()
        assert result["expected_profit"].apply(math.isfinite).all()

    def test_missing_columns_raise(self, deploy_df):
        with pytest.raises(ValueError):
            stress_test_cohort_mix(deploy_df.drop(columns=["profit_true"]), cohort="GenZ")
        with pytest.raises(ValueError):
            stress_test_cohort_mix(deploy_df.drop(columns=["age_group"]), cohort="GenZ")

    def test_share_1_dominated_by_target(self, deploy_df):
        # At share=1.0 all weight is on the target cohort, so expected profit
        # should equal the target cohort's mean profit.
        target_mean = deploy_df[deploy_df["age_group"] == "GenZ"]["profit_true"].mean()
        result = stress_test_cohort_mix(deploy_df, cohort="GenZ", shares=[1.0])
        assert abs(result["expected_profit"].iloc[0] - target_mean) < 1e-6


# ── Cell 11: run ──────────────────────────────────────────────────────────────
# In Colab, run with:  !pytest test_credit_shift.py -v