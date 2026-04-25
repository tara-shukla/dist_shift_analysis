from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import math
from sklearn.ensemble import RandomForestClassifier



def load_data(transactions_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and users CSVs and return (tx, users)."""
    tx = pd.read_csv(transactions_path)
    users = pd.read_csv(users_path)
    return tx, users


_SPEND_COLS = [
    "spend_rent",
    "spend_dining",
    "spend_groceries",
    "spend_travel",
    "spend_rideshare",
    "spend_other",
]

_CATEGORY_MAP = {
    "rideshare ": "rideshare",
    "ride_share": "rideshare",
    "uber/lyft": "rideshare",
    " dining": "dining",
    "restaurants": "dining",
    "groceries ": "groceries",
    "travel ": "travel",
    "flights": "travel",
    "other ": "other",
    "other": "other",
    "rent": "rent",
    "dining": "dining",
    "groceries": "groceries",
    "travel": "travel",
    "rideshare": "rideshare",
}


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions DataFrame per the homework spec.

    Key requirements:
      1) Drop exact duplicate rows.
      2) Normalize primary_category: strip whitespace, lowercase, map synonyms.
      3) Missing values:
         - spend_* NaNs -> 0
         - monthly_spend NaN -> sum(spend_*)
         - payment_rate NaN -> median within age_group (computed from *input tx*)
         - apr NaN -> median within age_group (computed from *input tx*)
      4) Clip spend_* at 99.5th percentile (computed on *input tx*).
      5) payment_rate in [0,1]; apr in [0,0.40]
      6) user_id/month ints; age_group str
    """
    df = tx.copy()
        
    df = df.drop_duplicates()
    df["primary_category"] = df["primary_category"].str.strip().str.lower().map(_CATEGORY_MAP)

    df[_SPEND_COLS] = df[_SPEND_COLS].fillna(0)

    df["monthly_spend"] = df["monthly_spend"].fillna(df[_SPEND_COLS].sum(axis=1))
    
    df["payment_rate"] = df["payment_rate"].fillna(df.groupby("age_group")["payment_rate"].transform("median"))
    df["apr"] = df["apr"].fillna(df.groupby("age_group")["apr"].transform("median"))

    df[_SPEND_COLS] = df[_SPEND_COLS].clip(upper=tx[_SPEND_COLS].quantile(0.995), axis=1)

    df["payment_rate"] = df["payment_rate"].clip(lower=0, upper=1)
    df["apr"] = df["apr"].clip(lower=0, upper=0.40)

    df['user_id'] = df['user_id'].astype(int)
    df['month'] = df['month'].astype(int)
    df['age_group'] = df['age_group'].astype(str)

    return df


def add_user_features(tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user-level columns from users into tx on user_id WITHOUT changing number of rows in tx.
    Deduplicate users on user_id before merging.
    """
    before = len(tx)
    
    cols_to_add = ["user_id"] + [c for c in users.columns if c not in tx.columns]
    merged = tx.merge(
        users[cols_to_add].drop_duplicates(subset="user_id"),
        on="user_id",
        how="left",
    )    
    after = len(merged)
    if after != before:
        raise ValueError(f"Row count changed after merge: {before} -> {after}")
    return merged


HARD_LEAKY_COLS = {
    "default_next",
    "profit_true",
    "chargeoff_loss",
    "pd_default",
    "balance_next",
    "interest_income",
    "interchange_rev",
    "reward_cost",
    "funding_cost",
}

LEAKY_SUBSTRINGS = [
    "default",
    "profit",
    "chargeoff",
    "interchange",
    "interest_income",
    "reward_cost",
    "funding_cost",
    "pd_",
    "label",
]


def _is_leaky(col: str) -> bool:
    c = col.lower()
    if col in HARD_LEAKY_COLS:
        return True
    return any(sub in c for sub in LEAKY_SUBSTRINGS)


def train_default_model(train_df: pd.DataFrame):
    """
    Train a Scikit-Learn Classifier to predict default_next.

    Returns
    -------
    model : fitted sklearn Pipeline
    feature_cols : list[str] raw feature columns used

    Constraints
    -----------
    - Do NOT use leaky columns (anything matching _is_leaky()).
    """
    df = train_df.copy()
    if "default_next" not in df.columns:
        raise ValueError("Missing target column default_next")

    feature_cols = [c for c in df.columns if c != "default_next" and not _is_leaky(c) and c!='split']

    ## TODO: Implement it
    # for col in feature_cols:
    #     if df[col].dtype == "object" or df[col].dtype == "category":
    #         df[col] = OneHotEncoder(df[col].astype("category"))
    
    X = df[feature_cols]
    y = df['default_next']

    categoricals = [c for c in feature_cols if X[c].dtype == object or str(X[c].dtype) == "category"]
    numericals = [c for c in feature_cols if c not in categoricals]

    process = ColumnTransformer(
        transformers = [
        ('categorical_values', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), categoricals),
        ('numerical_values', SimpleImputer(strategy="mean"), numericals)
       ]
    )

    params = {
    "n_estimators": 100,
    "max_depth": 2,
    "subsample": 0.6,
    "learning_rate": 0.01,
    "min_samples_leaf": 100,
    "random_state": 3,
    }
    # model = ensemble.GradientBoostingClassifier(**params)
    model = Pipeline(steps=[('preprocessor', process), ('classifier', ensemble.GradientBoostingClassifier(**params))])
    model.fit(X, y)


    return model, feature_cols


def evaluate_by_age_group(model, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Return per-age-group metrics: {age_group: {auc, brier, n}}."""
    if "age_group" not in df.columns:
        raise ValueError("df must contain age_group")
    if "default_next" not in df.columns:
        raise ValueError("df must contain default_next")

    ## TODO: Implement it:
    out: Dict[str, Dict[str, float]] = {}

    df = df.copy()
    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    df['p_hat'] = p_hat


    # df = df.group_by('age_group')

    def _compute_metrics(age_df):
      y_true = age_df['default_next']
      y_pred = age_df['p_hat']

      auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
      brier = sklearn.metrics.brier_score_loss(y_true, y_pred)
      n = len(age_df)

      out[age_df['age_group'].iloc[0]] = {
          'auc': auc,
          'brier': brier,
          'n': n
      }
      
    df.groupby('age_group').apply(lambda age_df: _compute_metrics(age_df))
    return out

def compute_uplift_score(model, df: pd.DataFrame, feature_cols: List[str], top_frac: float = 0.10) -> float:
    """
    Compute the top-fraction uplift score for a fitted default model.

    Parameters
    ----------
    model : fitted sklearn-like classifier
        Must support ``predict_proba(df[feature_cols])[:, 1]``.
    df : pd.DataFrame
        Evaluation data containing ``default_next`` and the raw feature columns.
    feature_cols : list[str]
        Raw feature columns consumed by the model.
    top_frac : float, default 0.10
        Fraction of rows to target. Must lie in ``(0, 1]``.

    Returns
    -------
    float
        Extra number of defaults captured in the targeted slice relative to a
        random policy with the same targeting rate.
    """
    if "default_next" not in df.columns:
        raise ValueError("df must contain default_next")
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError("top_frac must be in (0, 1]")
    
    df = df.copy()

    n = len(df)
    if n == 0:
        return 0.0
      
    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    df['p_hat'] = p_hat

    k = math.ceil(top_frac * n)

    df.sort_values(by='p_hat', ascending=False, inplace=True)
    df_top_k = df.iloc[:k]

    denominator = df['default_next'].mean()
    numerator = df_top_k['default_next'].mean()

    if denominator == 0:
        return 0.0
    
    uplift_score = numerator/denominator

    return uplift_score


def policy_profit_impact(
    model,
    deploy_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    offer_cost: float,
    delta_old: float,
    delta_genz: float,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    Stylized expected-profit simulation on deployment rows.

    Policy:
      - compute p_hat = P(default_next=1 | features)
      - offer incentive if p_hat <= threshold; cost offer_cost immediately
      - reduces default probability by delta (absolute), with cohort-specific values:
          delta_old for non-GenZ
          delta_genz for GenZ

    Returns:
      (profit_no_policy, profit_with_policy, breakdown)
    """
    df = deploy_df.copy()
    if "profit_true" not in df.columns:
        raise ValueError("deploy_df must contain profit_true")
    if "age_group" not in df.columns:
        raise ValueError("deploy_df must contain age_group")

    # 1
    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    df['p_hat'] = p_hat

    # 2
    df['incentive_offer'] = np.where(df['p_hat'] <= threshold, 1, 0)
    
    # 3
    # df['default_cost'] = df['default_cost']*delta_genz if df['age_group'] == 'GenZ' else df['default_cost']*delta_old
    df['delta'] = np.where(df['age_group'] == 'GenZ', delta_genz, delta_old)

    # 4
    df['balance'] = df['statement_balance']
    if 'balance_new' in df.columns:
        df['balance'] = df['balance_new']
    
    # 5
    df['avoided_loss'] = df['balance'] * df['incentive_offer'] *0.9 * df['delta']

    # 6
    df['policy_cost'] = offer_cost * df['incentive_offer']

    # 7
    profit_no = df['profit_true'].sum()
    # 8
    # profit_with = sum(df['profit_true']) + sum(df['avoided_loss']) - sum(df['policy_cost'])
    # profit_with = (df['profit_true'] + df['avoided_loss'].sum() - df['policy_cost']).sum()
    profit_with = (df['profit_true'] + df['avoided_loss'] - df['policy_cost']).sum()


    # 9
    breakdown: Dict[str, Dict[str, float]] = {}
    for age_group in df['age_group'].unique():
        breakdown[age_group] = {}
        breakdown[age_group]['n'] = len(df[df['age_group'] == age_group])
        breakdown[age_group]['profit_no_policy'] = sum(df[df['age_group'] == age_group]['profit_true'])
        breakdown[age_group]['profit_with_policy'] = sum(df[df['age_group'] == age_group]['profit_true'] + df[df['age_group'] == age_group]['avoided_loss']) - sum(df[df['age_group'] == age_group]['policy_cost'])
        breakdown[age_group]['offer_rate'] = sum(df[df['age_group'] == age_group]['incentive_offer'])/breakdown[age_group]['n']
    return profit_no, profit_with, breakdown


def stress_test_cohort_mix(
    df: pd.DataFrame,
    cohort: str,
    shares: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Stress test: reweight rows to simulate different cohort mixes (target cohort share varies).

    Returns DataFrame with columns: cohort, share, expected_profit
    """
    if shares is None:
        shares = [0.10, 0.20, 0.30, 0.45, 0.60]

    if "age_group" not in df.columns or "profit_true" not in df.columns:
        raise ValueError("df must contain age_group and profit_true")

    rows = []
    # 1
    target_cohort = df[df['age_group'] == cohort]
    n_target = len(target_cohort)

    non_target = df[df['age_group'] != cohort]
    n_non_target = len(non_target)

    # 5
    fallback = df['profit_true'].mean()

    # 3
    for s in shares: 
        if n_target == 0:
            rows.append({
                'cohort': cohort,
                'share': s,
                'expected_profit': fallback
            })
            continue
      
        #weights = np.where(df['age_group'] == cohort, s/n_target, (1-s)/n_non_target)
        
        target_w = s / n_target
        non_target_w = (1 - s) / n_non_target if n_non_target > 0 else 0.0
        weights = np.where(df['age_group'] == cohort, target_w, non_target_w)
        
        # 4
        expected_profit = (df['profit_true'] * weights).sum()
        rows.append({
            'cohort': cohort,
            'share': s,
            'expected_profit': expected_profit
        })
    
    return pd.DataFrame(rows)
    