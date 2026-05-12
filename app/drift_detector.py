"""
Drift Detection Module — the self-healing brain of the MLOps system.

Compares feature distributions between TRAINING data and PRODUCTION
(logged prediction) data to detect data drift.

How Drift Is Calculated
-----------------------
For each NUMERIC feature (tenure, MonthlyCharges, TotalCharges, SeniorCitizen):

    1. Compute the MEAN of the feature in training data vs production data.
    2. Compute the STD  of the feature in training data vs production data.
    3. Drift score = |mean_train - mean_prod| / (std_train + 1e-10)
       → This is essentially "how many training-std-devs has the mean shifted?"
       → A score > 1.0 means the production mean has moved more than one
         standard deviation away from the training mean — a strong signal.

For each CATEGORICAL feature (gender, Contract, PaymentMethod, etc.):

    1. Compute the FREQUENCY DISTRIBUTION (proportions) in training vs production.
    2. For each category, compute |proportion_train - proportion_prod|.
    3. Drift score = SUM of all absolute proportion differences / 2
       → This gives a value between 0.0 (identical distributions)
         and 1.0 (completely different distributions).
       → Known as "Total Variation Distance" — beginner-friendly but rigorous.

Overall Drift:
    overall_drift_score = mean(all per-feature drift scores)

Drift Alert:
    If overall_drift_score > DRIFT_THRESHOLD → DRIFT DETECTED → retrain!

Future Integration
------------------
• Jenkins: Schedule this as a cron job or post-deployment check.
• Kubernetes: Run as a CronJob that hits the /drift endpoint.
• Auto-Retraining: When drift is detected, trigger the retraining pipeline
  (retrain on fresh data from prediction logs + original training data).
"""

import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import numpy as np

from app.config import TRAINING_DATA_PATH, PREDICTION_LOG_PATH, DRIFT_THRESHOLD

logger = logging.getLogger(__name__)

# ── Features to monitor ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# ━━━━━━━━━━━━━━━━━━━━━ DATA LOADING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_training_data() -> pd.DataFrame:
    """Load the original training dataset."""
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}"
        )

    df = pd.read_csv(TRAINING_DATA_PATH)

    # Match the preprocessing done during training
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    logger.info("Loaded training data: %d rows", len(df))
    return df


def _load_production_data() -> pd.DataFrame:
    """Load the prediction log (production data)."""
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTION_LOG_PATH}"
        )

    df = pd.read_csv(PREDICTION_LOG_PATH)

    # Ensure numeric columns are properly typed
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded production data: %d rows", len(df))
    return df


# ━━━━━━━━━━━━━━━━━━━━━ DRIFT CALCULATIONS ━━━━━━━━━━━━━━━━━━━━━━━━


def _numeric_drift(
    train_series: pd.Series,
    prod_series: pd.Series,
) -> Dict:
    """
    Calculate drift for a single numeric feature.

    Method: Normalized mean difference
        drift_score = |mean_train - mean_prod| / (std_train + epsilon)

    Returns a dict with stats and drift score.
    """
    train_mean = float(train_series.mean())
    train_std = float(train_series.std())
    prod_mean = float(prod_series.mean())
    prod_std = float(prod_series.std())

    # Avoid division by zero
    epsilon = 1e-10
    drift_score = abs(train_mean - prod_mean) / (train_std + epsilon)

    return {
        "type": "numeric",
        "train_mean": round(train_mean, 4),
        "train_std": round(train_std, 4),
        "prod_mean": round(prod_mean, 4),
        "prod_std": round(prod_std, 4),
        "drift_score": round(drift_score, 4),
    }


def _categorical_drift(
    train_series: pd.Series,
    prod_series: pd.Series,
) -> Dict:
    """
    Calculate drift for a single categorical feature.

    Method: Total Variation Distance (TVD)
        drift_score = sum(|p_train - p_prod|) / 2

    Where p_train and p_prod are the proportion vectors.

    Returns a dict with distribution comparison and drift score.
    """
    # Get proportions (normalize to sum=1)
    train_dist = train_series.value_counts(normalize=True)
    prod_dist = prod_series.value_counts(normalize=True)

    # Union of all categories seen in either dataset
    all_categories = set(train_dist.index) | set(prod_dist.index)

    total_diff = 0.0
    category_diffs = {}

    for cat in sorted(all_categories):
        p_train = train_dist.get(cat, 0.0)
        p_prod = prod_dist.get(cat, 0.0)
        diff = abs(p_train - p_prod)
        total_diff += diff
        category_diffs[str(cat)] = {
            "train_pct": round(float(p_train) * 100, 2),
            "prod_pct": round(float(p_prod) * 100, 2),
            "diff_pct": round(float(diff) * 100, 2),
        }

    drift_score = total_diff / 2  # TVD

    return {
        "type": "categorical",
        "categories": category_diffs,
        "drift_score": round(drift_score, 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━ MAIN DETECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_drift_detection() -> Dict:
    """
    Run full drift detection: compare training vs production data.

    Returns
    -------
    dict with:
        - feature_drift: per-feature drift scores and stats
        - overall_drift_score: mean of all feature drift scores
        - drift_detected: True if overall score > threshold
        - threshold: the configured drift threshold
        - summary: human-readable summary string
        - timestamp: when the check was performed
        - training_rows / production_rows: data sizes
    """
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION — Starting analysis")
    logger.info("=" * 60)

    # ── Load data ───────────────────────────────────────────────────
    train_df = _load_training_data()
    prod_df = _load_production_data()

    if len(prod_df) == 0:
        return {
            "drift_detected": False,
            "overall_drift_score": 0.0,
            "threshold": DRIFT_THRESHOLD,
            "message": "No production data available yet.",
            "training_rows": len(train_df),
            "production_rows": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Analyze each feature ────────────────────────────────────────
    feature_drift = {}
    all_scores = []

    # Numeric features
    for feature in NUMERIC_FEATURES:
        if feature in train_df.columns and feature in prod_df.columns:
            train_col = train_df[feature].dropna()
            prod_col = prod_df[feature].dropna()

            if len(prod_col) > 0:
                result = _numeric_drift(train_col, prod_col)
                feature_drift[feature] = result
                all_scores.append(result["drift_score"])

                status = "⚠️  DRIFT" if result["drift_score"] > DRIFT_THRESHOLD else "✅ OK"
                logger.info(
                    "  [NUMERIC]  %-18s | score: %.4f | %s",
                    feature, result["drift_score"], status,
                )

    # Categorical features
    for feature in CATEGORICAL_FEATURES:
        if feature in train_df.columns and feature in prod_df.columns:
            train_col = train_df[feature].dropna()
            prod_col = prod_df[feature].dropna()

            if len(prod_col) > 0:
                result = _categorical_drift(train_col, prod_col)
                feature_drift[feature] = result
                all_scores.append(result["drift_score"])

                status = "⚠️  DRIFT" if result["drift_score"] > DRIFT_THRESHOLD else "✅ OK"
                logger.info(
                    "  [CATEG]    %-18s | score: %.4f | %s",
                    feature, result["drift_score"], status,
                )

    # ── Overall score ───────────────────────────────────────────────
    overall_score = float(np.mean(all_scores)) if all_scores else 0.0
    drift_detected = overall_score > DRIFT_THRESHOLD

    # Count how many individual features are drifting
    drifted_features = [
        name for name, info in feature_drift.items()
        if info["drift_score"] > DRIFT_THRESHOLD
    ]

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("-" * 60)
    if drift_detected:
        summary = (
            f"🚨 DRIFT DETECTED! Overall score: {overall_score:.4f} "
            f"(threshold: {DRIFT_THRESHOLD}). "
            f"{len(drifted_features)}/{len(all_scores)} features drifted. "
            f"Recommendation: RETRAIN the model."
        )
        logger.warning(summary)
    else:
        summary = (
            f"✅ No significant drift. Overall score: {overall_score:.4f} "
            f"(threshold: {DRIFT_THRESHOLD}). "
            f"Model is performing within expected data distribution."
        )
        logger.info(summary)

    logger.info("=" * 60)

    return {
        "drift_detected": drift_detected,
        "overall_drift_score": round(overall_score, 4),
        "threshold": DRIFT_THRESHOLD,
        "drifted_features": drifted_features,
        "total_features_checked": len(all_scores),
        "total_features_drifted": len(drifted_features),
        "feature_drift": feature_drift,
        "summary": summary,
        "training_rows": len(train_df),
        "production_rows": len(prod_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
