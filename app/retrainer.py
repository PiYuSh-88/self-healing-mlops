"""
Auto-Retrainer — the self-healing engine of the MLOps pipeline.

This module handles the complete retraining lifecycle:
    1. Load original training data + production prediction logs
    2. Combine them into an enriched dataset
    3. Preprocess (same pipeline as original training)
    4. Train a new RandomForestClassifier
    5. Evaluate on a held-out test set
    6. Save versioned model artifacts (model_v2.pkl, preprocessor_v2.pkl)
    7. Promote new model as the active model

How Retraining Gets Triggered
-----------------------------
• Manual:     POST /retrain
• Automated:  GET /drift  →  drift_detected=true  →  POST /retrain
              (Jenkins/K8s CronJob can chain these two calls)

How Model Replacement Works
---------------------------
1. New model is trained and saved with a version number:
       models/churn_model_v2.pkl
       models/preprocessor_v2.pkl
2. The new model is also saved to the ACTIVE paths:
       models/churn_model.pkl
       models/preprocessor.pkl
3. model.reload_artifacts() hot-swaps the in-memory model
   → zero downtime, no server restart needed.

How Rollback Works (Future)
---------------------------
All versions are kept in the models/ directory:
    models/churn_model_v1.pkl
    models/churn_model_v2.pkl
To rollback: copy the old version back to churn_model.pkl
and call model.reload_artifacts().
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from app.config import (
    TRAINING_DATA_PATH,
    PREDICTION_LOG_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━ VERSION TRACKING ━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_next_version() -> int:
    """
    Determine the next model version number by scanning existing files.

    Looks for files like churn_model_v1.pkl, churn_model_v2.pkl, etc.
    Returns the next version number (e.g., 3 if v1 and v2 exist).
    """
    existing_versions = []
    for f in MODELS_DIR.glob("churn_model_v*.pkl"):
        try:
            version = int(f.stem.split("_v")[-1])
            existing_versions.append(version)
        except ValueError:
            continue

    if not existing_versions:
        # First retraining — backup current model as v1
        return 2

    return max(existing_versions) + 1


def _backup_current_as_v1() -> None:
    """Backup the original model as v1 (only done once)."""
    v1_model = MODELS_DIR / "churn_model_v1.pkl"
    v1_preprocessor = MODELS_DIR / "preprocessor_v1.pkl"

    if not v1_model.exists() and Path(MODEL_PATH).exists():
        shutil.copy2(MODEL_PATH, v1_model)
        logger.info("Backed up original model as: %s", v1_model.name)

    if not v1_preprocessor.exists() and Path(PREPROCESSOR_PATH).exists():
        shutil.copy2(PREPROCESSOR_PATH, v1_preprocessor)
        logger.info("Backed up original preprocessor as: %s", v1_preprocessor.name)


# ━━━━━━━━━━━━━━━━━━━━━ DATA PREPARATION ━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_and_combine_data() -> pd.DataFrame:
    """
    Load original training data and production logs, then combine.

    The production log has a 'prediction_label' column but no 'Churn'
    column, so we map: prediction_label → Churn (Yes/No).
    """
    # ── Original training data ──────────────────────────────────────
    logger.info("Loading training data from: %s", TRAINING_DATA_PATH)
    train_df = pd.read_csv(TRAINING_DATA_PATH)
    train_df.drop("customerID", axis=1, inplace=True, errors="ignore")
    train_df["TotalCharges"] = pd.to_numeric(
        train_df["TotalCharges"], errors="coerce"
    )
    logger.info("  Training data: %d rows", len(train_df))

    # ── Production prediction logs ──────────────────────────────────
    prod_rows = 0
    if PREDICTION_LOG_PATH.exists():
        prod_df = pd.read_csv(PREDICTION_LOG_PATH)

        if len(prod_df) > 0:
            # Map prediction_label back to Churn column
            prod_df["Churn"] = prod_df["prediction_label"].map(
                {"Churn": "Yes", "No Churn": "No"}
            )

            # Keep only the columns that match training data
            train_cols = list(train_df.columns)
            common_cols = [c for c in train_cols if c in prod_df.columns]
            prod_df = prod_df[common_cols]

            # Ensure numeric types match
            prod_df["TotalCharges"] = pd.to_numeric(
                prod_df["TotalCharges"], errors="coerce"
            )
            prod_df["SeniorCitizen"] = pd.to_numeric(
                prod_df["SeniorCitizen"], errors="coerce"
            ).astype(int)

            prod_rows = len(prod_df)
            logger.info("  Production data: %d rows", prod_rows)

            # ── Combine ────────────────────────────────────────────
            combined = pd.concat([train_df, prod_df], ignore_index=True)
            logger.info("  Combined dataset: %d rows", len(combined))
            return combined

    logger.info("  No production data — using training data only")
    return train_df


# ━━━━━━━━━━━━━━━━━━━━━ TRAINING PIPELINE ━━━━━━━━━━━━━━━━━━━━━━━━━


def run_retraining() -> Dict:
    """
    Execute the full retraining pipeline.

    Returns
    -------
    dict with:
        - success: whether retraining completed
        - version: new model version number
        - metrics: accuracy, precision, recall, f1
        - dataset_size: total rows used for training
        - model_path / preprocessor_path: where saved
        - timestamp
    """
    logger.info("=" * 60)
    logger.info("AUTO-RETRAINING — Starting pipeline")
    logger.info("=" * 60)

    start_time = datetime.now(timezone.utc)

    try:
        # ── Step 1: Load and combine data ───────────────────────────
        logger.info("[Step 1/6] Loading and combining data …")
        df = _load_and_combine_data()

        # ── Step 2: Prepare features and target ─────────────────────
        logger.info("[Step 2/6] Preparing features and target …")
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        logger.info("  Numeric features: %s", list(numerical_cols))
        logger.info("  Categorical features: %s", list(categorical_cols))

        # ── Step 3: Build preprocessor ──────────────────────────────
        logger.info("[Step 3/6] Building preprocessor pipeline …")
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        # ── Step 4: Split, preprocess, and train ────────────────────
        logger.info("[Step 4/6] Training new model …")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        new_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )

        new_model.fit(X_train, y_train)

        # ── Step 5: Evaluate ────────────────────────────────────────
        logger.info("[Step 5/6] Evaluating retrained model …")
        y_pred = new_model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
        }

        logger.info("  Accuracy : %.4f", metrics["accuracy"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall   : %.4f", metrics["recall"])
        logger.info("  F1 Score : %.4f", metrics["f1_score"])

        # ── Step 6: Save versioned + active model ───────────────────
        logger.info("[Step 6/6] Saving new model …")

        # Backup original as v1 (first time only)
        _backup_current_as_v1()

        version = _get_next_version()

        # Save versioned copies
        versioned_model = MODELS_DIR / f"churn_model_v{version}.pkl"
        versioned_preprocessor = MODELS_DIR / f"preprocessor_v{version}.pkl"

        joblib.dump(new_model, versioned_model)
        joblib.dump(preprocessor, versioned_preprocessor)
        logger.info("  Saved: %s", versioned_model.name)
        logger.info("  Saved: %s", versioned_preprocessor.name)

        # Overwrite active model (the one the API uses)
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logger.info("  Active model updated: churn_model.pkl")
        logger.info("  Active preprocessor updated: preprocessor.pkl")

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        summary = (
            f"✅ Retraining complete! Model v{version} saved. "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}. "
            f"Took {elapsed:.1f}s."
        )
        logger.info(summary)
        logger.info("=" * 60)

        return {
            "success": True,
            "version": version,
            "metrics": metrics,
            "dataset_size": len(df),
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "model_path": str(versioned_model),
            "preprocessor_path": str(versioned_preprocessor),
            "summary": summary,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": start_time.isoformat(),
        }

    except Exception as exc:
        logger.exception("Retraining failed!")
        return {
            "success": False,
            "version": 0,
            "metrics": {},
            "dataset_size": 0,
            "training_rows": 0,
            "test_rows": 0,
            "model_path": "",
            "preprocessor_path": "",
            "summary": f"❌ Retraining failed: {str(exc)}",
            "elapsed_seconds": 0,
            "timestamp": start_time.isoformat(),
        }
