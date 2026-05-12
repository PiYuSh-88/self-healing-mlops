"""
Configuration settings for the Churn Prediction API.

All paths and app-level constants are centralized here
so they're easy to update when deploying or dockerizing.
"""

import os
from pathlib import Path

# ── Project root is one level above the `app/` package ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Model artifacts ─────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "churn_model.pkl"),
)

PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    str(PROJECT_ROOT / "models" / "preprocessor.pkl"),
)

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models"))
)

# ── Data paths ──────────────────────────────────────────────────────
DATA_DIR = Path(
    os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
)

TRAINING_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# ── Prediction logs ────────────────────────────────────────────────
LOGS_DIR = Path(
    os.getenv("LOGS_DIR", str(PROJECT_ROOT / "logs"))
)

PREDICTION_LOG_PATH = LOGS_DIR / "prediction_log.csv"

# ── API metadata ────────────────────────────────────────────────────
API_TITLE = "Churn Prediction API"
API_DESCRIPTION = (
    "A production-ready inference service for predicting customer churn. "
    "Part of the Self-Healing MLOps pipeline with drift detection "
    "and auto-retraining."
)
API_VERSION = "1.0.0"

# ── Drift detection ────────────────────────────────────────────────
DRIFT_THRESHOLD = float(
    os.getenv("DRIFT_THRESHOLD", "0.5")
)
