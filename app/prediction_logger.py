"""
Prediction Logger — logs every prediction to CSV for future drift detection.

This module provides a reusable, thread-safe logging utility that captures:
  • All input features
  • Model prediction (0/1)
  • Prediction label (Churn / No Churn)
  • Churn probability
  • No-churn probability
  • Timestamp (UTC)

The CSV log file is designed to be consumed by:
  1. Drift detection module — compare live distributions vs training baseline
  2. Monitoring dashboards — track prediction trends over time
  3. Auto-retraining pipeline — use logged data as new training data
"""

import csv
import logging
from datetime import datetime, timezone
from threading import Lock

from app.config import PREDICTION_LOG_PATH, LOGS_DIR

logger = logging.getLogger(__name__)

# ── Thread-safe file lock (uvicorn can serve concurrent requests) ───
_write_lock = Lock()

# ── Column order for the CSV log ────────────────────────────────────
# Input features come first (same order as training), then predictions.
INPUT_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]

PREDICTION_COLUMNS = [
    "prediction",
    "prediction_label",
    "churn_probability",
    "no_churn_probability",
    "timestamp",
]

ALL_COLUMNS = INPUT_FEATURE_COLUMNS + PREDICTION_COLUMNS


def _ensure_log_file() -> None:
    """Create the logs directory and CSV header if they don't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTION_LOG_PATH.exists():
        with open(PREDICTION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ALL_COLUMNS)
        logger.info("Created prediction log file: %s", PREDICTION_LOG_PATH)


def log_prediction(
    input_data: dict,
    prediction: int,
    prediction_label: str,
    churn_probability: float,
    no_churn_probability: float,
) -> None:
    """
    Append a single prediction record to the CSV log.

    Parameters
    ----------
    input_data : dict
        Raw customer features (from the API request body).
    prediction : int
        Model output — 0 (No Churn) or 1 (Churn).
    prediction_label : str
        Human-readable label — "Churn" or "No Churn".
    churn_probability : float
        P(Churn).
    no_churn_probability : float
        P(No Churn).
    """
    _ensure_log_file()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Build row in the exact column order
    row = []
    for col in INPUT_FEATURE_COLUMNS:
        row.append(input_data.get(col, ""))

    row.extend([
        prediction,
        prediction_label,
        churn_probability,
        no_churn_probability,
        timestamp,
    ])

    with _write_lock:
        with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    logger.debug("Prediction logged at %s", timestamp)


def get_log_stats() -> dict:
    """
    Return basic stats about the prediction log.

    Useful for the /health or a future /stats endpoint.
    """
    if not PREDICTION_LOG_PATH.exists():
        return {"total_predictions": 0, "log_file": str(PREDICTION_LOG_PATH)}

    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as f:
        # Subtract 1 for the header row
        line_count = sum(1 for _ in f) - 1

    return {
        "total_predictions": max(line_count, 0),
        "log_file": str(PREDICTION_LOG_PATH),
        "log_size_bytes": PREDICTION_LOG_PATH.stat().st_size,
    }
