"""
Model loading and inference logic.

This module is the ONLY place that touches joblib / sklearn.
Everything else in the app talks to it through `predict()`.
"""

import logging
import joblib
import pandas as pd
import numpy as np
from typing import Tuple

from app.config import MODEL_PATH, PREPROCESSOR_PATH

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once at startup) ────────────────
_model = None
_preprocessor = None


def load_artifacts() -> None:
    """
    Load the trained model and preprocessor from disk.

    Called once during application startup (see main.py lifespan).
    Raises FileNotFoundError if either .pkl file is missing.
    """
    global _model, _preprocessor

    logger.info("Loading model from: %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully — type: %s", type(_model).__name__)

    logger.info("Loading preprocessor from: %s", PREPROCESSOR_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info(
        "Preprocessor loaded successfully — type: %s",
        type(_preprocessor).__name__,
    )


def is_ready() -> Tuple[bool, bool]:
    """Return (model_loaded, preprocessor_loaded) status flags."""
    return _model is not None, _preprocessor is not None


def reload_artifacts() -> None:
    """
    Hot-reload model artifacts from disk (zero downtime).

    Called after retraining to swap in the new model without
    restarting the server. The global references are updated
    atomically so in-flight requests won't break.
    """
    logger.info("Hot-reloading model artifacts …")
    load_artifacts()
    logger.info("Model hot-reload complete.")


def predict(data: dict) -> Tuple[int, float, float]:
    """
    Run a single prediction.

    Parameters
    ----------
    data : dict
        Customer feature dictionary (must match training schema).

    Returns
    -------
    prediction : int
        0 = No Churn, 1 = Churn
    churn_prob : float
        Probability of class 1 (churn).
    no_churn_prob : float
        Probability of class 0 (no churn).

    Raises
    ------
    RuntimeError
        If artifacts haven't been loaded yet.
    """
    if _model is None or _preprocessor is None:
        raise RuntimeError(
            "Model artifacts not loaded. Call load_artifacts() first."
        )

    # ── Build a single-row DataFrame identical to training input ────
    # Remove customerID — the model was never trained on it.
    features = {k: v for k, v in data.items() if k != "customerID"}

    df = pd.DataFrame([features])

    # TotalCharges can arrive as None (mirrors blanks in the CSV).
    # Convert to NaN so the preprocessor's SimpleImputer handles it.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    logger.debug("Input DataFrame shape: %s", df.shape)
    logger.debug("Input columns: %s", list(df.columns))

    # ── Preprocess and predict ──────────────────────────────────────
    X = _preprocessor.transform(df)

    prediction = int(_model.predict(X)[0])
    probabilities = _model.predict_proba(X)[0]

    no_churn_prob = float(np.round(probabilities[0], 4))
    churn_prob = float(np.round(probabilities[1], 4))

    logger.info(
        "Prediction: %s | churn_prob=%.4f | no_churn_prob=%.4f",
        prediction,
        churn_prob,
        no_churn_prob,
    )

    return prediction, churn_prob, no_churn_prob
