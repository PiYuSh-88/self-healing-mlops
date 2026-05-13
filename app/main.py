"""
FastAPI application entry point.

Endpoints
---------
GET  /           → welcome message
GET  /health     → readiness check (includes log stats)
POST /predict    → churn prediction (with automatic logging)
GET  /drift      → run drift detection analysis
POST /retrain    → auto-retrain the model on fresh data
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.schemas import CustomerData, PredictionResponse, HealthResponse, DriftResponse, RetrainResponse
from app import model
from app import prediction_logger
from app import drift_detector
from app import retrainer

from prometheus_fastapi_instrumentator import Instrumentator

# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts before the first request; clean up on shutdown."""
    logger.info("Starting up — loading model artifacts …")
    try:
        model.load_artifacts()
        logger.info("All artifacts loaded. API is ready.")
    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        raise
    yield
    logger.info("Shutting down — goodbye!")


# ── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# ── Prometheus Instrumentation ──────────────────────────────────────
# Exposes the /metrics endpoint to track requests, latency, and errors
Instrumentator().instrument(app).expose(app)

# ── CORS — allow the React frontend to call this API ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━ ROUTES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/", tags=["General"])
async def root():
    """Welcome message — handy for a quick browser check."""
    return {
        "message": "Welcome to the Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Readiness probe — returns load status and prediction log stats."""
    model_ok, preprocessor_ok = model.is_ready()
    status = "healthy" if (model_ok and preprocessor_ok) else "unhealthy"
    log_stats = prediction_logger.get_log_stats()
    return HealthResponse(
        status=status,
        model_loaded=model_ok,
        preprocessor_loaded=preprocessor_ok,
        total_predictions_logged=log_stats["total_predictions"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """
    Predict whether a customer will churn.

    Send customer attributes as JSON; receive a prediction
    with class probabilities. Every request is automatically
    logged for drift detection and monitoring.
    """
    try:
        logger.info(
            "Prediction request received for customer: %s",
            customer.customerID or "N/A",
        )

        input_data = customer.model_dump()

        prediction, churn_prob, no_churn_prob = model.predict(input_data)

        label = "Churn" if prediction == 1 else "No Churn"

        # ── Log the prediction for drift detection & monitoring ─────
        prediction_logger.log_prediction(
            input_data=input_data,
            prediction=prediction,
            prediction_label=label,
            churn_probability=churn_prob,
            no_churn_probability=no_churn_prob,
        )

        return PredictionResponse(
            customer_id=customer.customerID,
            prediction=label,
            churn_probability=churn_prob,
            no_churn_probability=no_churn_prob,
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(exc)}",
        )


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
async def check_drift():
    """
    Run drift detection analysis.

    Compares feature distributions between the original training
    data and production prediction logs. Returns per-feature
    drift scores and an overall drift assessment.

    Can be called manually or scheduled via Jenkins/K8s CronJob.
    """
    try:
        result = drift_detector.run_drift_detection()
        return DriftResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Drift detection failed")
        raise HTTPException(
            status_code=500,
            detail=f"Drift detection error: {str(exc)}",
        )


@app.post("/retrain", response_model=RetrainResponse, tags=["MLOps"])
async def retrain_model():
    """
    Trigger the auto-retraining pipeline.

    Combines original training data with new production logs,
    retrains the model, saves a new versioned copy, and hot-swaps
    it into the active API memory.

    Returns the new model metrics and version.
    """
    try:
        result = retrainer.run_retraining()
        
        if result["success"]:
            # Hot-reload the new model into memory
            model.reload_artifacts()
            return RetrainResponse(**result)
        else:
            raise HTTPException(
                status_code=500, 
                detail=result["summary"]
            )
            
    except Exception as exc:
        logger.exception("Retraining pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Retraining error: {str(exc)}",
        )
