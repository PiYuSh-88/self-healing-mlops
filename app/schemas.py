"""
Pydantic schemas for request validation and response serialization.

These schemas define the exact JSON shape the API expects and returns.
FastAPI uses them to auto-generate OpenAPI docs AND validate every request.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class CustomerData(BaseModel):
    """
    Input schema — mirrors every feature the trained model expects.

    Notes
    -----
    • `customerID` is accepted but ignored during prediction.
    • `TotalCharges` is Optional because the raw dataset has blanks
      for zero-tenure customers; the preprocessor handles imputation.
    """

    customerID: Optional[str] = Field(
        default=None,
        description="Unique customer identifier (ignored by the model)",
    )
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="1 = senior, 0 = not senior")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Months the customer has stayed")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(
        ..., description="Yes, No, or No phone service"
    )
    InternetService: str = Field(
        ..., description="DSL, Fiber optic, or No"
    )
    OnlineSecurity: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    OnlineBackup: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    DeviceProtection: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    TechSupport: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    StreamingTV: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    StreamingMovies: str = Field(
        ..., description="Yes, No, or No internet service"
    )
    Contract: str = Field(
        ..., description="Month-to-month, One year, or Two year"
    )
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ...,
        description=(
            "Electronic check, Mailed check, "
            "Bank transfer (automatic), or Credit card (automatic)"
        ),
    )
    MonthlyCharges: float = Field(
        ..., description="Monthly charge amount in dollars"
    )
    TotalCharges: Optional[float] = Field(
        default=None, description="Total charges to date (can be blank)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema returned by the /predict endpoint.
    """

    customer_id: Optional[str] = Field(
        default=None, description="Echo of the input customer ID"
    )
    prediction: str = Field(
        ..., description="Churn or No Churn"
    )
    churn_probability: float = Field(
        ..., description="Probability of churn (0.0 – 1.0)"
    )
    no_churn_probability: float = Field(
        ..., description="Probability of NOT churning (0.0 – 1.0)"
    )


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    total_predictions_logged: int = 0


class DriftResponse(BaseModel):
    """Response for the /drift endpoint."""

    drift_detected: bool = Field(
        ..., description="True if overall drift exceeds threshold"
    )
    overall_drift_score: float = Field(
        ..., description="Mean drift score across all features"
    )
    threshold: float = Field(
        ..., description="Configured drift threshold"
    )
    drifted_features: List[str] = Field(
        default=[], description="Features that exceeded the threshold"
    )
    total_features_checked: int = 0
    total_features_drifted: int = 0
    feature_drift: Dict[str, Any] = Field(
        default={}, description="Per-feature drift details"
    )
    summary: str = Field(
        ..., description="Human-readable drift summary"
    )
    training_rows: int = 0
    production_rows: int = 0
    timestamp: str = ""


class RetrainResponse(BaseModel):
    """Response for the /retrain endpoint."""

    model_config = {"protected_namespaces": ()}

    success: bool = Field(
        ..., description="True if retraining completed successfully"
    )
    version: int = Field(
        ..., description="New model version number"
    )
    metrics: Dict[str, float] = Field(
        default={}, description="Accuracy, precision, recall, f1 of new model"
    )
    dataset_size: int = Field(
        ..., description="Total rows used (training + production logs)"
    )
    training_rows: int = 0
    test_rows: int = 0
    model_path: str = ""
    preprocessor_path: str = ""
    summary: str = Field(
        ..., description="Human-readable result summary"
    )
    elapsed_seconds: float = 0.0
    timestamp: str = ""
