"""
main.py  -  FastAPI churn prediction API
"""
from __future__ import annotations
import json, logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from data_processing import ALL_FEATURES
from explainability import get_top_risk_factors, load_global_feature_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR     = Path(__file__).parent / "artifacts"
MODEL_PATH        = ARTIFACTS_DIR / "xgb_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_PATH      = ARTIFACTS_DIR / "metrics.json"

app_state: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts ...")
    try:
        app_state["model"]        = joblib.load(MODEL_PATH)
        app_state["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        try:
            app_state["feature_names"] = list(app_state["preprocessor"].get_feature_names_out())
        except Exception as e:
            raise RuntimeError(f"Cannot read feature names - re-run train.py. Detail: {e}") from e
        with open(METRICS_PATH) as f:
            app_state["metrics"] = json.load(f)
        app_state["feature_importance"] = load_global_feature_importance()
        logger.info("Artifacts loaded.")
    except FileNotFoundError as exc:
        raise RuntimeError("Artifacts missing - run train.py first.") from exc
    yield
    app_state.clear()

app = FastAPI(title="ChurnGuard API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class CustomerFeatures(BaseModel):
    tenure_months: float       = Field(..., ge=0, le=120)
    monthly_charges: float     = Field(..., ge=0, le=500)
    total_charges: float       = Field(..., ge=0)
    usage_frequency: float     = Field(..., ge=0, le=31)
    support_calls: float       = Field(..., ge=0, le=20)
    num_products: float        = Field(..., ge=1, le=10)
    login_days_last_30: float  = Field(..., ge=0, le=30)
    avg_session_minutes: float = Field(..., ge=0)
    contract_type: str         = Field(...)
    payment_method: str        = Field(...)
    internet_service: str      = Field(...)
    has_tech_support: str      = Field(...)
    has_online_backup: str     = Field(...)

    @field_validator("contract_type")
    @classmethod
    def v_contract(cls, v):
        if v not in {"Month-to-Month","One Year","Two Year"}: raise ValueError("invalid")
        return v
    @field_validator("payment_method")
    @classmethod
    def v_payment(cls, v):
        if v not in {"Credit Card","Bank Transfer","Electronic Check","Mailed Check"}: raise ValueError("invalid")
        return v
    @field_validator("internet_service")
    @classmethod
    def v_internet(cls, v):
        if v not in {"Fiber Optic","DSL","No"}: raise ValueError("invalid")
        return v
    @field_validator("has_tech_support","has_online_backup")
    @classmethod
    def v_yesno(cls, v):
        if v not in {"Yes","No"}: raise ValueError("must be Yes or No")
        return v

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    risk_factors: list[dict[str, Any]]

class MetricsResponse(BaseModel):
    model_metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]

def _risk(prob):
    return "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    try:
        row = pd.DataFrame([customer.model_dump()])
        X   = app_state["preprocessor"].transform(row)
        prob = float(app_state["model"].predict_proba(X)[0, 1])
        risk_factors = get_top_risk_factors(
            feature_names=app_state["feature_names"],
            feature_values=X[0],
            feature_importances=app_state["model"].feature_importances_,
            top_n=3,
        )
        return PredictionResponse(
            churn_probability=round(prob,4),
            churn_prediction=prob>=0.50,
            risk_level=_risk(prob),
            risk_factors=risk_factors,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(
        model_metrics=app_state["metrics"],
        feature_importance=app_state["feature_importance"],
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
