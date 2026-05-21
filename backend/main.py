"""
main.py — Indian Market Customer Churn Prediction API
FastAPI backend for Render deployment.
Endpoints: POST /train, POST /predict/batch, POST /explain
"""
from __future__ import annotations

import io
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Indian Churn Prediction API",
    description="Production-grade churn prediction for Indian SaaS/subscription market",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  IN-MEMORY STATE (stateless per request for /predict & /explain)
# ─────────────────────────────────────────────
model_state: dict[str, Any] = {}

# ─────────────────────────────────────────────
#  FEATURE SCHEMA
# ─────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "monthly_charges_inr",
    "tenure_months",
    "payment_failures_last_3m",
    "support_tickets_open",
    "usage_drop_rate_pct",       # % drop in usage vs prev month
    "account_velocity_score",    # logins / actions per month
    "num_products_subscribed",
    "days_since_last_login",
    "upi_failure_count",         # RBI mandate UPI failures
    "contract_value_inr",        # tenure * monthly_charges
]

CATEGORICAL_FEATURES = [
    "payment_method",            # UPI / Credit Card / NetBanking / Wallet
    "contract_type",             # Monthly / Quarterly / Annual
    "business_segment",          # SME / Enterprise / Startup / Individual
    "internet_tier",             # Basic / Standard / Premium
    "has_gst_invoice",           # Yes / No — proxy for B2B
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"


# ─────────────────────────────────────────────
#  SYNTHETIC INDIAN MARKET DATA GENERATOR
# ─────────────────────────────────────────────
def generate_indian_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 61, size=n).astype(float)
    monthly = rng.choice(
        [499, 999, 1499, 2499, 4999, 7999, 12999, 14999],
        size=n, p=[0.10, 0.18, 0.20, 0.22, 0.15, 0.08, 0.05, 0.02]
    ).astype(float)
    contract_value = tenure * monthly * rng.uniform(0.88, 1.12, size=n)

    payment_failures = rng.integers(0, 6, size=n).astype(float)
    upi_failures     = rng.integers(0, 4, size=n).astype(float)
    support_tickets  = rng.integers(0, 8, size=n).astype(float)
    usage_drop       = rng.uniform(0, 80, size=n)           # % drop
    velocity         = rng.uniform(0.5, 100, size=n)
    num_products     = rng.integers(1, 6, size=n).astype(float)
    days_inactive    = rng.integers(0, 90, size=n).astype(float)

    payment_method   = rng.choice(
        ["UPI", "Credit Card", "NetBanking", "Wallet"],
        size=n, p=[0.42, 0.28, 0.22, 0.08]
    )
    contract_type    = rng.choice(
        ["Monthly", "Quarterly", "Annual"],
        size=n, p=[0.50, 0.30, 0.20]
    )
    business_segment = rng.choice(
        ["SME", "Enterprise", "Startup", "Individual"],
        size=n, p=[0.35, 0.20, 0.25, 0.20]
    )
    internet_tier    = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n, p=[0.30, 0.45, 0.25]
    )
    has_gst          = rng.choice(["Yes", "No"], size=n, p=[0.55, 0.45])

    # Churn logic — Indian market weighted
    log_odds = (
        -0.04  * tenure
        + 0.012 * monthly / 1000
        + 0.45  * payment_failures
        + 0.55  * upi_failures           # RBI mandate failures → high churn signal
        + 0.30  * support_tickets
        + 0.025 * usage_drop
        - 0.015 * velocity
        + 0.35  * (contract_type == "Monthly").astype(float)
        - 0.40  * (contract_type == "Annual").astype(float)
        + 0.25  * (payment_method == "Wallet").astype(float)
        - 0.20  * (business_segment == "Enterprise").astype(float)
        + 0.18  * (internet_tier == "Basic").astype(float)
        + 0.15  * days_inactive / 30
        - 0.3
        + rng.normal(0, 0.25, size=n)
    )
    prob    = 1 / (1 + np.exp(-log_odds))
    churned = (rng.uniform(size=n) < prob).astype(int)

    customer_ids = [f"IND-{1001 + i}" for i in range(n)]

    df = pd.DataFrame({
        "customer_id":             customer_ids,
        "monthly_charges_inr":     monthly,
        "tenure_months":           tenure,
        "payment_failures_last_3m": payment_failures,
        "support_tickets_open":    support_tickets,
        "usage_drop_rate_pct":     usage_drop,
        "account_velocity_score":  velocity,
        "num_products_subscribed": num_products,
        "days_since_last_login":   days_inactive,
        "upi_failure_count":       upi_failures,
        "contract_value_inr":      contract_value,
        "payment_method":          payment_method,
        "contract_type":           contract_type,
        "business_segment":        business_segment,
        "internet_tier":           internet_tier,
        "has_gst_invoice":         has_gst,
        TARGET:                    churned,
    })
    logger.info("Generated %d rows | churn rate: %.1f%%", n, churned.mean() * 100)
    return df


# ─────────────────────────────────────────────
#  BUILD PREPROCESSOR
# ─────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERICAL_FEATURES),
            ("cat", Pipeline([
                ("enc", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ─────────────────────────────────────────────
#  TRAIN PIPELINE
# ─────────────────────────────────────────────
def run_training(df: pd.DataFrame) -> dict[str, Any]:
    X = df[ALL_FEATURES]
    y = df[TARGET]

    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    neg, pos = np.bincount(y)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_t, y)

    explainer = shap.TreeExplainer(clf)

    return {
        "model":         clf,
        "preprocessor":  preprocessor,
        "feature_names": feature_names,
        "explainer":     explainer,
        "churn_rate":    round(float(y.mean() * 100), 2),
        "n_samples":     len(df),
    }


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": bool(model_state)}


# ── /train ────────────────────────────────────
@app.post("/train")
async def train():
    """
    Generate synthetic Indian market data and train XGBoost + SHAP explainer.
    Stores model in memory for /predict and /explain calls.
    """
    try:
        df = generate_indian_data(n=1000)
        state = run_training(df)
        model_state.update(state)
        return {
            "status":      "trained",
            "n_samples":   state["n_samples"],
            "churn_rate":  state["churn_rate"],
            "features":    ALL_FEATURES,
        }
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict/batch ────────────────────────────
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept CSV upload, return per-customer churn probabilities.
    Each call is completely stateless — trains fresh on uploaded data.
    """
    if not model_state:
        raise HTTPException(status_code=400, detail="Model not initialised. Call /train first.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = set(ALL_FEATURES) - set(df.columns)
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

        X = df[ALL_FEATURES]
        X_t = model_state["preprocessor"].transform(X)
        probs = model_state["model"].predict_proba(X_t)[:, 1]

        result_df = df.copy()
        result_df["churn_probability"] = np.round(probs, 4)
        result_df["risk_tier"] = pd.cut(
            probs,
            bins=[0, 0.35, 0.65, 1.0],
            labels=["Low", "Medium", "High"],
        ).astype(str)

        return result_df.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── /explain ─────────────────────────────────
class CustomerPayload(BaseModel):
    customer_id: str
    monthly_charges_inr: float
    tenure_months: float
    payment_failures_last_3m: float
    support_tickets_open: float
    usage_drop_rate_pct: float
    account_velocity_score: float
    num_products_subscribed: float
    days_since_last_login: float
    upi_failure_count: float
    contract_value_inr: float
    payment_method: str
    contract_type: str
    business_segment: str
    internet_tier: str
    has_gst_invoice: str


@app.post("/explain")
async def explain(customer: CustomerPayload):
    """
    Return churn probability + SHAP feature impact weights for a single customer.
    """
    if not model_state:
        raise HTTPException(status_code=400, detail="Model not initialised. Call /train first.")
    try:
        row = pd.DataFrame([customer.model_dump()])
        X_t = model_state["preprocessor"].transform(row[ALL_FEATURES])
        prob = float(model_state["model"].predict_proba(X_t)[0, 1])

        shap_vals = model_state["explainer"].shap_values(X_t)
        # shap_vals shape: (1, n_features)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # binary classification list format
        else:
            sv = shap_vals[0]

        feature_names = model_state["feature_names"]
        shap_dict = {
            fn: round(float(sv[i]), 6)
            for i, fn in enumerate(feature_names)
        }
        # Sort by absolute impact descending
        shap_sorted = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "customer_id":       customer.customer_id,
            "churn_probability": round(prob, 4),
            "risk_tier":         "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low",
            "shap_impacts":      shap_sorted,
        }
    except Exception as e:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail=str(e))
