"""
data_processing.py
------------------
Generates synthetic subscription customer data and builds a preprocessing
pipeline using ColumnTransformer (scaling + one-hot encoding).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_SAMPLES = 5_000

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "usage_frequency",
    "support_calls",
    "num_products",
    "login_days_last_30",
    "avg_session_minutes",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "has_tech_support",
    "has_online_backup",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"


def generate_mock_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_samples).astype(float)
    monthly_charges = rng.uniform(20, 120, size=n_samples)
    total_charges = tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_samples)
    usage_frequency = rng.integers(0, 31, size=n_samples).astype(float)
    support_calls = rng.integers(0, 10, size=n_samples).astype(float)
    num_products = rng.integers(1, 6, size=n_samples).astype(float)
    login_days = rng.integers(0, 31, size=n_samples).astype(float)
    avg_session = rng.uniform(5, 120, size=n_samples)

    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_samples, p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        size=n_samples, p=[0.30, 0.25, 0.30, 0.15],
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No"],
        size=n_samples, p=[0.45, 0.40, 0.15],
    )
    has_tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.40, 0.60])
    has_online_backup = rng.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])

    log_odds = (
        -0.05 * tenure
        + 0.015 * monthly_charges
        + 0.25 * support_calls
        - 0.10 * usage_frequency
        - 0.08 * login_days
        + 0.40 * (contract_type == "Month-to-Month").astype(float)
        - 0.30 * (contract_type == "Two Year").astype(float)
        + 0.20 * (payment_method == "Electronic Check").astype(float)
        - 0.15 * (has_tech_support == "Yes").astype(float)
        - 0.10 * (has_online_backup == "Yes").astype(float)
        - 0.5
        + rng.normal(0, 0.3, size=n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churned = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "num_products": num_products,
        "login_days_last_30": login_days,
        "avg_session_minutes": avg_session,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "has_tech_support": has_tech_support,
        "has_online_backup": has_online_backup,
        "churned": churned,
    })

    logger.info("Generated %d rows | churn rate: %.1f%%", n_samples, churned.mean() * 100)
    return df


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def load_or_generate_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        logger.info("Loading data from %s", csv_path)
        df = pd.read_csv(csv_path)
        required = set(ALL_FEATURES + [TARGET])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return df
    logger.info("No CSV provided – generating synthetic dataset.")
    return generate_mock_data()
