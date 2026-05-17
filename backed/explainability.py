"""
explainability.py
-----------------
Feature-importance utilities and retention recommendations.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)
FEATURE_IMPORTANCE_PATH = Path(__file__).parent / "artifacts" / "feature_importance.json"

FEATURE_LABELS: dict[str, str] = {
    "num__tenure_months": "Customer Tenure",
    "num__monthly_charges": "Monthly Charges",
    "num__total_charges": "Total Charges",
    "num__usage_frequency": "Usage Frequency",
    "num__support_calls": "Customer Support Calls",
    "num__num_products": "Number of Products",
    "num__login_days_last_30": "Login Activity (Last 30 Days)",
    "num__avg_session_minutes": "Avg. Session Duration",
    "cat__contract_type_One Year": "Contract: One Year",
    "cat__contract_type_Two Year": "Contract: Two Year",
    "cat__payment_method_Credit Card": "Payment: Credit Card",
    "cat__payment_method_Electronic Check": "Payment: Electronic Check",
    "cat__payment_method_Mailed Check": "Payment: Mailed Check",
    "cat__internet_service_DSL": "Internet Service: DSL",
    "cat__internet_service_No": "No Internet Service",
    "cat__has_tech_support_Yes": "Has Tech Support",
    "cat__has_online_backup_Yes": "Has Online Backup",
}

RETENTION_STRATEGIES: dict[str, dict[str, str]] = {
    "num__support_calls": {
        "label": "High Support Call Volume", "icon": "📞",
        "strategy": "This customer has made frequent support calls, signalling ongoing friction. Assign a dedicated Account Manager for proactive outreach within 24 hours. Offer a complimentary service review session and escalate unresolved tickets.",
    },
    "num__monthly_charges": {
        "label": "High Monthly Charges", "icon": "💸",
        "strategy": "High billing relative to perceived value is a leading churn driver. Offer a loyalty discount (10-20%) or a downgrade path that preserves core features. Highlight the ROI of features they're already using.",
    },
    "num__tenure_months": {
        "label": "Short Tenure", "icon": "⏳",
        "strategy": "New customers are at the highest churn risk. Trigger an onboarding health-check at day 30 and 60. Send a personalised Getting Started guide and offer a live product walkthrough with a success specialist.",
    },
    "num__usage_frequency": {
        "label": "Low Usage Frequency", "icon": "📉",
        "strategy": "Infrequent usage suggests low product adoption. Send targeted in-app nudges highlighting unused features. Set up an automated We Miss You email at 7 days of inactivity and offer a free training webinar.",
    },
    "num__login_days_last_30": {
        "label": "Low Login Activity", "icon": "🔒",
        "strategy": "Low login frequency is a strong early churn signal. Trigger a re-engagement campaign with personalised content. Consider a check-in call from customer success to understand barriers to usage.",
    },
    "cat__contract_type_One Year": {
        "label": "Short-Term Contract", "icon": "📋",
        "strategy": "Month-to-month customers churn 3x more often. Offer an incentive (e.g., 2 months free) to upgrade to an annual plan. Frame it as cost-saving rather than lock-in.",
    },
    "cat__payment_method_Electronic Check": {
        "label": "Electronic Check Payment", "icon": "🏦",
        "strategy": "Electronic check users show higher churn. Encourage migration to auto-pay (credit card or bank transfer) with a small billing credit. Auto-pay reduces involuntary churn from failed payments.",
    },
    "num__avg_session_minutes": {
        "label": "Short Session Duration", "icon": "⏱️",
        "strategy": "Short sessions indicate the customer isn't finding value quickly. Improve time-to-value through UX enhancements, better onboarding tooltips, and a streamlined dashboard highlighting key actions.",
    },
    "cat__has_tech_support_Yes": {
        "label": "Lacks Tech Support Add-on", "icon": "🛠️",
        "strategy": "Customers without tech support report higher frustration. Offer a free 30-day trial of the Tech Support add-on. Customers who adopt support add-ons show significantly higher retention rates.",
    },
    "num__num_products": {
        "label": "Low Product Adoption", "icon": "📦",
        "strategy": "Customers using fewer products have shallower platform integration. Run a product discovery campaign showing complementary features. Offer a bundle discount for adding a second product.",
    },
}

_DEFAULT_STRATEGY: dict[str, str] = {
    "label": "General Churn Risk", "icon": "⚠️",
    "strategy": "Schedule a proactive check-in call from the customer success team. Offer a personalised retention incentive based on the customer's usage history.",
}


def load_global_feature_importance() -> list[dict[str, Any]]:
    try:
        with open(FEATURE_IMPORTANCE_PATH) as f:
            data: list[dict] = json.load(f)
        for item in data:
            item["label"] = FEATURE_LABELS.get(item["feature"], item["feature"])
        return data
    except FileNotFoundError:
        logger.error("Feature importance file not found. Run train.py first.")
        return []


def get_top_risk_factors(
    feature_names: list[str],
    feature_values: list[float],
    feature_importances: "np.ndarray",
    top_n: int = 3,
) -> list[dict[str, Any]]:
    abs_values = np.abs(feature_values)
    combined_score = feature_importances * abs_values
    top_indices = np.argsort(combined_score)[::-1][:top_n]
    risk_factors: list[dict[str, Any]] = []
    for idx in top_indices:
        fname = feature_names[idx]
        strategy_info = RETENTION_STRATEGIES.get(fname, _DEFAULT_STRATEGY)
        risk_factors.append({
            "feature": fname,
            "label": FEATURE_LABELS.get(fname, fname),
            "importance": round(float(feature_importances[idx]), 6),
            "combined_score": round(float(combined_score[idx]), 6),
            "icon": strategy_info.get("icon", "⚠️"),
            "strategy_label": strategy_info.get("label", fname),
            "retention_strategy": strategy_info.get("strategy", ""),
        })
    return risk_factors
