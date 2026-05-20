"""
app.py — Indian Market Customer Churn Prediction Dashboard
Streamlit Cloud frontend — connects to FastAPI backend on Render.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────
#  CONFIG — swap this to your live Render URL
# ─────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "https://your-render-app-url.onrender.com")

# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield India · AI Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; background: #080b12; }
h1,h2,h3,.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif !important; }

section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1c2333; }
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
.main { background: #080b12; }
.block-container { padding: 1.8rem 2.5rem; max-width: 1400px; }

/* Metric cards */
.kpi-card { background: #0d1117; border: 1px solid #1c2333; border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center; }
.kpi-label { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #556; margin-bottom: 0.3rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: #e8ecf4; }
.kpi-sub { font-size: 0.75rem; color: #556; margin-top: 0.2rem; }

/* Risk badges */
.badge { display:inline-block; padding:3px 12px; border-radius:999px; font-size:0.75rem; font-weight:700; }
.badge-high   { background:rgba(255,59,59,0.15);  color:#ff3b3b; border:1px solid #ff3b3b; }
.badge-medium { background:rgba(255,165,0,0.15);  color:#ffa500; border:1px solid #ffa500; }
.badge-low    { background:rgba(0,230,118,0.15);  color:#00e676; border:1px solid #00e676; }

/* Retention alert cards */
.alert-card { background:#0d1117; border-left:3px solid #4f6ef7; border-radius:0 12px 12px 0; padding:0.9rem 1.1rem; margin-bottom:0.7rem; }
.alert-title { font-family:'Syne',sans-serif; font-weight:700; font-size:0.88rem; color:#4f6ef7; margin-bottom:0.25rem; }
.alert-body  { font-size:0.82rem; color:#8a97b5; line-height:1.6; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:0.5rem; background:transparent; border-bottom:1px solid #1c2333; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px 8px 0 0; color:#556 !important; font-weight:600; padding:0.45rem 1.1rem; font-size:0.88rem; }
.stTabs [aria-selected="true"] { background:#0d1117 !important; color:#e8ecf4 !important; border-bottom:2px solid #4f6ef7 !important; }

div[data-testid="metric-container"] { background:#0d1117; border:1px solid #1c2333; border-radius:10px; padding:0.7rem 1rem; }
hr { border-color:#1c2333; }
.stDataFrame { background:#0d1117; }

/* Section headers */
.section-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#e8ecf4; margin-bottom:0.2rem; }
.section-sub   { font-size:0.8rem; color:#556; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS — Indian formatting
# ─────────────────────────────────────────────
def fmt_inr(amount: float) -> str:
    """Format number in Indian system: Lakhs / Crores."""
    if amount >= 1_00_00_000:
        return f"₹{amount/1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


def risk_badge(tier: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(tier, "badge-low")
    return f"<span class='badge {cls}'>{tier}</span>"


# ─────────────────────────────────────────────
#  API CALLS
# ─────────────────────────────────────────────
def api_train() -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}/train", timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.warning("⏳ Backend is waking up (cold start ~30s). Please try again.")
    except Exception as e:
        st.error(f"Train error: {e}")
    return None


def api_predict_batch(csv_bytes: bytes) -> list[dict] | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/predict/batch",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.warning("⏳ Request timed out. Backend may be waking up — try again.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return None


def api_explain(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}/explain", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Explain error: {e}")
    return None


# ─────────────────────────────────────────────
#  SAMPLE DATA GENERATOR (Indian schema)
# ─────────────────────────────────────────────
def generate_sample_csv() -> bytes:
    rng = np.random.default_rng(99)
    n = 50
    tenure = rng.integers(1, 48, size=n).astype(float)
    monthly = rng.choice([499,999,1499,2499,4999,7999,12999], size=n).astype(float)
    data = pd.DataFrame({
        "customer_id":              [f"IND-{1001+i}" for i in range(n)],
        "monthly_charges_inr":      monthly,
        "tenure_months":            tenure,
        "payment_failures_last_3m": rng.integers(0, 5, size=n).astype(float),
        "support_tickets_open":     rng.integers(0, 7, size=n).astype(float),
        "usage_drop_rate_pct":      rng.uniform(0, 75, size=n).round(1),
        "account_velocity_score":   rng.uniform(1, 90, size=n).round(1),
        "num_products_subscribed":  rng.integers(1, 5, size=n).astype(float),
        "days_since_last_login":    rng.integers(0, 60, size=n).astype(float),
        "upi_failure_count":        rng.integers(0, 4, size=n).astype(float),
        "contract_value_inr":       (tenure * monthly * rng.uniform(0.9,1.1,size=n)).round(0),
        "payment_method":           rng.choice(["UPI","Credit Card","NetBanking","Wallet"], size=n),
        "contract_type":            rng.choice(["Monthly","Quarterly","Annual"], size=n, p=[0.5,0.3,0.2]),
        "business_segment":         rng.choice(["SME","Enterprise","Startup","Individual"], size=n),
        "internet_tier":            rng.choice(["Basic","Standard","Premium"], size=n),
        "has_gst_invoice":          rng.choice(["Yes","No"], size=n),
    })
    return data.to_csv(index=False).encode()


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
def gauge_chart(prob: float, customer_id: str) -> go.Figure:
    color = "#ff3b3b" if prob >= 0.65 else "#ffa500" if prob >= 0.35 else "#00e676"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": "#e8ecf4", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2d3a5c", "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1117", "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "rgba(0,230,118,0.06)"},
                {"range": [35, 65], "color": "rgba(255,165,0,0.06)"},
                {"range": [65,100], "color": "rgba(255,59,59,0.06)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": prob*100},
        },
        title={"text": f"Churn Risk — {customer_id}", "font": {"size": 13, "color": "#556"}},
    ))
    fig.update_layout(paper_bgcolor="#080b12", plot_bgcolor="#080b12",
                      margin=dict(l=20, r=20, t=40, b=10), height=290)
    return fig


def shap_bar_chart(shap_impacts: dict[str, float]) -> go.Figure:
    items = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    labels = [k.replace("num__", "").replace("cat__", "").replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]
    colors = ["#ff3b3b" if v > 0 else "#00e676" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(color="#556", size=10),
    ))
    fig.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
        font=dict(color="#8a97b5", family="DM Sans"),
        xaxis=dict(showgrid=True, gridcolor="#1c2333", zeroline=True, zerolinecolor="#2d3a5c"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=70, t=10, b=30), height=400,
        title=dict(text="SHAP Feature Impact (Red=Increases Churn Risk, Green=Decreases)", font=dict(size=12, color="#556")),
    )
    return fig


def prob_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["churn_probability"],
        nbinsx=30,
        marker=dict(
            color=df["churn_probability"],
            colorscale=[[0,"#00e676"],[0.5,"#ffa500"],[1,"#ff3b3b"]],
            showscale=True,
            colorbar=dict(title="Prob", tickfont=dict(color="#556")),
        ),
        opacity=0.85,
    ))
    fig.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
        font=dict(color="#8a97b5"),
        xaxis=dict(title="Churn Probability", showgrid=True, gridcolor="#1c2333"),
        yaxis=dict(title="Customer Count",    showgrid=True, gridcolor="#1c2333"),
        margin=dict(l=10, r=10, t=30, b=40), height=320,
        title=dict(text="Churn Probability Distribution", font=dict(size=13, color="#8a97b5")),
    )
    return fig


def segment_bar(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = df.groupby(col)["churn_probability"].mean().reset_index().sort_values("churn_probability", ascending=True)
    fig = go.Figure(go.Bar(
        x=grp["churn_probability"], y=grp[col], orientation="h",
        marker=dict(color=grp["churn_probability"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#4f6ef7"],[1,"#ff3b3b"]],
                    showscale=False),
        text=[f"{v*100:.1f}%" for v in grp["churn_probability"]],
        textposition="outside", textfont=dict(color="#556", size=10),
    ))
    fig.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
        font=dict(color="#8a97b5"), title=dict(text=title, font=dict(size=12, color="#8a97b5")),
        xaxis=dict(showgrid=True, gridcolor="#1c2333", tickformat=".0%"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=50, t=35, b=20), height=260,
    )
    return fig


# ─────────────────────────────────────────────
#  INDIAN RETENTION STRATEGIES
# ─────────────────────────────────────────────
RETENTION_PLAYBOOK = {
    "upi_failure": {
        "icon": "📲", "title": "UPI / RBI Mandate Failure",
        "action": "Trigger WhatsApp Business API alert with Razorpay UPI deep-link for instant re-authorization. Offer ₹100 cashback on successful mandate renewal within 24 hours.",
    },
    "payment_failure": {
        "icon": "💳", "title": "Payment Failure Pattern",
        "action": "Send automated SMS via MSG91 with Net Banking fallback link. Escalate to CS team for manual NEFT/IMPS collection after 2 failures. Offer 7-day grace period.",
    },
    "monthly_contract": {
        "icon": "📋", "title": "Month-to-Month Contract Risk",
        "action": "Pitch discounted 12-month Annual plan via NetBanking/UPI with 2 months free (effective 17% saving). Highlight GST input tax credit benefit for B2B clients.",
    },
    "high_usage_drop": {
        "icon": "📉", "title": "Usage Drop-off Detected",
        "action": "Trigger in-app onboarding nudge via Intercom. Schedule 30-min product walkthrough with Customer Success. Send Hindi/regional-language feature highlight email.",
    },
    "high_support": {
        "icon": "🎧", "title": "Elevated Support Tickets",
        "action": "Assign dedicated Key Account Manager. Offer free migration assistance. Escalate SLA to Priority tier. Follow up via WhatsApp with resolution ETAs.",
    },
    "low_velocity": {
        "icon": "🔒", "title": "Low Engagement / Inactivity",
        "action": "Send re-engagement campaign via WhatsApp + Email. Offer 15-day free extension for dormant accounts. Book live demo session with regional sales rep.",
    },
    "wallet_payment": {
        "icon": "👛", "title": "Wallet Payment Method",
        "action": "Migrate to UPI AutoPay via Razorpay for stable recurring billing. Wallet balances expire — set up proactive low-balance alert and top-up reminder via SMS.",
    },
    "basic_tier": {
        "icon": "⬆️", "title": "Basic Plan Ceiling",
        "action": "Offer upgrade to Standard plan with 30-day free trial of premium features. Show ROI calculator comparing plan benefits. Provide EMI option via Bajaj Finserv.",
    },
    "default": {
        "icon": "⚠️", "title": "General Retention Action",
        "action": "Schedule proactive CS check-in call. Offer loyalty discount of 10–15% on next renewal. Send personalised value report showing platform ROI in INR terms.",
    },
}


def get_retention_alerts(shap_impacts: dict, customer_row: dict) -> list[dict]:
    alerts = []
    seen = set()

    def add(key):
        if key not in seen:
            seen.add(key)
            alerts.append(RETENTION_PLAYBOOK[key])

    # Map SHAP features → retention strategies
    top_features = sorted(shap_impacts.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, val in top_features:
        if val <= 0:
            continue
        f = feat.lower()
        if "upi_failure" in f:           add("upi_failure")
        elif "payment_failures" in f:     add("payment_failure")
        elif "monthly" in f and "contract" not in f: add("payment_failure")
        elif "contract_type_monthly" in f: add("monthly_contract")
        elif "usage_drop" in f:           add("high_usage_drop")
        elif "support_tickets" in f:      add("high_support")
        elif "velocity" in f:             add("low_velocity")
        elif "wallet" in f:               add("wallet_payment")
        elif "basic" in f:                add("basic_tier")

    # Also check raw customer values
    if customer_row.get("upi_failure_count", 0) >= 2:   add("upi_failure")
    if customer_row.get("payment_failures_last_3m", 0) >= 2: add("payment_failure")
    if customer_row.get("contract_type") == "Monthly":   add("monthly_contract")
    if customer_row.get("usage_drop_rate_pct", 0) >= 40: add("high_usage_drop")
    if customer_row.get("support_tickets_open", 0) >= 3: add("high_support")
    if customer_row.get("payment_method") == "Wallet":   add("wallet_payment")
    if customer_row.get("internet_tier") == "Basic":     add("basic_tier")

    if not alerts:
        add("default")
    return alerts[:4]


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown(
        "<h2 style='font-family:Syne;font-size:1.2rem;color:#e8ecf4;margin-bottom:0;'>"
        "🛡️ ChurnShield India</h2>"
        "<p style='font-size:0.75rem;color:#556;margin-top:2px;'>AI Churn Intelligence Platform</p>"
        "<hr style='border-color:#1c2333;margin:0.5rem 0 1rem;'>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**Step 1 — Initialise Model**")
    if st.sidebar.button("🚀 Train Backend Model", use_container_width=True, type="primary"):
        with st.spinner("Training XGBoost + SHAP on Indian market data..."):
            result = api_train()
        if result:
            st.sidebar.success(f"✅ Trained on {result['n_samples']:,} customers\nChurn rate: {result['churn_rate']}%")
            st.session_state["model_ready"] = True

    st.sidebar.markdown("<br>**Step 2 — Load Customer Data**", unsafe_allow_html=True)
    sample_btn = st.sidebar.button("✨ Auto-Generate Sample Indian Client Base", use_container_width=True)
    if sample_btn:
        st.session_state["sample_csv"] = generate_sample_csv()
        st.sidebar.success("50 sample Indian customers generated!")

    uploaded = st.sidebar.file_uploader("Or upload your own CSV", type=["csv"])
    if uploaded:
        st.session_state["uploaded_csv"] = uploaded.read()
        st.session_state["sample_csv"]   = None

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='background:#0d1117;border:1px solid #1c2333;border-radius:8px;padding:0.7rem;font-size:0.75rem;color:#556;'>"
        "⚡ <strong style='color:#8a97b5;'>Cold Start Note:</strong> Render free tier sleeps after 15 min. "
        "First train request may take ~30s."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  TAB 1: PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────
def tab_portfolio():
    st.markdown(
        "<h1 style='font-family:Syne;font-size:1.8rem;color:#e8ecf4;margin-bottom:0.1rem;'>"
        "📈 Portfolio Revenue-at-Risk Overview</h1>"
        "<p style='color:#556;margin-bottom:1.2rem;font-size:0.85rem;'>"
        "Upload or generate your Indian client base, then run the analysis engine.</p>",
        unsafe_allow_html=True,
    )

    # Get CSV bytes
    csv_bytes = st.session_state.get("sample_csv") or st.session_state.get("uploaded_csv")

    if csv_bytes:
        preview_df = pd.read_csv(io.BytesIO(csv_bytes))
        with st.expander(f"📄 Preview uploaded data ({len(preview_df)} rows)", expanded=False):
            st.dataframe(preview_df.head(10), use_container_width=True)

    analyze_btn = st.button(
        "🔍 Run Advanced Churn Analysis",
        use_container_width=True,
        type="primary",
        disabled=not csv_bytes,
    )

    if not csv_bytes:
        st.info("👈 Generate sample data or upload a CSV from the sidebar first, then click Analyse.")
        return

    if analyze_btn:
        with st.spinner("Running churn analysis pipeline..."):
            results = api_predict_batch(csv_bytes)

        if not results:
            return

        pred_df = pd.DataFrame(results)
        st.session_state["pred_df"] = pred_df  # share with Tab 2

        # ── KPI Cards ──────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        total_customers = len(pred_df)
        high_risk = pred_df[pred_df["risk_tier"] == "High"]
        med_risk  = pred_df[pred_df["risk_tier"] == "Medium"]

        mrr_total    = pred_df["monthly_charges_inr"].sum()
        mrr_at_risk  = high_risk["monthly_charges_inr"].sum()
        arr_at_risk  = mrr_at_risk * 12

        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        cards = [
            ("Total Customers",       str(total_customers),           ""),
            ("🔴 High Risk",          str(len(high_risk)),            f"{len(high_risk)/total_customers*100:.1f}% of base"),
            ("🟡 Medium Risk",        str(len(med_risk)),             f"{len(med_risk)/total_customers*100:.1f}% of base"),
            ("MRR at Risk",           fmt_inr(mrr_at_risk),           "Monthly recurring"),
            ("ARR at Risk",           fmt_inr(arr_at_risk),           "Annual exposure"),
        ]
        for col, (label, value, sub) in zip([c1,c2,c3,c4,c5], cards):
            with col:
                st.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
                    f"<div class='kpi-value'>{value}</div>"
                    f"<div class='kpi-sub'>{sub}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Histogram ──────────────────
        col_hist, col_seg = st.columns([3, 2], gap="large")
        with col_hist:
            st.plotly_chart(prob_histogram(pred_df), use_container_width=True)
        with col_seg:
            if "contract_type" in pred_df.columns:
                st.plotly_chart(segment_bar(pred_df, "contract_type", "Avg Churn Risk by Contract"), use_container_width=True)

        # ── Segment breakdown ─────────────────────
        seg_cols = st.columns(2, gap="large")
        seg_fields = [("business_segment","Avg Churn Risk by Segment"), ("payment_method","Avg Churn Risk by Payment Method")]
        for col, (field, title) in zip(seg_cols, seg_fields):
            if field in pred_df.columns:
                with col:
                    st.plotly_chart(segment_bar(pred_df, field, title), use_container_width=True)

        # ── High-risk customer table ───────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>🚨 High-Risk Customer Table</div>"
            "<div class='section-sub'>Customers with highest predicted churn probability</div>",
            unsafe_allow_html=True,
        )
        display_cols = [c for c in [
            "customer_id","churn_probability","risk_tier",
            "monthly_charges_inr","tenure_months","contract_type",
            "payment_method","business_segment","upi_failure_count"
        ] if c in pred_df.columns]

        table_df = pred_df[display_cols].sort_values("churn_probability", ascending=False).head(20).copy()
        table_df["churn_probability"] = (table_df["churn_probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    elif "pred_df" in st.session_state:
        st.info("Results from last run shown in Tab 2. Click **Run Advanced Churn Analysis** to refresh.")


# ─────────────────────────────────────────────
#  TAB 2: INDIVIDUAL CLIENT RISK AUDITOR
# ─────────────────────────────────────────────
def tab_individual():
    st.markdown(
        "<h1 style='font-family:Syne;font-size:1.8rem;color:#e8ecf4;margin-bottom:0.1rem;'>"
        "🔍 Individual Client Risk Auditor</h1>"
        "<p style='color:#556;margin-bottom:1.2rem;font-size:0.85rem;'>"
        "Select a customer to get SHAP-powered explainability and tailored Indian retention strategies.</p>",
        unsafe_allow_html=True,
    )

    pred_df = st.session_state.get("pred_df")
    if pred_df is None:
        st.info("👈 Run the churn analysis in Tab 1 first.")
        return

    csv_bytes = st.session_state.get("sample_csv") or st.session_state.get("uploaded_csv")
    if csv_bytes is None:
        st.warning("Source CSV not found. Please re-upload and re-run.")
        return

    source_df = pd.read_csv(io.BytesIO(csv_bytes))

    # Customer selector
    all_ids = pred_df["customer_id"].tolist() if "customer_id" in pred_df.columns else pred_df.index.tolist()
    selected_id = st.selectbox("🔎 Select Customer ID", all_ids)

    row_pred   = pred_df[pred_df["customer_id"] == selected_id].iloc[0]
    row_source = source_df[source_df["customer_id"] == selected_id].iloc[0] if "customer_id" in source_df.columns else source_df.iloc[0]

    col_gauge, col_meta = st.columns([1, 1], gap="large")
    with col_gauge:
        prob = float(row_pred["churn_probability"])
        st.plotly_chart(gauge_chart(prob, selected_id), use_container_width=True)

    with col_meta:
        st.markdown("<br>", unsafe_allow_html=True)
        tier = row_pred.get("risk_tier", "Low")
        st.markdown(f"**Risk Tier:** {risk_badge(tier)}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Monthly Charges", f"₹{int(row_pred.get('monthly_charges_inr', 0)):,}")
        m2.metric("Tenure", f"{int(row_pred.get('tenure_months', 0))} months")
        m3, m4 = st.columns(2)
        m3.metric("Contract",       row_pred.get("contract_type", "—"))
        m4.metric("Payment Method", row_pred.get("payment_method", "—"))
        m5, m6 = st.columns(2)
        m5.metric("UPI Failures",   int(row_pred.get("upi_failure_count", 0)))
        m6.metric("Support Tickets",int(row_pred.get("support_tickets_open", 0)))

    st.divider()

    # SHAP explain call
    explain_btn = st.button("⚡ Get SHAP Explanation & Retention Plan", type="primary")
    if explain_btn:
        NUMERICAL_FEATURES = [
            "monthly_charges_inr","tenure_months","payment_failures_last_3m",
            "support_tickets_open","usage_drop_rate_pct","account_velocity_score",
            "num_products_subscribed","days_since_last_login","upi_failure_count","contract_value_inr",
        ]
        CATEGORICAL_FEATURES = [
            "payment_method","contract_type","business_segment","internet_tier","has_gst_invoice",
        ]
        payload = {}
        for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
            val = row_source.get(f, row_pred.get(f, 0))
            payload[f] = float(val) if f in NUMERICAL_FEATURES else str(val)
        payload["customer_id"] = selected_id

        with st.spinner("Calculating SHAP impacts..."):
            explain_result = api_explain(payload)

        if not explain_result:
            return

        shap_impacts = explain_result.get("shap_impacts", {})
        shap_prob    = explain_result.get("churn_probability", prob)

        # SHAP bar chart
        col_shap, col_ret = st.columns([3, 2], gap="large")
        with col_shap:
            st.markdown(
                "<div class='section-title'>📊 SHAP Feature Impact</div>"
                "<div class='section-sub'>Why is this customer at risk? Red = increases churn, Green = decreases churn</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(shap_bar_chart(shap_impacts), use_container_width=True)

        with col_ret:
            st.markdown(
                "<div class='section-title'>🇮🇳 Retention Action Matrix</div>"
                "<div class='section-sub'>Localised Indian market interventions based on churn root cause</div>",
                unsafe_allow_html=True,
            )
            alerts = get_retention_alerts(shap_impacts, row_source.to_dict())
            for alert in alerts:
                st.markdown(
                    f"<div class='alert-card'>"
                    f"<div class='alert-title'>{alert['icon']} {alert['title']}</div>"
                    f"<div class='alert-body'>{alert['action']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # Init session state
    for key in ["pred_df", "sample_csv", "uploaded_csv", "model_ready"]:
        if key not in st.session_state:
            st.session_state[key] = None

    render_sidebar()

    tab1, tab2 = st.tabs([
        "📈 Portfolio Revenue-at-Risk Overview",
        "🔍 Individual Client Risk Auditor",
    ])
    with tab1: tab_portfolio()
    with tab2: tab_individual()


if __name__ == "__main__":
    main()
