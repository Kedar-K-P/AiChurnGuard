"""
app.py  -  ChurnGuard · Customer Churn Prediction Dashboard (Streamlit)
"""
from __future__ import annotations
import os
from typing import Any
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("CHURN_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ChurnGuard · Prediction Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif !important; }
section[data-testid="stSidebar"] { background: #0d0f14; border-right: 1px solid #1e2330; }
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
.main { background: #0d0f14; }
.block-container { padding: 2rem 3rem; }
.metric-card { background: #141720; border: 1px solid #1e2330; border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center; }
.metric-card .label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7a99; margin-bottom: 0.4rem; }
.metric-card .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #e8ecf4; }
.risk-badge { display: inline-block; padding: 0.5rem 1.4rem; border-radius: 999px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; letter-spacing: 0.06em; text-transform: uppercase; }
.risk-high   { background: rgba(255,59,59,0.15);  color: #ff3b3b; border: 1px solid #ff3b3b; }
.risk-medium { background: rgba(255,165,0,0.15);  color: #ffa500; border: 1px solid #ffa500; }
.risk-low    { background: rgba(0,230,118,0.15);  color: #00e676; border: 1px solid #00e676; }
.strategy-card { background: #141720; border-left: 3px solid #4f6ef7; border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.strategy-card .strat-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.92rem; color: #4f6ef7; margin-bottom: 0.3rem; }
.strategy-card .strat-body { font-size: 0.88rem; color: #9aa5be; line-height: 1.55; }
div[data-testid="metric-container"] { background: #141720; border: 1px solid #1e2330; border-radius: 10px; padding: 0.8rem 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 0.6rem; background: transparent; border-bottom: 1px solid #1e2330; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px 8px 0 0; color: #6b7a99 !important; font-weight: 600; padding: 0.5rem 1.2rem; font-size: 0.9rem; }
.stTabs [aria-selected="true"] { background: #141720 !important; color: #e8ecf4 !important; border-bottom: 2px solid #4f6ef7 !important; }
.pricing-wrapper { background: #0d0f14; border: 1px solid #1e2330; border-radius: 16px; padding: 2rem; margin-top: 1rem; }
.pricing-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: #e8ecf4; text-align: center; margin-bottom: 0.4rem; }
.pricing-sub { color: #6b7a99; text-align: center; font-size: 0.88rem; margin-bottom: 1.6rem; }
.plan-card { background: #141720; border: 1px solid #1e2330; border-radius: 14px; padding: 1.6rem; position: relative; }
.plan-card.featured { border-color: #4f6ef7; box-shadow: 0 0 30px rgba(79,110,247,0.18); }
.plan-label { font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#e8ecf4; }
.plan-badge { display:inline-block; background:rgba(79,110,247,0.18); color:#4f6ef7; border:1px solid #4f6ef7; border-radius:999px; font-size:0.7rem; font-weight:700; padding:2px 10px; margin-left:8px; }
.plan-badge-secondary { display:inline-block; background:#1e2330; color:#9aa5be; border:1px solid #2d3a5c; border-radius:999px; font-size:0.7rem; font-weight:600; padding:2px 10px; margin-left:8px; }
.plan-price { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:#e8ecf4; margin:1.2rem 0 0.2rem; }
.plan-price span { font-size:1.1rem; color:#6b7a99; font-weight:400; }
.plan-old { color:#6b7a99; font-size:0.85rem; text-decoration:line-through; margin-bottom:1rem; }
.plan-desc { color:#6b7a99; font-size:0.82rem; margin-bottom:1.4rem; }
.plan-btn { display:block; width:100%; text-align:center; padding:0.65rem; border-radius:10px; font-family:'Syne',sans-serif; font-weight:700; font-size:0.88rem; cursor:pointer; text-decoration:none; }
.plan-btn-primary { background:linear-gradient(135deg,#4f6ef7,#7b5ea7); color:#fff; border:none; }
.plan-btn-outline { background:transparent; color:#9aa5be; border:1px solid #2d3a5c; }
.plan-features { margin-top:1.2rem; }
.plan-feature { display:flex; align-items:center; gap:8px; color:#9aa5be; font-size:0.83rem; padding:4px 0; }
.plan-feature .check { color:#00e676; font-size:0.9rem; }
.shield-note { text-align:center; color:#6b7a99; font-size:0.8rem; margin-top:1rem; display:flex; align-items:center; justify-content:center; gap:6px; }
hr { border-color: #1e2330; }
</style>
""", unsafe_allow_html=True)


def call_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Is it running on port 8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    return None

@st.cache_data(ttl=300)
def call_metrics() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Is it running on port 8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    return None


def render_gauge(prob: float) -> go.Figure:
    color = "#ff3b3b" if prob >= 0.70 else "#ffa500" if prob >= 0.40 else "#00e676"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": "#e8ecf4", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3d4560", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#141720", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(0,230,118,0.07)"},
                {"range": [40, 70], "color": "rgba(255,165,0,0.07)"},
                {"range": [70,100], "color": "rgba(255,59,59,0.07)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob*100},
        },
        title={"text": "Churn Probability", "font": {"size": 14, "color": "#6b7a99"}},
    ))
    fig.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                      margin=dict(l=20,r=20,t=30,b=10), height=260)
    return fig

def render_feature_chart(data: list[dict]) -> go.Figure:
    top = sorted(data, key=lambda x: x["importance"], reverse=True)[:12]
    labels = [d.get("label", d["feature"]) for d in top]
    values = [d["importance"] for d in top]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker=dict(color=values[::-1],
                    colorscale=[[0,"#1e2b6e"],[0.5,"#4f6ef7"],[1,"#a78bfa"]],
                    showscale=False),
        text=[f"{v:.4f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(color="#6b7a99", size=10),
    ))
    fig.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#141720",
                      font=dict(color="#9aa5be", family="DM Sans"),
                      xaxis=dict(showgrid=True, gridcolor="#1e2330"),
                      yaxis=dict(showgrid=False),
                      margin=dict(l=10,r=60,t=10,b=40), height=420)
    return fig


def sidebar_inputs() -> dict:
    st.sidebar.markdown(
        "<h2 style='font-family:Syne;font-size:1.3rem;color:#e8ecf4;margin-bottom:0.2rem;'>"
        "🛡️ ChurnGuard</h2>"
        "<p style='font-size:0.78rem;color:#6b7a99;margin-top:0;'>Customer Churn Intelligence</p>"
        "<hr style='border-color:#1e2330;margin:0.6rem 0 1rem;'>",
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Customer Profile")
    tenure   = st.sidebar.slider("Tenure (months)", 0, 72, 24)
    monthly  = st.sidebar.slider("Monthly Charges ($)", 20, 120, 65)
    total = round(float(tenure * monthly), 2)
    st.sidebar.markdown(
        f"<div style='font-size:0.78rem;color:#6b7a99;padding:4px 0 8px;'>"
        f"Total Charges (auto): <strong style='color:#9aa5be;'>${total:,.2f}</strong></div>",
        unsafe_allow_html=True,
    )
    usage    = st.sidebar.slider("Usage Frequency (days/month)", 0, 31, 18)
    calls    = st.sidebar.slider("Support Calls", 0, 10, 2)
    products = st.sidebar.slider("Number of Products", 1, 5, 2)
    logins   = st.sidebar.slider("Login Days (last 30)", 0, 30, 15)
    session  = st.sidebar.slider("Avg Session (minutes)", 5, 120, 40)
    st.sidebar.subheader("Subscription Details")
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment  = st.sidebar.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    internet = st.sidebar.selectbox("Internet Service", ["Fiber Optic", "DSL", "No"])
    tech     = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    backup   = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    return dict(
        tenure_months=tenure, monthly_charges=monthly, total_charges=total,
        usage_frequency=usage, support_calls=calls, num_products=products,
        login_days_last_30=logins, avg_session_minutes=session,
        contract_type=contract, payment_method=payment,
        internet_service=internet, has_tech_support=tech, has_online_backup=backup,
    )


def tab_prediction(payload: dict) -> None:
    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#e8ecf4;margin-bottom:0.2rem;'>"
        "Customer Churn Prediction</h1>"
        "<p style='color:#6b7a99;margin-bottom:1.4rem;'>Configure the customer profile in the "
        "sidebar, then click <strong>Run Prediction</strong>.</p>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔍 Run Prediction", use_container_width=True, type="primary"):
        with st.spinner("Analysing customer profile ..."):
            result = call_predict(payload)
        if result:
            prob = result["churn_probability"]
            risk = result["risk_level"]
            rc   = f"risk-{risk.lower()}"
            col_gauge, col_detail = st.columns([1, 1], gap="large")
            with col_gauge:
                st.plotly_chart(render_gauge(prob), use_container_width=True)
            with col_detail:
                st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                st.markdown(f"<span class='risk-badge {rc}'>{risk} Risk</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                verdict = "⚠️ Likely to churn" if result["churn_prediction"] else "✅ Likely to stay"
                st.metric("Verdict", verdict)
                st.metric("Churn Probability", f"{prob*100:.1f}%")
                st.metric("Risk Level", risk)
            st.divider()
            st.markdown(
                "<h3 style='font-family:Syne;color:#e8ecf4;'>🎯 Retention Insights</h3>"
                "<p style='color:#6b7a99;font-size:0.85rem;margin-bottom:1rem;'>"
                "Top 3 risk factors driving this prediction, with recommended actions.</p>",
                unsafe_allow_html=True,
            )
            for factor in result.get("risk_factors", []):
                st.markdown(
                    f'<div class="strategy-card">'
                    f'<div class="strat-title">{factor.get("icon","⚠️")} &nbsp;{factor.get("strategy_label", factor["label"])}</div>'
                    f'<div class="strat-body">{factor.get("retention_strategy","")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("👈  Configure the customer profile in the sidebar and click **Run Prediction**.")


def tab_analytics() -> None:
    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#e8ecf4;margin-bottom:0.2rem;'>"
        "Model Analytics</h1>"
        "<p style='color:#6b7a99;margin-bottom:1.4rem;'>"
        "Global model performance & feature importance — cached for 5 minutes.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Fetching metrics ..."):
        data = call_metrics()
    if not data:
        return
    m  = data.get("model_metrics", {})
    fi = data.get("feature_importance", [])
    if not m:
        st.warning("Model metrics not available. Check that the backend trained successfully.")
        return
    cols = st.columns(4, gap="small")
    for col, (label, val) in zip(cols, [
        ("ROC-AUC", m.get("roc_auc", 0)), ("F1-Score", m.get("f1_score", 0)),
        ("Precision", m.get("precision", 0)), ("Recall", m.get("recall", 0)),
    ]):
        with col:
            st.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{val:.3f}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cv_cols = st.columns(3, gap="small")
    with cv_cols[0]:
        st.markdown(f"<div class='metric-card'><div class='label'>CV AUC Mean</div><div class='value'>{m.get('cv_auc_mean',0):.3f}</div></div>", unsafe_allow_html=True)
    with cv_cols[1]:
        total_s = m.get("train_samples",0) + m.get("test_samples",0)
        st.markdown(f"<div class='metric-card'><div class='label'>Dataset Size</div><div class='value'>{total_s:,}</div></div>", unsafe_allow_html=True)
    with cv_cols[2]:
        st.markdown(f"<div class='metric-card'><div class='label'>Churn Rate</div><div class='value'>{m.get('churn_rate_pct',0)}%</div></div>", unsafe_allow_html=True)
    if fi:
        st.divider()
        st.markdown("<h3 style='font-family:Syne;color:#e8ecf4;'>Feature Importance (Top 12)</h3>", unsafe_allow_html=True)
        st.plotly_chart(render_feature_chart(fi), use_container_width=True)
    cm = m.get("confusion_matrix")
    if cm:
        st.divider()
        st.markdown("<h3 style='font-family:Syne;color:#e8ecf4;'>Confusion Matrix (Test Set)</h3>", unsafe_allow_html=True)
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retained","Churned"], y=["Retained","Churned"],
            color_continuous_scale=[[0,"#141720"],[1,"#4f6ef7"]], text_auto=True)
        fig_cm.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                             font=dict(color="#9aa5be"), margin=dict(l=10,r=10,t=10,b=10), height=320)
        fig_cm.update_traces(textfont_size=18)
        st.plotly_chart(fig_cm, use_container_width=True)


def tab_pricing() -> None:
    st.markdown(
        "<h1 style='font-family:Syne;font-size:2rem;color:#e8ecf4;margin-bottom:0.2rem;'>"
        "Pricing</h1>"
        "<p style='color:#6b7a99;margin-bottom:1rem;'>Simple, transparent pricing based on your success.</p>",
        unsafe_allow_html=True,
    )
    features = ["Unlimited churn predictions","Real-time risk scoring","Retention strategy insights","Feature importance dashboard","API access","Export to CSV"]
    features_html = "".join(f'<div class="plan-feature"><span class="check">✓</span>{f}</div>' for f in features)
    st.markdown(f"""
    <div class="pricing-wrapper">
        <div style="text-align:center;margin-bottom:0.5rem;">
            <span style="background:#1e2330;color:#6b7a99;border:1px solid #2d3a5c;border-radius:999px;font-size:0.75rem;padding:3px 14px;font-family:monospace;">Pricing</span>
        </div>
        <div class="pricing-title">Pricing Based on Your Success</div>
        <div class="pricing-sub">One plan. All features. No hidden fees.</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;max-width:680px;margin:0 auto;">
            <div class="plan-card">
                <div><span class="plan-label">Monthly</span><span class="plan-badge-secondary">11% off</span></div>
                <div class="plan-old">was $8.99/mo</div>
                <div class="plan-price">$7<span>.99 /mo</span></div>
                <div class="plan-desc">Best value for growing businesses!</div>
                <a href="#" class="plan-btn plan-btn-outline">Start Your Journey</a>
                <div class="plan-features">{features_html}</div>
            </div>
            <div class="plan-card featured">
                <div><span class="plan-label">Yearly</span><span class="plan-badge">22% off</span></div>
                <div class="plan-old">was $8.99/mo</div>
                <div class="plan-price">$6<span>.99 /mo</span></div>
                <div class="plan-desc">Unlock savings with an annual commitment!</div>
                <a href="#" class="plan-btn plan-btn-primary">Get Started Now</a>
                <div class="plan-features">{features_html}</div>
            </div>
        </div>
        <div class="shield-note">🛡️ Access to all features with no hidden fees</div>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    payload = sidebar_inputs()
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Customer Prediction",
        "📊 Model Analytics",
        "💳 Pricing",
    ])
    with tab1: tab_prediction(payload)
    with tab2: tab_analytics()
    with tab3: tab_pricing()

if __name__ == "__main__":
    main()
