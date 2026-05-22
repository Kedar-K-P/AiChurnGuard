"""
app.py — Aegis Intelligence · Indian SaaS Churn Predictor
Streamlit frontend matching the Aegis Intelligence UI design from Stitch.
"""
from __future__ import annotations
import io, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "https://your-render-app-url.onrender.com")

st.set_page_config(
    page_title="Aegis Intelligence · Churn Control",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; box-sizing: border-box; }
.stApp, .main, [data-testid="stAppViewContainer"] { background: #080c14 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
section[data-testid="stSidebar"] { background: #0b1120 !important; border-right: 1px solid #1a2540 !important; width: 240px !important; min-width: 240px !important; }
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: #7a8aaa !important; }
section[data-testid="stSidebar"] .stButton > button { background: #00f5a0 !important; color: #080c14 !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 13px !important; padding: 12px 16px !important; width: 100% !important; margin: 4px 0 !important; }
section[data-testid="stSidebar"] .stButton > button:hover { background: #00daa0 !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; border-bottom: 1px solid #1a2540 !important; padding: 0 32px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #4a5a7a !important; font-size: 13px !important; font-weight: 500 !important; padding: 12px 20px !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #00f5a0 !important; border-bottom: 2px solid #00f5a0 !important; }
div[data-testid="metric-container"] { display: none !important; }
.stSelectbox > div > div { background: #0f1829 !important; border: 1px solid #1a2540 !important; color: #c8d4ee !important; border-radius: 8px !important; }
.stFileUploader { background: #0f1829 !important; border: 1px solid #1a2540 !important; border-radius: 8px !important; padding: 8px !important; }
.stSpinner > div { border-top-color: #00f5a0 !important; }
.stAlert { background: #0f1829 !important; border: 1px solid #1a2540 !important; color: #7a8aaa !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────
def fmt_inr(v):
    if v >= 1_00_00_000: return f"₹{v/1_00_00_000:.2f} Cr"
    if v >= 1_00_000:    return f"₹{v/1_00_000:.2f} L"
    return f"₹{v:,.0f}"

def fmt_inr_short(v):
    if v >= 1_00_00_000: return f"₹{v/1_00_00_000:.1f}Cr"
    if v >= 1_00_000:    return f"₹{v/1_00_000:.1f}L"
    return f"₹{v:,.0f}"


# ── API ───────────────────────────────────────
def api_train():
    try:
        r = requests.post(f"{BACKEND_URL}/train", timeout=60)
        r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout: st.warning("⏳ Backend waking up (~30s). Try again.")
    except Exception as e: st.error(f"Train error: {e}")
    return None

def api_predict_batch(csv_bytes):
    try:
        r = requests.post(f"{BACKEND_URL}/predict/batch",
            files={"file": ("data.csv", csv_bytes, "text/csv")}, timeout=60)
        r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout: st.warning("⏳ Timed out — try again.")
    except Exception as e: st.error(f"Prediction error: {e}")
    return None

def api_explain(payload):
    try:
        r = requests.post(f"{BACKEND_URL}/explain", json=payload, timeout=30)
        r.raise_for_status(); return r.json()
    except Exception as e: st.error(f"Explain error: {e}")
    return None


# ── Sample data ───────────────────────────────
def generate_sample_csv():
    rng = np.random.default_rng(42)
    n = 50
    tenure  = rng.integers(1, 48, n).astype(float)
    monthly = rng.choice([499,999,1499,2499,4999,7999,12999], n).astype(float)
    df = pd.DataFrame({
        "customer_id":              [f"IND-{1001+i}" for i in range(n)],
        "monthly_charges_inr":      monthly,
        "tenure_months":            tenure,
        "payment_failures_last_3m": rng.integers(0,5,n).astype(float),
        "support_tickets_open":     rng.integers(0,7,n).astype(float),
        "usage_drop_rate_pct":      rng.uniform(0,75,n).round(1),
        "account_velocity_score":   rng.uniform(1,90,n).round(1),
        "num_products_subscribed":  rng.integers(1,5,n).astype(float),
        "days_since_last_login":    rng.integers(0,60,n).astype(float),
        "upi_failure_count":        rng.integers(0,4,n).astype(float),
        "contract_value_inr":       (tenure*monthly*rng.uniform(0.9,1.1,n)).round(0),
        "payment_method":           rng.choice(["UPI","Credit Card","NetBanking","Wallet"],n),
        "contract_type":            rng.choice(["Monthly","Quarterly","Annual"],n,p=[0.5,0.3,0.2]),
        "business_segment":         rng.choice(["SME","Enterprise","Startup","Individual"],n),
        "internet_tier":            rng.choice(["Basic","Standard","Premium"],n),
        "has_gst_invoice":          rng.choice(["Yes","No"],n),
    })
    return df.to_csv(index=False).encode()


# ── Retention playbook ────────────────────────
PLAYBOOK = {
    "upi_failure":      {"icon":"📲","title":"UPI Mandate Failure",    "action":"Trigger WhatsApp Business alert with Razorpay UPI deep-link. Offer ₹100 cashback on mandate renewal within 24h."},
    "payment_failure":  {"icon":"💳","title":"Payment Failure Pattern", "action":"Send SMS via MSG91 with NetBanking fallback. Escalate to CS after 2 failures. Offer 7-day grace period."},
    "monthly_contract": {"icon":"📋","title":"Month-to-Month Risk",     "action":"Pitch 12-month Annual plan via UPI with 2 months free (17% saving). Highlight GST input credit for B2B clients."},
    "high_usage_drop":  {"icon":"📉","title":"Usage Drop-off",          "action":"Trigger Intercom in-app nudge. Schedule 30-min walkthrough with CS in Hindi/regional language."},
    "high_support":     {"icon":"🎧","title":"Support Ticket Spike",    "action":"Assign dedicated KAM. Offer free migration. Escalate SLA to Priority. Follow up via WhatsApp."},
    "low_velocity":     {"icon":"🔒","title":"Low Engagement",          "action":"Re-engagement via WhatsApp+Email. Offer 15-day free extension. Book live demo with regional rep."},
    "wallet_payment":   {"icon":"👛","title":"Wallet Payment Risk",     "action":"Migrate to UPI AutoPay via Razorpay. Set proactive low-balance alert via SMS."},
    "basic_tier":       {"icon":"⬆️","title":"Basic Plan Ceiling",      "action":"Offer Standard upgrade with 30-day premium trial. Show INR ROI calculator. EMI via Bajaj Finserv."},
    "default":          {"icon":"⚠️","title":"General Risk",            "action":"Schedule CS check-in. Offer 10–15% loyalty discount on renewal. Send personalised INR value report."},
}

def get_alerts(shap_impacts, row):
    alerts, seen = [], set()
    def add(k):
        if k not in seen: seen.add(k); alerts.append(PLAYBOOK[k])
    for feat, val in sorted(shap_impacts.items(), key=lambda x: x[1], reverse=True)[:5]:
        if val <= 0: continue
        f = feat.lower()
        if "upi_failure" in f:             add("upi_failure")
        elif "payment_failures" in f:      add("payment_failure")
        elif "contract_type_monthly" in f: add("monthly_contract")
        elif "usage_drop" in f:            add("high_usage_drop")
        elif "support_tickets" in f:       add("high_support")
        elif "velocity" in f:              add("low_velocity")
        elif "wallet" in f:                add("wallet_payment")
        elif "basic" in f:                 add("basic_tier")
    if row.get("upi_failure_count",0) >= 2:        add("upi_failure")
    if row.get("payment_failures_last_3m",0) >= 2: add("payment_failure")
    if row.get("contract_type") == "Monthly":      add("monthly_contract")
    if row.get("usage_drop_rate_pct",0) >= 40:     add("high_usage_drop")
    if row.get("support_tickets_open",0) >= 3:     add("high_support")
    if row.get("payment_method") == "Wallet":      add("wallet_payment")
    if row.get("internet_tier") == "Basic":        add("basic_tier")
    if not alerts: add("default")
    return alerts[:4]


# ── Sidebar ───────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:24px 20px 20px;border-bottom:1px solid #1a2540;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#00f5a0,#00c8f8);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">🛡</div>
            <div>
              <div style="font-size:14px;font-weight:700;color:#e8f0ff;letter-spacing:0.5px;">Aegis Intelligence</div>
              <div style="font-size:10px;color:#3a4a6a;letter-spacing:1px;text-transform:uppercase;">Enterprise Tier</div>
            </div>
          </div>
        </div>
        <div style="padding:16px 12px 8px;">
          <div style="display:flex;align-items:center;gap:10px;background:#0f1829;border-radius:8px;padding:10px 14px;margin-bottom:4px;border-left:3px solid #00f5a0;">
            <span style="font-size:14px;">⊞</span><span style="font-size:13px;font-weight:600;color:#e8f0ff;">Dashboard</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;margin-bottom:4px;">
            <span style="font-size:14px;">✦</span><span style="font-size:13px;color:#4a5a7a;">Predictive Insights</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;margin-bottom:4px;">
            <span style="font-size:14px;">◈</span><span style="font-size:13px;color:#4a5a7a;">Customer Segments</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;margin-bottom:4px;">
            <span style="font-size:14px;">⚡</span><span style="font-size:13px;color:#4a5a7a;">Automated Actions</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;margin-bottom:16px;">
            <span style="font-size:14px;">⚙</span><span style="font-size:13px;color:#4a5a7a;">System Settings</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="padding:0 12px 6px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">Model Engine</div></div>', unsafe_allow_html=True)
        if st.button("⚡ Initialise Model", use_container_width=True):
            with st.spinner("Training XGBoost + SHAP on Indian data..."):
                result = api_train()
            if result:
                st.success(f"✅ Trained · {result['n_samples']:,} clients · {result['churn_rate']}% churn")
                st.session_state["model_ready"] = True

        st.markdown('<div style="padding:8px 12px 6px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">Data Source</div></div>', unsafe_allow_html=True)
        if st.button("✨ Auto-Generate Indian Client Base", use_container_width=True):
            st.session_state["sample_csv"]   = generate_sample_csv()
            st.session_state["uploaded_csv"] = None
            st.success("50 Indian clients generated!")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            st.session_state["uploaded_csv"] = uploaded.read()
            st.session_state["sample_csv"]   = None

        st.markdown('<div style="padding:10px 12px 0;">', unsafe_allow_html=True)
        run_clicked = st.button("🔍 Run Advanced Churn Analysis", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:16px 20px;border-top:1px solid #1a2540;margin-top:24px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="color:#3a4a6a;">⊕</span><span style="font-size:12px;color:#3a4a6a;">Support</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="color:#3a4a6a;">→</span><span style="font-size:12px;color:#3a4a6a;">Sign Out</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    return run_clicked


# ── Top bar ───────────────────────────────────
def render_topbar():
    st.markdown("""
    <div style="background:#0b1120;border-bottom:1px solid #1a2540;padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:52px;">
      <div style="display:flex;align-items:center;gap:0;">
        <div style="font-size:18px;font-weight:700;color:#e8f0ff;margin-right:32px;">Aegis Churn Control</div>
        <div style="padding:8px 16px;font-size:12px;font-weight:600;color:#00f5a0;border-bottom:2px solid #00f5a0;">B2B SaaS</div>
        <div style="padding:8px 16px;font-size:12px;color:#3a4a6a;">OTT Streaming</div>
        <div style="padding:8px 16px;font-size:12px;color:#3a4a6a;">EdTech</div>
      </div>
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="background:#0f1829;border:1px solid #1a2540;border-radius:8px;padding:6px 14px;font-size:12px;color:#3a4a6a;">🔍 Search accounts...</div>
        <div style="width:28px;height:28px;background:linear-gradient(135deg,#00f5a0,#00c8f8);border-radius:50%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── KPI cards ─────────────────────────────────
def render_kpis(pred_df):
    high = pred_df[pred_df["risk_tier"] == "High"]
    mrr_risk = high["monthly_charges_inr"].sum() if "monthly_charges_inr" in pred_df else 0
    arr_risk = mrr_risk * 12
    n_crit   = len(high)
    recovery = pred_df[pred_df["risk_tier"]=="Low"]["monthly_charges_inr"].sum() * 0.12 if "monthly_charges_inr" in pred_df else 0

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;">
      <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:24px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:12px;">Total ARR at Risk</div>
        <div style="font-size:28px;font-weight:700;color:#e8f0ff;margin-bottom:6px;">{fmt_inr(arr_risk)}</div>
        <div style="font-size:11px;color:#4a5a7a;">{fmt_inr(mrr_risk)} Monthly Projected Leakage</div>
        <div style="height:3px;background:linear-gradient(90deg,#ff4d6d,#ff9a3c);border-radius:2px;margin-top:16px;width:60%;"></div>
      </div>
      <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:24px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:12px;">Critical Accounts</div>
        <div style="display:flex;align-items:baseline;gap:10px;">
          <div style="font-size:28px;font-weight:700;color:#e8f0ff;">{n_crit}</div>
          <div style="font-size:12px;color:#ff4d6d;">+{max(0,n_crit-5)} this week</div>
        </div>
        <div style="font-size:11px;color:#4a5a7a;margin-top:6px;">Accounts above 65% Churn Probability</div>
        <div style="height:3px;background:linear-gradient(90deg,#ff4d6d,#f5a623);border-radius:2px;margin-top:16px;width:40%;"></div>
      </div>
      <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:24px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#00f5a0;margin-bottom:12px;">Active Recovery</div>
        <div style="font-size:28px;font-weight:700;color:#e8f0ff;margin-bottom:6px;">{fmt_inr(recovery)}</div>
        <div style="font-size:11px;color:#4a5a7a;">{fmt_inr(recovery*12)} ARR Under Automated Mitigation</div>
        <div style="height:3px;background:linear-gradient(90deg,#00f5a0,#00c8f8);border-radius:2px;margin-top:16px;width:80%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Density histogram ─────────────────────────
def density_chart(pred_df):
    probs = pred_df["churn_probability"].values
    fig = go.Figure()
    for data, name, color in [
        (probs[probs < 0.35],                              "Safe",     "#00f5a0"),
        (probs[(probs>=0.35)&(probs<0.65)],               "At Risk",  "#f5a623"),
        (probs[probs >= 0.65],                             "Critical", "#ff4d6d"),
    ]:
        if len(data):
            fig.add_trace(go.Histogram(x=data, nbinsx=15, name=name,
                marker_color=color, opacity=0.85))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#0f1829", plot_bgcolor="#0f1829",
        font=dict(color="#4a5a7a", family="Space Grotesk"),
        xaxis=dict(title="Churn Probability Score", showgrid=False, tickformat=".0%"),
        yaxis=dict(title="Customer Count", showgrid=True, gridcolor="#1a2540"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10,r=10,t=30,b=40), height=260,
    )
    return fig


# ── Critical threat list ──────────────────────
def render_threat_list(pred_df):
    high = pred_df[pred_df["risk_tier"]=="High"].sort_values("churn_probability",ascending=False).head(6)
    # Header
    n_high = len(high)
    st.markdown(f"""
    <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#ff4d6d;">⚠ Critical Threat List</div>
        <div style="background:#ff4d6d;color:#fff;font-size:10px;font-weight:700;border-radius:6px;padding:3px 10px;">{n_high} TOTAL</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if high.empty:
        st.markdown('<div style="color:#3a4a6a;font-size:12px;padding:8px 16px;">No critical accounts detected.</div>', unsafe_allow_html=True)
        return
    # Render each card separately to avoid f-string HTML escaping issues
    for _, row in high.iterrows():
        prob = float(row.get("churn_probability", 0))
        cid  = str(row.get("customer_id", "—"))
        seg  = str(row.get("business_segment", "—"))
        cont = str(row.get("contract_type", "—"))
        mrr  = fmt_inr_short(float(row.get("monthly_charges_inr", 0)))
        drop = "HIGH" if float(row.get("usage_drop_rate_pct", 0)) > 40 else "LOW"
        vel  = "HIGH" if float(row.get("account_velocity_score", 0)) > 50 else "LOW"
        dc   = "#ff4d6d" if drop == "HIGH" else "#00f5a0"
        vc   = "#00f5a0" if vel  == "HIGH" else "#f5a623"
        pct  = f"{prob*100:.0f}"
        st.markdown(
            "<div style='background:#0b1120;border:1px solid #1a2540;border-radius:10px;"
            "padding:14px;margin:0 0 8px;'>"
            "<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;'>"
            f"<div><div style='font-size:13px;font-weight:600;color:#e8f0ff;'>{cid}</div>"
            f"<div style='font-size:10px;color:#3a4a6a;margin-top:2px;'>{seg} · {cont}</div></div>"
            f"<div style='text-align:right;'><div style='font-size:12px;font-weight:600;color:#ff4d6d;'>{mrr}</div>"
            f"<div style='font-size:10px;color:#ff4d6d;'>{pct}% Risk</div></div>"
            "</div>"
            "<div style='display:flex;gap:6px;margin-top:8px;'>"
            f"<div style='font-size:9px;letter-spacing:0.8px;background:rgba(255,77,109,0.12);color:{dc};"
            f"border-radius:4px;padding:2px 7px;'>DROP-OFF {drop}</div>"
            f"<div style='font-size:9px;letter-spacing:0.8px;background:rgba(0,245,160,0.08);color:{vc};"
            f"border-radius:4px;padding:2px 7px;'>VELOCITY {vel}</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ── Gauge ─────────────────────────────────────
def gauge_chart(prob, cid):
    color = "#ff4d6d" if prob>=0.65 else "#f5a623" if prob>=0.35 else "#00f5a0"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number={"suffix":"%","font":{"size":48,"color":"#e8f0ff","family":"Space Grotesk"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#1a2540","tickwidth":1},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"#0f1829","borderwidth":0,
            "steps":[
                {"range":[0,35],"color":"rgba(0,245,160,0.06)"},
                {"range":[35,65],"color":"rgba(245,166,35,0.06)"},
                {"range":[65,100],"color":"rgba(255,77,109,0.06)"},
            ],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.85,"value":prob*100},
        },
        title={"text":f"Churn Probability · {cid}","font":{"size":13,"color":"#4a5a7a"}},
    ))
    fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
                      margin=dict(l=20,r=20,t=40,b=10),height=300)
    return fig


# ── SHAP bar ──────────────────────────────────
def shap_bar(shap_impacts):
    items  = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [k.replace("num__","").replace("cat__","").replace("_"," ").title() for k,_ in items]
    values = [v for _,v in items]
    colors = ["#ff4d6d" if v>0 else "#00f5a0" for v in values]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(color="#4a5a7a",size=10,family="JetBrains Mono"),
    ))
    fig.update_layout(
        paper_bgcolor="#0f1829", plot_bgcolor="#0f1829",
        font=dict(color="#7a8aaa",family="Space Grotesk"),
        xaxis=dict(showgrid=True,gridcolor="#1a2540",zeroline=True,zerolinecolor="#2a3a5a"),
        yaxis=dict(showgrid=False),
        title=dict(text="Key Contributing Factors (SHAP Impact)",font=dict(size=12,color="#4a5a7a")),
        margin=dict(l=10,r=70,t=40,b=20), height=380,
    )
    return fig


# ── Tab 1: Portfolio ──────────────────────────
def tab_portfolio(run_clicked):
    render_topbar()
    st.markdown("""
    <div style="padding:28px 32px 20px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <h1 style="font-size:22px;font-weight:700;color:#e8f0ff;margin:0 0 6px;">Portfolio Revenue-at-Risk Arena</h1>
          <div style="font-size:13px;color:#3a4a6a;">Real-time financial impact projection across Indian business clusters</div>
        </div>
        <div style="display:flex;gap:8px;margin-top:4px;">
          <div style="background:#00f5a0;color:#080c14;font-size:11px;font-weight:700;border-radius:6px;padding:5px 14px;letter-spacing:0.5px;">LIVE FEED</div>
          <div style="background:#0f1829;border:1px solid #1a2540;color:#4a5a7a;font-size:11px;border-radius:6px;padding:5px 14px;">HISTORICAL</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    csv_bytes = st.session_state.get("sample_csv") or st.session_state.get("uploaded_csv")
    if not csv_bytes:
        st.markdown("""
        <div style="margin:0 32px;background:#0f1829;border:1px dashed #1a2540;border-radius:14px;padding:56px;text-align:center;">
          <div style="font-size:36px;margin-bottom:14px;">⊞</div>
          <div style="font-size:14px;color:#3a4a6a;">Initialise the model, then load Indian client data from the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if run_clicked:
        with st.spinner("Running churn analysis pipeline..."):
            results = api_predict_batch(csv_bytes)
        if results:
            st.session_state["pred_df"] = pd.DataFrame(results)

    pred_df = st.session_state.get("pred_df")
    if pred_df is None:
        st.markdown('<div style="margin:0 32px;color:#3a4a6a;font-size:13px;padding-top:16px;">Click "Run Advanced Churn Analysis" in the sidebar to begin.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    render_kpis(pred_df)

    col_hist, col_threat = st.columns([3,2], gap="large")
    with col_hist:
        st.markdown("""
        <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px 20px 10px;">
          <div style="font-size:14px;font-weight:600;color:#e8f0ff;margin-bottom:2px;">Churn Score Density Spread</div>
          <div style="font-size:11px;color:#3a4a6a;margin-bottom:4px;">Probability distribution across user cohorts</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(density_chart(pred_df), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
    with col_threat:
        render_threat_list(pred_df)

    st.markdown("<br>", unsafe_allow_html=True)
    if "business_segment" in pred_df.columns and "payment_method" in pred_df.columns:
        sc1, sc2 = st.columns(2, gap="large")
        for col, field, title in [(sc1,"business_segment","Avg Risk by Segment"),(sc2,"payment_method","Avg Risk by Payment")]:
            grp = pred_df.groupby(field)["churn_probability"].mean().reset_index().sort_values("churn_probability")
            fig = go.Figure(go.Bar(
                x=grp["churn_probability"], y=grp[field], orientation="h",
                marker=dict(color=grp["churn_probability"],
                    colorscale=[[0,"#00f5a0"],[0.5,"#f5a623"],[1,"#ff4d6d"]],showscale=False),
                text=[f"{v*100:.1f}%" for v in grp["churn_probability"]],
                textposition="outside",textfont=dict(color="#4a5a7a",size=10),
            ))
            fig.update_layout(
                paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
                font=dict(color="#7a8aaa",family="Space Grotesk"),
                title=dict(text=title,font=dict(size=12,color="#4a5a7a")),
                xaxis=dict(showgrid=True,gridcolor="#1a2540",tickformat=".0%"),
                yaxis=dict(showgrid=False),
                margin=dict(l=10,r=50,t=40,b=10),height=240,
            )
            with col:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 2: Client Auditor ─────────────────────
def tab_auditor():
    render_topbar()
    st.markdown("""
    <div style="padding:28px 32px 20px;">
      <h1 style="font-size:22px;font-weight:700;color:#e8f0ff;margin:0 0 6px;">Client Risk Auditor Deep-Dive</h1>
      <div style="font-size:13px;color:#3a4a6a;">Individual SHAP explainability + Indian retention action matrix</div>
    </div>
    """, unsafe_allow_html=True)

    pred_df = st.session_state.get("pred_df")
    if pred_df is None:
        st.markdown('<div style="margin:0 32px;color:#3a4a6a;font-size:13px;">Run the churn analysis in Tab 1 first.</div>', unsafe_allow_html=True)
        return

    csv_bytes = st.session_state.get("sample_csv") or st.session_state.get("uploaded_csv")
    source_df = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes else pred_df

    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    all_ids = pred_df["customer_id"].tolist() if "customer_id" in pred_df.columns else list(range(len(pred_df)))
    selected_id = st.selectbox("Select Client ID", all_ids, label_visibility="collapsed")

    row_pred   = pred_df[pred_df["customer_id"]==selected_id].iloc[0]
    row_source = source_df[source_df["customer_id"]==selected_id].iloc[0] if "customer_id" in source_df.columns else source_df.iloc[0]

    prob = float(row_pred.get("churn_probability",0))
    tier = row_pred.get("risk_tier","Low")
    tier_color = {"High":"#ff4d6d","Medium":"#f5a623","Low":"#00f5a0"}.get(tier,"#00f5a0")
    tier_bg    = {"High":"rgba(255,77,109,0.1)","Medium":"rgba(245,166,35,0.1)","Low":"rgba(0,245,160,0.1)"}.get(tier,"rgba(0,245,160,0.1)")

    col_g, col_m = st.columns([1,1], gap="large")
    with col_g:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob, selected_id), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;margin-top:4px;"><span style="background:{tier_bg};color:{tier_color};border:1px solid {tier_color};border-radius:20px;padding:4px 18px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{tier} RISK</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m:
        st.markdown(f"""
        <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:24px;">
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:16px;">Account Profile</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Monthly MRR</div><div style="font-size:20px;font-weight:700;color:#e8f0ff;">₹{int(row_pred.get('monthly_charges_inr',0)):,}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Tenure</div><div style="font-size:20px;font-weight:700;color:#e8f0ff;">{int(row_pred.get('tenure_months',0))}m</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Contract</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('contract_type','—')}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Payment Method</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('payment_method','—')}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">UPI Failures</div><div style="font-size:20px;font-weight:700;color:{'#ff4d6d' if row_pred.get('upi_failure_count',0)>=2 else '#c8d4ee'};">{int(row_pred.get('upi_failure_count',0))}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Support Tickets</div><div style="font-size:20px;font-weight:700;color:{'#ff4d6d' if row_pred.get('support_tickets_open',0)>=3 else '#c8d4ee'};">{int(row_pred.get('support_tickets_open',0))}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Usage Drop</div><div style="font-size:14px;font-weight:600;color:{'#ff4d6d' if row_pred.get('usage_drop_rate_pct',0)>=40 else '#f5a623'};">{row_pred.get('usage_drop_rate_pct',0):.1f}%</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Segment</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('business_segment','—')}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Get SHAP Explanation + Retention Plan", type="primary"):
        NUM = ["monthly_charges_inr","tenure_months","payment_failures_last_3m","support_tickets_open",
               "usage_drop_rate_pct","account_velocity_score","num_products_subscribed",
               "days_since_last_login","upi_failure_count","contract_value_inr"]
        CAT = ["payment_method","contract_type","business_segment","internet_tier","has_gst_invoice"]
        payload = {"customer_id": selected_id}
        for f in NUM: payload[f] = float(row_source.get(f, row_pred.get(f, 0)))
        for f in CAT: payload[f] = str(row_source.get(f, row_pred.get(f, "")))

        with st.spinner("Calculating SHAP feature impacts..."):
            result = api_explain(payload)

        if result:
            shap_impacts = result.get("shap_impacts", {})
            col_s, col_r = st.columns([3,2], gap="large")

            with col_s:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px 20px 10px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">Key Contributing Factors</div>', unsafe_allow_html=True)
                st.plotly_chart(shap_bar(shap_impacts), use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:14px;">Retention Action Matrix · India</div>', unsafe_allow_html=True)
                for a in get_alerts(shap_impacts, row_source.to_dict()):
                    st.markdown(f"""
                    <div style="background:#0b1120;border-left:3px solid #00f5a0;border-radius:0 10px 10px 0;padding:12px 14px;margin-bottom:10px;">
                      <div style="font-size:12px;font-weight:600;color:#00f5a0;margin-bottom:5px;">{a['icon']} {a['title']}</div>
                      <div style="font-size:11px;color:#4a5a7a;line-height:1.6;">{a['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────
def main():
    for k in ["pred_df","sample_csv","uploaded_csv","model_ready"]:
        if k not in st.session_state: st.session_state[k] = None

    run_clicked = render_sidebar()

    tab1, tab2 = st.tabs(["📈 Portfolio Revenue-at-Risk Overview","🔍 Client Risk Auditor Deep-Dive"])
    with tab1: tab_portfolio(run_clicked)
    with tab2: tab_auditor()

if __name__ == "__main__":
    main()
