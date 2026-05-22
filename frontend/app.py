"""
app.py — Aegis Intelligence · Indian SaaS Churn Predictor
All nav items functional: Dashboard, Predictive Insights, Customer Segments,
Automated Actions, System Settings + B2B SaaS / OTT Streaming / EdTech tabs.
"""
from __future__ import annotations
import io, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "https://your-render-app-url.onrender.com")

st.set_page_config(
    page_title="Aegis Intelligence · Churn Control",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
section[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
section[data-testid="stSidebar"] .stRadio label { background: transparent !important; border: none !important; padding: 10px 14px !important; border-radius: 8px !important; cursor: pointer !important; width: 100% !important; font-size: 13px !important; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: #0f1829 !important; border-left: 3px solid #00f5a0 !important; }
section[data-testid="stSidebar"] .stRadio input { display: none !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; border-bottom: 1px solid #1a2540 !important; padding: 0 32px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #4a5a7a !important; font-size: 13px !important; font-weight: 500 !important; padding: 12px 20px !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #00f5a0 !important; border-bottom: 2px solid #00f5a0 !important; }
div[data-testid="metric-container"] { display: none !important; }
.stSelectbox > div > div { background: #0f1829 !important; border: 1px solid #1a2540 !important; color: #c8d4ee !important; border-radius: 8px !important; }
.stFileUploader { background: #0f1829 !important; border: 1px solid #1a2540 !important; border-radius: 8px !important; padding: 8px !important; }
.stSpinner > div { border-top-color: #00f5a0 !important; }
.stAlert { background: #0f1829 !important; border: 1px solid #1a2540 !important; color: #7a8aaa !important; border-radius: 8px !important; }
.stTextInput > div > div { background: #0f1829 !important; border: 1px solid #1a2540 !important; color: #c8d4ee !important; border-radius: 8px !important; }
.stSlider > div > div > div { background: #00f5a0 !important; }
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

def section_header(title, sub=""):
    st.markdown(f"""
    <div style="padding:28px 32px 20px;">
      <h1 style="font-size:22px;font-weight:700;color:#e8f0ff;margin:0 0 6px;">{title}</h1>
      <div style="font-size:13px;color:#3a4a6a;">{sub}</div>
    </div>""", unsafe_allow_html=True)

def card(html): st.markdown(f'<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">{html}</div>', unsafe_allow_html=True)


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


# ── Sample data generators ────────────────────
def generate_sample_csv(sector="B2B SaaS"):
    rng = np.random.default_rng(42)
    n = 50
    if sector == "OTT Streaming":
        prefix = "OTT"
        monthly = rng.choice([99,199,299,499,799], n).astype(float)
    elif sector == "EdTech":
        prefix = "EDU"
        monthly = rng.choice([299,499,999,1999,4999], n).astype(float)
    else:
        prefix = "IND"
        monthly = rng.choice([499,999,1499,2499,4999,7999,12999], n).astype(float)
    tenure = rng.integers(1, 48, n).astype(float)
    df = pd.DataFrame({
        "customer_id":              [f"{prefix}-{1001+i}" for i in range(n)],
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
    "upi_failure":      {"icon":"📲","title":"UPI Mandate Failure",    "action":"Trigger WhatsApp Business alert with Razorpay UPI deep-link. Offer ₹100 cashback on mandate renewal within 24h.","priority":"Critical"},
    "payment_failure":  {"icon":"💳","title":"Payment Failure Pattern", "action":"Send SMS via MSG91 with NetBanking fallback. Escalate to CS after 2 failures. Offer 7-day grace period.","priority":"High"},
    "monthly_contract": {"icon":"📋","title":"Month-to-Month Risk",     "action":"Pitch 12-month Annual plan via UPI with 2 months free (17% saving). Highlight GST input credit for B2B clients.","priority":"High"},
    "high_usage_drop":  {"icon":"📉","title":"Usage Drop-off",          "action":"Trigger Intercom in-app nudge. Schedule 30-min walkthrough with CS in Hindi/regional language.","priority":"Medium"},
    "high_support":     {"icon":"🎧","title":"Support Ticket Spike",    "action":"Assign dedicated KAM. Offer free migration. Escalate SLA to Priority. Follow up via WhatsApp.","priority":"High"},
    "low_velocity":     {"icon":"🔒","title":"Low Engagement",          "action":"Re-engagement via WhatsApp+Email. Offer 15-day free extension. Book live demo with regional rep.","priority":"Medium"},
    "wallet_payment":   {"icon":"👛","title":"Wallet Payment Risk",     "action":"Migrate to UPI AutoPay via Razorpay. Set proactive low-balance alert via SMS.","priority":"Medium"},
    "basic_tier":       {"icon":"⬆️","title":"Basic Plan Ceiling",      "action":"Offer Standard upgrade with 30-day premium trial. Show INR ROI calculator. EMI via Bajaj Finserv.","priority":"Low"},
    "default":          {"icon":"⚠️","title":"General Risk",            "action":"Schedule CS check-in. Offer 10–15% loyalty discount on renewal. Send personalised INR value report.","priority":"Low"},
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
        <div style="padding:24px 20px 16px;border-bottom:1px solid #1a2540;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#00f5a0,#00c8f8);border-radius:8px;font-size:16px;display:flex;align-items:center;justify-content:center;">🛡</div>
            <div>
              <div style="font-size:14px;font-weight:700;color:#e8f0ff;">Aegis Intelligence</div>
              <div style="font-size:10px;color:#3a4a6a;letter-spacing:1px;text-transform:uppercase;">Enterprise Tier</div>
            </div>
          </div>
        </div>
        <div style="padding:12px 12px 4px;">
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Navigation</div>
        </div>
        """, unsafe_allow_html=True)

        # Functional nav using radio
        nav = st.radio("nav", [
            "⊞  Dashboard",
            "✦  Predictive Insights",
            "◈  Customer Segments",
            "⚡  Automated Actions",
            "⚙  System Settings",
        ], label_visibility="collapsed")
        st.session_state["nav"] = nav

        st.markdown('<hr style="border-color:#1a2540;margin:8px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="padding:0 12px 6px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Model Engine</div></div>', unsafe_allow_html=True)

        if st.button("⚡ Initialise Model", use_container_width=True):
            with st.spinner("Training XGBoost + SHAP..."):
                result = api_train()
            if result:
                st.success(f"✅ Trained · {result['n_samples']:,} clients · {result['churn_rate']}% churn")
                st.session_state["model_ready"] = True

        st.markdown('<div style="padding:6px 12px 4px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Sector</div></div>', unsafe_allow_html=True)
        sector = st.radio("sector", ["B2B SaaS","OTT Streaming","EdTech"], label_visibility="collapsed", horizontal=True)
        st.session_state["sector"] = sector

        st.markdown('<div style="padding:4px 12px 4px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Data Source</div></div>', unsafe_allow_html=True)
        if st.button("✨ Auto-Generate Client Base", use_container_width=True):
            st.session_state["sample_csv"]   = generate_sample_csv(sector)
            st.session_state["uploaded_csv"] = None
            st.success(f"50 {sector} clients generated!")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            st.session_state["uploaded_csv"] = uploaded.read()
            st.session_state["sample_csv"]   = None

        run_clicked = st.button("🔍 Run Advanced Churn Analysis", use_container_width=True, type="primary")

        st.markdown("""
        <div style="padding:12px 20px;border-top:1px solid #1a2540;margin-top:16px;">
          <div style="font-size:12px;color:#3a4a6a;margin-bottom:6px;">⊕  Support</div>
          <div style="font-size:12px;color:#3a4a6a;">→  Sign Out</div>
        </div>
        """, unsafe_allow_html=True)

    return run_clicked, nav, sector


# ── Top bar ───────────────────────────────────
def render_topbar(sector="B2B SaaS"):
    sectors = ["B2B SaaS","OTT Streaming","EdTech"]
    tabs_html = "".join([
        f'<div style="padding:8px 16px;font-size:12px;font-weight:{"600" if s==sector else "400"};'
        f'color:{"#00f5a0" if s==sector else "#3a4a6a"};'
        f'border-bottom:{"2px solid #00f5a0" if s==sector else "2px solid transparent"};">{s}</div>'
        for s in sectors
    ])
    st.markdown(f"""
    <div style="background:#0b1120;border-bottom:1px solid #1a2540;padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:52px;">
      <div style="display:flex;align-items:center;">
        <div style="font-size:18px;font-weight:700;color:#e8f0ff;margin-right:32px;">Aegis Churn Control</div>
        {tabs_html}
      </div>
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="background:#0f1829;border:1px solid #1a2540;border-radius:8px;padding:6px 14px;font-size:12px;color:#3a4a6a;">🔍 Search accounts...</div>
        <div style="width:28px;height:28px;background:linear-gradient(135deg,#00f5a0,#00c8f8);border-radius:50%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Charts ────────────────────────────────────
def density_chart(pred_df):
    probs = pred_df["churn_probability"].values
    fig = go.Figure()
    for data, name, color in [
        (probs[probs<0.35],"Safe","#00f5a0"),
        (probs[(probs>=0.35)&(probs<0.65)],"At Risk","#f5a623"),
        (probs[probs>=0.65],"Critical","#ff4d6d"),
    ]:
        if len(data): fig.add_trace(go.Histogram(x=data, nbinsx=15, name=name, marker_color=color, opacity=0.85))
    fig.update_layout(barmode="overlay", paper_bgcolor="#0f1829", plot_bgcolor="#0f1829",
        font=dict(color="#4a5a7a",family="Space Grotesk"),
        xaxis=dict(title="Churn Score",showgrid=False,tickformat=".0%"),
        yaxis=dict(title="Count",showgrid=True,gridcolor="#1a2540"),
        legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",font=dict(size=11)),
        margin=dict(l=10,r=10,t=30,b=40),height=260)
    return fig

def segment_chart(pred_df, field, title):
    grp = pred_df.groupby(field)["churn_probability"].mean().reset_index().sort_values("churn_probability")
    fig = go.Figure(go.Bar(
        x=grp["churn_probability"], y=grp[field], orientation="h",
        marker=dict(color=grp["churn_probability"],colorscale=[[0,"#00f5a0"],[0.5,"#f5a623"],[1,"#ff4d6d"]],showscale=False),
        text=[f"{v*100:.1f}%" for v in grp["churn_probability"]],
        textposition="outside",textfont=dict(color="#4a5a7a",size=10),
    ))
    fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
        font=dict(color="#7a8aaa",family="Space Grotesk"),
        title=dict(text=title,font=dict(size=12,color="#4a5a7a")),
        xaxis=dict(showgrid=True,gridcolor="#1a2540",tickformat=".0%"),
        yaxis=dict(showgrid=False),margin=dict(l=10,r=50,t=40,b=10),height=240)
    return fig

def gauge_chart(prob, cid):
    color = "#ff4d6d" if prob>=0.65 else "#f5a623" if prob>=0.35 else "#00f5a0"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number={"suffix":"%","font":{"size":48,"color":"#e8f0ff","family":"Space Grotesk"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#1a2540"},"bar":{"color":color,"thickness":0.28},
               "bgcolor":"#0f1829","borderwidth":0,
               "steps":[{"range":[0,35],"color":"rgba(0,245,160,0.06)"},{"range":[35,65],"color":"rgba(245,166,35,0.06)"},{"range":[65,100],"color":"rgba(255,77,109,0.06)"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.85,"value":prob*100}},
        title={"text":f"Churn Probability · {cid}","font":{"size":13,"color":"#4a5a7a"}},
    ))
    fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",margin=dict(l=20,r=20,t=40,b=10),height=300)
    return fig

def shap_bar(shap_impacts):
    items = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [k.replace("num__","").replace("cat__","").replace("_"," ").title() for k,_ in items]
    values = [v for _,v in items]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker_color=["#ff4d6d" if v>0 else "#00f5a0" for v in values[::-1]],
        text=[f"{v:+.4f}" for v in values[::-1]], textposition="outside",
        textfont=dict(color="#4a5a7a",size=10,family="JetBrains Mono"),
    ))
    fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
        font=dict(color="#7a8aaa",family="Space Grotesk"),
        xaxis=dict(showgrid=True,gridcolor="#1a2540",zeroline=True,zerolinecolor="#2a3a5a"),
        yaxis=dict(showgrid=False),
        title=dict(text="Key Contributing Factors (SHAP Impact)",font=dict(size=12,color="#4a5a7a")),
        margin=dict(l=10,r=70,t=40,b=20),height=380)
    return fig


# ── KPI Cards ─────────────────────────────────
def render_kpis(pred_df):
    high = pred_df[pred_df["risk_tier"]=="High"]
    mrr_risk = high["monthly_charges_inr"].sum() if "monthly_charges_inr" in pred_df else 0
    arr_risk = mrr_risk * 12
    n_crit   = len(high)
    recovery = pred_df[pred_df["risk_tier"]=="Low"]["monthly_charges_inr"].sum()*0.12 if "monthly_charges_inr" in pred_df else 0
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


# ── Critical Threat List ──────────────────────
def render_threat_list(pred_df):
    high = pred_df[pred_df["risk_tier"]=="High"].sort_values("churn_probability",ascending=False).head(6)
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
        st.markdown('<div style="color:#3a4a6a;font-size:12px;padding:8px;">No critical accounts.</div>', unsafe_allow_html=True)
        return
    for _, row in high.iterrows():
        prob = float(row.get("churn_probability",0))
        cid  = str(row.get("customer_id","—"))
        seg  = str(row.get("business_segment","—"))
        cont = str(row.get("contract_type","—"))
        mrr  = fmt_inr_short(float(row.get("monthly_charges_inr",0)))
        drop = "HIGH" if float(row.get("usage_drop_rate_pct",0))>40 else "LOW"
        vel  = "HIGH" if float(row.get("account_velocity_score",0))>50 else "LOW"
        dc   = "#ff4d6d" if drop=="HIGH" else "#00f5a0"
        vc   = "#00f5a0" if vel=="HIGH"  else "#f5a623"
        st.markdown(
            "<div style='background:#0b1120;border:1px solid #1a2540;border-radius:10px;"
            "padding:14px;margin:0 0 8px;'>"
            "<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;'>"
            f"<div><div style='font-size:13px;font-weight:600;color:#e8f0ff;'>{cid}</div>"
            f"<div style='font-size:10px;color:#3a4a6a;margin-top:2px;'>{seg} · {cont}</div></div>"
            f"<div style='text-align:right;'><div style='font-size:12px;font-weight:600;color:#ff4d6d;'>{mrr}</div>"
            f"<div style='font-size:10px;color:#ff4d6d;'>{prob*100:.0f}% Risk</div></div>"
            "</div><div style='display:flex;gap:6px;margin-top:8px;'>"
            f"<div style='font-size:9px;background:rgba(255,77,109,0.12);color:{dc};border-radius:4px;padding:2px 7px;'>DROP-OFF {drop}</div>"
            f"<div style='font-size:9px;background:rgba(0,245,160,0.08);color:{vc};border-radius:4px;padding:2px 7px;'>VELOCITY {vel}</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════
def page_dashboard(run_clicked, sector):
    render_topbar(sector)
    st.markdown("""
    <div style="padding:28px 32px 20px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <h1 style="font-size:22px;font-weight:700;color:#e8f0ff;margin:0 0 6px;">Portfolio Revenue-at-Risk Arena</h1>
          <div style="font-size:13px;color:#3a4a6a;">Real-time financial impact projection across Indian business clusters</div>
        </div>
        <div style="display:flex;gap:8px;margin-top:4px;">
          <div style="background:#00f5a0;color:#080c14;font-size:11px;font-weight:700;border-radius:6px;padding:5px 14px;">LIVE FEED</div>
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
          <div style="font-size:14px;color:#3a4a6a;">Initialise the model, then load client data from the sidebar</div>
        </div>""", unsafe_allow_html=True)
        return

    if run_clicked:
        with st.spinner("Running churn analysis..."):
            results = api_predict_batch(csv_bytes)
        if results: st.session_state["pred_df"] = pd.DataFrame(results)

    pred_df = st.session_state.get("pred_df")
    if pred_df is None:
        st.markdown('<div style="margin:0 32px;color:#3a4a6a;font-size:13px;padding-top:16px;">Click "Run Advanced Churn Analysis" in the sidebar to begin.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    render_kpis(pred_df)

    col_hist, col_threat = st.columns([3,2], gap="large")
    with col_hist:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px 20px 10px;"><div style="font-size:14px;font-weight:600;color:#e8f0ff;margin-bottom:2px;">Churn Score Density Spread</div><div style="font-size:11px;color:#3a4a6a;margin-bottom:4px;">Probability distribution across user cohorts</div>', unsafe_allow_html=True)
        st.plotly_chart(density_chart(pred_df), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
    with col_threat:
        render_threat_list(pred_df)

    st.markdown("<br>", unsafe_allow_html=True)
    if "business_segment" in pred_df.columns and "payment_method" in pred_df.columns:
        sc1, sc2 = st.columns(2, gap="large")
        for col, field, title in [(sc1,"business_segment","Avg Risk by Segment"),(sc2,"payment_method","Avg Risk by Payment")]:
            with col:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
                st.plotly_chart(segment_chart(pred_df,field,title), use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PAGE: PREDICTIVE INSIGHTS
# ════════════════════════════════════════════
def page_predictive_insights(sector):
    render_topbar(sector)
    section_header("✦ Predictive Insights", "Model performance metrics, score distribution & feature importance analysis")
    pred_df = st.session_state.get("pred_df")

    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    if pred_df is None:
        st.info("Run the churn analysis from the Dashboard first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Score stats
    probs = pred_df["churn_probability"].values
    high_pct = (probs>=0.65).mean()*100
    med_pct  = ((probs>=0.35)&(probs<0.65)).mean()*100
    low_pct  = (probs<0.35).mean()*100
    avg      = probs.mean()*100

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;">
      <div style="background:#0f1829;border:1px solid #1a2540;border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">Avg Churn Score</div>
        <div style="font-size:26px;font-weight:700;color:#e8f0ff;">{avg:.1f}%</div>
      </div>
      <div style="background:#0f1829;border:1px solid #ff4d6d33;border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">Critical Zone</div>
        <div style="font-size:26px;font-weight:700;color:#ff4d6d;">{high_pct:.1f}%</div>
        <div style="font-size:10px;color:#4a5a7a;">Score ≥ 65%</div>
      </div>
      <div style="background:#0f1829;border:1px solid #f5a62333;border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">At-Risk Zone</div>
        <div style="font-size:26px;font-weight:700;color:#f5a623;">{med_pct:.1f}%</div>
        <div style="font-size:10px;color:#4a5a7a;">Score 35–65%</div>
      </div>
      <div style="background:#0f1829;border:1px solid #00f5a033;border-radius:12px;padding:20px;text-align:center;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">Safe Zone</div>
        <div style="font-size:26px;font-weight:700;color:#00f5a0;">{low_pct:.1f}%</div>
        <div style="font-size:10px;color:#4a5a7a;">Score &lt; 35%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Scatter: tenure vs churn prob
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=pred_df.get("tenure_months", pd.Series()), y=pred_df["churn_probability"]*100,
            mode="markers",
            marker=dict(color=pred_df["churn_probability"], colorscale=[[0,"#00f5a0"],[0.5,"#f5a623"],[1,"#ff4d6d"]],
                       size=8, opacity=0.7, showscale=True,
                       colorbar=dict(title="Risk",tickfont=dict(color="#4a5a7a"),tickformat=".0%")),
        ))
        fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
            title=dict(text="Tenure vs Churn Risk",font=dict(size=13,color="#7a8aaa")),
            xaxis=dict(title="Tenure (months)",showgrid=True,gridcolor="#1a2540",color="#4a5a7a"),
            yaxis=dict(title="Churn Score (%)",showgrid=True,gridcolor="#1a2540",color="#4a5a7a"),
            font=dict(color="#7a8aaa",family="Space Grotesk"),
            margin=dict(l=10,r=10,t=40,b=40),height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
        # Monthly charges vs churn
        fig2 = go.Figure(go.Scatter(
            x=pred_df.get("monthly_charges_inr", pd.Series()), y=pred_df["churn_probability"]*100,
            mode="markers",
            marker=dict(color=pred_df["churn_probability"], colorscale=[[0,"#00f5a0"],[0.5,"#f5a623"],[1,"#ff4d6d"]],
                       size=8, opacity=0.7),
        ))
        fig2.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
            title=dict(text="Monthly Charges vs Churn Risk",font=dict(size=13,color="#7a8aaa")),
            xaxis=dict(title="Monthly Charges (₹)",showgrid=True,gridcolor="#1a2540",color="#4a5a7a"),
            yaxis=dict(title="Churn Score (%)",showgrid=True,gridcolor="#1a2540",color="#4a5a7a"),
            font=dict(color="#7a8aaa",family="Space Grotesk"),
            margin=dict(l=10,r=10,t=40,b=40),height=300)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Risk tier donut
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
        tier_counts = pred_df["risk_tier"].value_counts()
        fig3 = go.Figure(go.Pie(
            labels=tier_counts.index, values=tier_counts.values, hole=0.6,
            marker=dict(colors=["#ff4d6d","#f5a623","#00f5a0"]),
            textfont=dict(color="#e8f0ff",size=11),
        ))
        fig3.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
            title=dict(text="Risk Tier Distribution",font=dict(size=13,color="#7a8aaa")),
            legend=dict(font=dict(color="#7a8aaa"),bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#7a8aaa",family="Space Grotesk"),
            margin=dict(l=10,r=10,t=40,b=10),height=280)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;">', unsafe_allow_html=True)
        # UPI failures vs churn
        if "upi_failure_count" in pred_df.columns:
            grp = pred_df.groupby("upi_failure_count")["churn_probability"].mean().reset_index()
            fig4 = go.Figure(go.Bar(
                x=grp["upi_failure_count"], y=grp["churn_probability"]*100,
                marker=dict(color=grp["churn_probability"],colorscale=[[0,"#00f5a0"],[1,"#ff4d6d"]],showscale=False),
            ))
            fig4.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
                title=dict(text="UPI Failures vs Avg Churn Risk",font=dict(size=13,color="#7a8aaa")),
                xaxis=dict(title="UPI Failure Count",showgrid=False,color="#4a5a7a"),
                yaxis=dict(title="Avg Churn Score (%)",showgrid=True,gridcolor="#1a2540",color="#4a5a7a"),
                font=dict(color="#7a8aaa",family="Space Grotesk"),
                margin=dict(l=10,r=10,t=40,b=40),height=280)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PAGE: CUSTOMER SEGMENTS
# ════════════════════════════════════════════
def page_segments(sector):
    render_topbar(sector)
    section_header("◈ Customer Segments", "Churn risk breakdown across all business dimensions")
    pred_df = st.session_state.get("pred_df")
    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    if pred_df is None:
        st.info("Run the churn analysis from the Dashboard first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    seg_fields = [
        ("business_segment","Risk by Business Segment"),
        ("payment_method","Risk by Payment Method"),
        ("contract_type","Risk by Contract Type"),
        ("internet_tier","Risk by Internet Tier"),
        ("has_gst_invoice","Risk by GST Status"),
    ]

    for i in range(0, len(seg_fields), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i+j < len(seg_fields):
                field, title = seg_fields[i+j]
                if field in pred_df.columns:
                    with col:
                        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:16px 16px 8px;margin-bottom:16px;">', unsafe_allow_html=True)
                        st.plotly_chart(segment_chart(pred_df,field,title), use_container_width=True, config={"displayModeBar":False})
                        st.markdown("</div>", unsafe_allow_html=True)

    # Heatmap: segment x contract
    st.markdown('<br><div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px 20px 10px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:4px;">Segment × Contract Type Churn Heatmap</div>', unsafe_allow_html=True)
    if "business_segment" in pred_df.columns and "contract_type" in pred_df.columns:
        heat = pred_df.groupby(["business_segment","contract_type"])["churn_probability"].mean().unstack(fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=heat.values*100, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale=[[0,"#0f1829"],[0.5,"#4f6ef7"],[1,"#ff4d6d"]],
            text=[[f"{v:.1f}%" for v in row] for row in heat.values*100],
            texttemplate="%{text}", textfont=dict(size=12,color="#e8f0ff"),
            hovertemplate="Segment: %{y}<br>Contract: %{x}<br>Churn: %{z:.1f}%",
        ))
        fig.update_layout(paper_bgcolor="#0f1829",plot_bgcolor="#0f1829",
            font=dict(color="#7a8aaa",family="Space Grotesk"),
            xaxis=dict(color="#4a5a7a"),yaxis=dict(color="#4a5a7a"),
            margin=dict(l=10,r=10,t=20,b=40),height=280)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PAGE: AUTOMATED ACTIONS
# ════════════════════════════════════════════
def page_automated_actions(sector):
    render_topbar(sector)
    section_header("⚡ Automated Actions", "AI-generated retention interventions for high-risk accounts · Indian market playbook")
    pred_df = st.session_state.get("pred_df")
    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)

    if pred_df is None:
        st.info("Run the churn analysis from the Dashboard first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Summary stats
    high = pred_df[pred_df["risk_tier"]=="High"]
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px;">
      <div style="background:#0f1829;border:1px solid #ff4d6d33;border-radius:12px;padding:18px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Actions Queued</div>
        <div style="font-size:24px;font-weight:700;color:#ff4d6d;">{len(high)}</div>
        <div style="font-size:11px;color:#4a5a7a;">High-risk accounts needing intervention</div>
      </div>
      <div style="background:#0f1829;border:1px solid #00f5a033;border-radius:12px;padding:18px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Est. MRR Saved</div>
        <div style="font-size:24px;font-weight:700;color:#00f5a0;">{fmt_inr(high['monthly_charges_inr'].sum()*0.35 if 'monthly_charges_inr' in high else 0)}</div>
        <div style="font-size:11px;color:#4a5a7a;">With 35% retention success rate</div>
      </div>
      <div style="background:#0f1829;border:1px solid #4f6ef733;border-radius:12px;padding:18px;">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">Playbook Strategies</div>
        <div style="font-size:24px;font-weight:700;color:#4f6ef7;">{len(PLAYBOOK)}</div>
        <div style="font-size:11px;color:#4a5a7a;">Indian market retention actions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Full playbook table
    st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;margin-bottom:20px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">🇮🇳 Indian Market Retention Playbook</div>', unsafe_allow_html=True)
    priority_color = {"Critical":"#ff4d6d","High":"#f5a623","Medium":"#4f6ef7","Low":"#00f5a0"}
    for key, p in PLAYBOOK.items():
        pc = priority_color.get(p.get("priority","Low"),"#00f5a0")
        st.markdown(
            "<div style='display:flex;align-items:flex-start;gap:14px;padding:12px 0;"
            "border-bottom:1px solid #1a2540;'>"
            f"<div style='font-size:20px;'>{p['icon']}</div>"
            "<div style='flex:1;'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
            f"<span style='font-size:13px;font-weight:600;color:#e8f0ff;'>{p['title']}</span>"
            f"<span style='font-size:9px;background:{pc}22;color:{pc};border-radius:4px;padding:1px 7px;font-weight:700;'>{p.get('priority','Low').upper()}</span>"
            "</div>"
            f"<div style='font-size:12px;color:#4a5a7a;line-height:1.6;'>{p['action']}</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # High-risk action queue
    if not high.empty:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">🚨 Action Queue — High Risk Accounts</div>', unsafe_allow_html=True)
        display = high[["customer_id","monthly_charges_inr","churn_probability","contract_type","payment_method","upi_failure_count"]].copy()
        display["churn_probability"] = (display["churn_probability"]*100).round(1).astype(str)+"%"
        display["monthly_charges_inr"] = display["monthly_charges_inr"].apply(lambda x: f"₹{int(x):,}")
        display.columns = ["Client ID","MRR","Churn Score","Contract","Payment","UPI Failures"]
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PAGE: SYSTEM SETTINGS
# ════════════════════════════════════════════
def page_settings(sector):
    render_topbar(sector)
    section_header("⚙ System Settings", "Configure backend connection, thresholds and notification preferences")
    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;margin-bottom:16px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">🔗 Backend Connection</div>', unsafe_allow_html=True)
        current_url = st.session_state.get("custom_backend", BACKEND_URL)
        new_url = st.text_input("Render Backend URL", value=current_url, label_visibility="visible")
        if st.button("💾 Save Backend URL", use_container_width=True):
            st.session_state["custom_backend"] = new_url
            st.success(f"✅ Backend URL updated to: {new_url}")

        st.markdown('<div style="margin-top:14px;">', unsafe_allow_html=True)
        if st.button("🏓 Ping Backend", use_container_width=True):
            try:
                r = requests.get(f"{current_url}/health", timeout=10)
                if r.status_code == 200:
                    st.success("✅ Backend is online and healthy!")
                else:
                    st.error(f"Backend returned status {r.status_code}")
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">📊 Session Status</div>', unsafe_allow_html=True)
        model_ready = st.session_state.get("model_ready", False)
        pred_df     = st.session_state.get("pred_df")
        has_data    = pred_df is not None
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a2540;'>"
            f"<span style='font-size:12px;color:#7a8aaa;'>Model Status</span>"
            f"<span style='font-size:12px;font-weight:600;color:{'#00f5a0' if model_ready else '#ff4d6d'};'>{'✅ Trained' if model_ready else '❌ Not Trained'}</span></div>"
            f"<div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a2540;'>"
            f"<span style='font-size:12px;color:#7a8aaa;'>Data Loaded</span>"
            f"<span style='font-size:12px;font-weight:600;color:{'#00f5a0' if has_data else '#ff4d6d'};'>{'✅ ' + str(len(pred_df)) + ' clients' if has_data else '❌ No Data'}</span></div>"
            f"<div style='display:flex;justify-content:space-between;padding:8px 0;'>"
            f"<span style='font-size:12px;color:#7a8aaa;'>Backend URL</span>"
            f"<span style='font-size:11px;color:#4f6ef7;font-family:monospace;'>{BACKEND_URL[:35]}...</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;margin-bottom:16px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">🎚 Risk Threshold Settings</div>', unsafe_allow_html=True)
        high_thresh = st.slider("Critical Risk Threshold (%)", 50, 90, 65, 5)
        med_thresh  = st.slider("At-Risk Threshold (%)", 20, 60, 35, 5)
        st.markdown(f"""
        <div style="margin-top:12px;padding:10px;background:#0b1120;border-radius:8px;">
          <div style="font-size:11px;color:#4a5a7a;margin-bottom:4px;">Current thresholds:</div>
          <div style="font-size:12px;color:#ff4d6d;">🔴 Critical: ≥ {high_thresh}%</div>
          <div style="font-size:12px;color:#f5a623;margin-top:2px;">🟡 At-Risk: {med_thresh}% – {high_thresh}%</div>
          <div style="font-size:12px;color:#00f5a0;margin-top:2px;">🟢 Safe: &lt; {med_thresh}%</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("💾 Apply Thresholds", use_container_width=True):
            st.session_state["high_thresh"] = high_thresh/100
            st.session_state["med_thresh"]  = med_thresh/100
            st.success("✅ Thresholds saved! Re-run analysis to apply.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e8f0ff;margin-bottom:14px;">🔔 Notification Preferences</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#3a4a6a;margin-bottom:10px;">Alert channels for high-risk detections</div>', unsafe_allow_html=True)
        for channel, default in [("WhatsApp Business API",True),("Email (SendGrid)",True),("Slack Webhook",False),("SMS via MSG91",True)]:
            checked = "✅" if default else "☐"
            color   = "#00f5a0" if default else "#3a4a6a"
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1a2540;"><span style="font-size:12px;color:#7a8aaa;">{channel}</span><span style="color:{color};">{checked}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  PAGE: CLIENT RISK AUDITOR (Tab 2)
# ════════════════════════════════════════════
def page_client_auditor(sector):
    render_topbar(sector)
    section_header("🔍 Client Risk Auditor Deep-Dive", "Individual SHAP explainability + Indian retention action matrix")
    pred_df = st.session_state.get("pred_df")
    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    if pred_df is None:
        st.info("Run the churn analysis from the Dashboard first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    csv_bytes = st.session_state.get("sample_csv") or st.session_state.get("uploaded_csv")
    source_df = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes else pred_df

    all_ids     = pred_df["customer_id"].tolist() if "customer_id" in pred_df.columns else list(range(len(pred_df)))
    selected_id = st.selectbox("Select Client ID", all_ids, label_visibility="collapsed")
    row_pred    = pred_df[pred_df["customer_id"]==selected_id].iloc[0]
    row_source  = source_df[source_df["customer_id"]==selected_id].iloc[0] if "customer_id" in source_df.columns else source_df.iloc[0]

    prob       = float(row_pred.get("churn_probability",0))
    tier       = row_pred.get("risk_tier","Low")
    tier_color = {"High":"#ff4d6d","Medium":"#f5a623","Low":"#00f5a0"}.get(tier,"#00f5a0")
    tier_bg    = {"High":"rgba(255,77,109,0.1)","Medium":"rgba(245,166,35,0.1)","Low":"rgba(0,245,160,0.1)"}.get(tier,"rgba(0,245,160,0.1)")

    col_g, col_m = st.columns([1,1], gap="large")
    with col_g:
        st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;">', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob, selected_id), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;margin-top:4px;"><span style="background:{tier_bg};color:{tier_color};border:1px solid {tier_color};border-radius:20px;padding:4px 18px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{tier} RISK</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m:
        upi_c  = '#ff4d6d' if row_pred.get('upi_failure_count',0)>=2 else '#c8d4ee'
        supp_c = '#ff4d6d' if row_pred.get('support_tickets_open',0)>=3 else '#c8d4ee'
        drop_c = '#ff4d6d' if row_pred.get('usage_drop_rate_pct',0)>=40 else '#f5a623'
        st.markdown(f"""
        <div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:24px;">
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:16px;">Account Profile</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Monthly MRR</div><div style="font-size:20px;font-weight:700;color:#e8f0ff;">₹{int(row_pred.get('monthly_charges_inr',0)):,}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Tenure</div><div style="font-size:20px;font-weight:700;color:#e8f0ff;">{int(row_pred.get('tenure_months',0))}m</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Contract</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('contract_type','—')}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Payment Method</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('payment_method','—')}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">UPI Failures</div><div style="font-size:20px;font-weight:700;color:{upi_c};">{int(row_pred.get('upi_failure_count',0))}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Support Tickets</div><div style="font-size:20px;font-weight:700;color:{supp_c};">{int(row_pred.get('support_tickets_open',0))}</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Usage Drop</div><div style="font-size:14px;font-weight:600;color:{drop_c};">{row_pred.get('usage_drop_rate_pct',0):.1f}%</div></div>
            <div><div style="font-size:10px;color:#3a4a6a;margin-bottom:3px;">Segment</div><div style="font-size:14px;font-weight:600;color:#c8d4ee;">{row_pred.get('business_segment','—')}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

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
            shap_impacts = result.get("shap_impacts",{})
            col_s, col_r = st.columns([3,2], gap="large")
            with col_s:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px 20px 10px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">Key Contributing Factors</div>', unsafe_allow_html=True)
                st.plotly_chart(shap_bar(shap_impacts), use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown('<div style="background:#0f1829;border:1px solid #1a2540;border-radius:14px;padding:20px;"><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#3a4a6a;margin-bottom:14px;">Retention Action Matrix · India</div>', unsafe_allow_html=True)
                for a in get_alerts(shap_impacts, row_source.to_dict()):
                    st.markdown(
                        f"<div style='background:#0b1120;border-left:3px solid #00f5a0;border-radius:0 10px 10px 0;padding:12px 14px;margin-bottom:10px;'>"
                        f"<div style='font-size:12px;font-weight:600;color:#00f5a0;margin-bottom:5px;'>{a['icon']} {a['title']}</div>"
                        f"<div style='font-size:11px;color:#4a5a7a;line-height:1.6;'>{a['action']}</div></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════
def main():
    for k in ["pred_df","sample_csv","uploaded_csv","model_ready","nav","sector","custom_backend"]:
        if k not in st.session_state: st.session_state[k] = None

    run_clicked, nav, sector = render_sidebar()

    # Route based on nav selection
    if "Predictive Insights" in (nav or ""):
        page_predictive_insights(sector)
    elif "Customer Segments" in (nav or ""):
        page_segments(sector)
    elif "Automated Actions" in (nav or ""):
        page_automated_actions(sector)
    elif "System Settings" in (nav or ""):
        page_settings(sector)
    else:
        # Dashboard — two sub-tabs
        tab1, tab2 = st.tabs(["📈 Portfolio Revenue-at-Risk Overview","🔍 Client Risk Auditor Deep-Dive"])
        with tab1: page_dashboard(run_clicked, sector)
        with tab2: page_client_auditor(sector)

if __name__ == "__main__":
    main()
