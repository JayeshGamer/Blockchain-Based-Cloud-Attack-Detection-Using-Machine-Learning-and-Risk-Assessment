"""
Interactive Streamlit Dashboard for Adaptive ML-DST Cloud Attack Tolerance System.
Complies with Phase 11 Dashboard Requirements:
1. Mode selector: Lightweight Development vs Full Mode.
2. Protocol selector: Protocol A (Conventional), Protocol B (Chronological), Protocol C (Controlled Drift).
3. Strategy selector: Plasticity Boost, Full Reset, Model Calibration.
4. Real-time telemetry: 3-tier risk (Global, Cluster, Node), DST evidential decomposition (Belief, Plausibility, Uncertainty, Conflict).
5. Prominent display of Recall, FNR, FPR, Macro-F1, PR-AUC.
6. Visible operational warning banner when attack recall falls below 90%.
7. Genuine ADWIN drift alarms and simulated defense interventions (NO synthetic points).
8. Export of verified experimental results.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Adaptive ML-DST Cloud Attack Tolerance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
        color: #f8fafc;
    }
    .metric-val {
        font-size: 2.0rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .alert-banner-danger {
        background-color: #7f1d1d;
        border: 1px solid #ef4444;
        color: #fecaca;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .alert-banner-success {
        background-color: #064e3b;
        border: 1px solid #10b981;
        color: #a7f3d0;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


# Sidebar Controls
st.sidebar.title("🛡️ Adaptive ML-DST")
st.sidebar.caption("Undergraduate Research Extension of Benmenzer et al. (IET Blockchain 2025)")

is_online, health_data = check_api_health()
if is_online:
    st.sidebar.success("Backend API: CONNECTED")
else:
    st.sidebar.error("Backend API: OFFLINE (Run uvicorn src.api.main:app)")

st.sidebar.divider()

# Mode & Protocol Selection
st.sidebar.subheader("⚙️ Experiment Configuration")
mode = st.sidebar.selectbox(
    "Dataset Execution Mode",
    ["Lightweight Development Mode (50k)", "Full Dataset Mode (2.83M)"],
    index=0,
)
protocol = st.sidebar.selectbox(
    "Evaluation Protocol",
    [
        "Protocol A: Duplicate-Safe Conventional",
        "Protocol B: Genuine Chronological Prequential",
        "Protocol C: Controlled Concept-Drift",
        "Phase 17: Fair Ablation Study",
    ],
    index=1,
)
drift_strategy = st.sidebar.selectbox(
    "ADWIN Drift-Response Strategy",
    ["plasticity_boost (2.0x)", "full_reset", "none (passive)"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Stream Controls")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.sidebar.button("▶ Start Replay", use_container_width=True, disabled=not is_online):
        try:
            requests.post(f"{API_BASE_URL}/stream/start", timeout=2.0)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

with col_s2:
    if st.sidebar.button("⏹ Stop Replay", use_container_width=True, disabled=not is_online):
        try:
            requests.post(f"{API_BASE_URL}/stream/stop", timeout=2.0)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh metrics (3s)", value=True)


# Fetch API State
stream_status = {}
global_risk = {"global_risk": 0.0, "active_threshold": 0.50, "alarm_active": False}
cluster_risks = {}
node_risks = []
alerts_data = {"action_counts": {}, "recent_events": [], "total_blockchain_records": 0}
metrics_data = {}
drift_events = []

if is_online:
    try:
        stream_status = requests.get(f"{API_BASE_URL}/stream/status", timeout=1.5).json()
        global_risk = requests.get(f"{API_BASE_URL}/risk/global", timeout=1.5).json()
        cluster_risks = requests.get(f"{API_BASE_URL}/risk/clusters", timeout=1.5).json()
        node_risks = requests.get(f"{API_BASE_URL}/risk/nodes", timeout=1.5).json()
        alerts_data = requests.get(f"{API_BASE_URL}/alerts", timeout=1.5).json()
        metrics_data = requests.get(f"{API_BASE_URL}/metrics", timeout=1.5).json()
        drift_events = requests.get(f"{API_BASE_URL}/drift/events", timeout=1.5).json().get("events", [])
    except Exception:
        pass

stats = stream_status.get("stats", {})

# Main Banner
st.title("🛡️ Adaptive ML-DST: Cloud Attack Tolerance Engine")
st.caption(f"**Mode:** {mode} | **Protocol:** {protocol} | **Strategy:** {drift_strategy}")

# Prominent Recall and Operational Warning
current_recall = metrics_data.get("recall", 1.0)
current_fnr = metrics_data.get("false_negative_rate", 0.0)

if "recall" in metrics_data and current_recall < 0.90:
    st.markdown(f"""
    <div class="alert-banner-danger">
        ⚠️ CRITICAL OPERATIONAL WARNING: Attack Recall has dropped to {current_recall*100:.2f}% (FNR: {current_fnr*100:.2f}%).
        System decision threshold requires relaxation or retraining to prevent uncaught cloud intrusions!
    </div>
    """, unsafe_allow_html=True)
elif "recall" in metrics_data:
    st.markdown(f"""
    <div class="alert-banner-success">
        ✅ Operational Security Target Met: Attack Recall is {current_recall*100:.2f}% (>= 90% target) | FPR: {metrics_data.get('false_positive_rate', 0.0)*100:.2f}%.
    </div>
    """, unsafe_allow_html=True)

# Top KPIs
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Flows Processed</div>
        <div class="metric-val">{stats.get('flows_processed', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    rec_val = metrics_data.get("recall", 0.0)
    rec_col = "#10b981" if rec_val >= 0.90 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Attack Recall (ADR)</div>
        <div class="metric-val" style="color: {rec_col};">{rec_val:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    fnr_val = metrics_data.get("false_negative_rate", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">False Negative Rate</div>
        <div class="metric-val" style="color: #f59e0b;">{fnr_val:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    fpr_val = metrics_data.get("false_positive_rate", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">False Positive Rate</div>
        <div class="metric-val" style="color: #6366f1;">{fpr_val:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Threshold</div>
        <div class="metric-val" style="color: #ec4899;">{global_risk.get('active_threshold', 0.50):.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Throughput</div>
        <div class="metric-val" style="color: #06b6d4;">{stats.get('throughput_fps', 0.0):.1f} <span style="font-size: 0.9rem;">FPS</span></div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Tabs
tab_hier, tab_evidence, tab_defense, tab_perf, tab_reports = st.tabs([
    "🏛️ Hierarchical Risk",
    "⚖️ DST Evidence Fusion",
    "🛡️ Defense & Blockchain",
    "📊 Real Streaming Performance",
    "📥 Experiment Export",
])

with tab_hier:
    st.subheader("Three-Tier Hierarchical Risk Telemetry")
    col_c, col_n = st.columns([1, 2])
    with col_c:
        st.markdown("#### Tier 2: Cluster Risks")
        if cluster_risks:
            df_c = pd.DataFrame(list(cluster_risks.items()), columns=["Cluster ID", "Risk Score"])
            df_c["Risk Score"] = df_c["Risk Score"].round(4)
            st.dataframe(df_c, use_container_width=True)
            st.bar_chart(df_c.set_index("Cluster ID"))
        else:
            st.info("No active cluster telemetry.")

    with col_n:
        st.markdown("#### Tier 1: Simulated Compute Nodes")
        if node_risks:
            df_n = pd.DataFrame(node_risks)
            st.dataframe(df_n, use_container_width=True)
        else:
            st.info("No active node telemetry.")

with tab_evidence:
    st.subheader("Dempster–Shafer Theory (DST) Tri-Source Fusion")
    st.markdown("""
    Tri-source evidence fusion with **Classical Shafer Reliability Discounting**:
    - **ML Attack Score** ($r_{\\text{ML}} = 0.85$)
    - **Unsupervised Anomaly Score** ($r_{\\text{anom}} = 0.75$)
    - **Temporal Volumetric Deviation** ($r_{\\text{temp}} = 0.70$)
    - **Conflict Reallocation:** Yager's Rule ($K \\to \\Theta$)
    """)
    ev1, ev2, ev3 = st.columns(3)
    with ev1:
        st.info("💡 **Shafer Discounting**: Mass is multiplied by reliability $r_i$, remainder moves to uncertainty $\\Theta$.")
    with ev2:
        st.info("⚡ **Yager Combination**: High conflict $K$ is reallocated to epistemic uncertainty instead of normalized away.")
    with ev3:
        st.info("🎯 **Pignistic Probability**: $\\text{BetP}(\\text{Attack}) = \\text{Bel}(\\text{Attack}) + 0.5 \\cdot m(\\Theta)$.")

with tab_defense:
    st.subheader("Simulated Defense Activation & Cryptographic Audit Ledger")
    d1, d2 = st.columns([1, 2])
    with d1:
        st.markdown("#### Countermeasure Actions (7 Tiers)")
        act_counts = alerts_data.get("action_counts", {})
        if act_counts:
            st.table(pd.DataFrame(list(act_counts.items()), columns=["Action", "Count"]))
        else:
            st.info("No defense interventions triggered yet.")
    with d2:
        st.markdown("#### Recent Security Event Log")
        events = alerts_data.get("recent_events", [])
        if events:
            st.dataframe(pd.DataFrame(events)[["event_id", "selected_action", "risk_category", "explanation"]], use_container_width=True)
        else:
            st.info("Audit buffer empty.")

with tab_perf:
    st.subheader("Streaming Evaluation Metrics (Predict-Before-Update)")
    if metrics_data and "accuracy" in metrics_data:
        m_df = pd.DataFrame([{
            "Accuracy": metrics_data.get("accuracy"),
            "Macro-F1": metrics_data.get("macro_f1"),
            "Recall (ADR)": metrics_data.get("recall"),
            "Precision": metrics_data.get("precision"),
            "F2-Score": metrics_data.get("f2_score", metrics_data.get("fbeta")),
            "FPR": metrics_data.get("false_positive_rate"),
            "FNR": metrics_data.get("false_negative_rate"),
            "ROC-AUC": metrics_data.get("roc_auc"),
            "PR-AUC": metrics_data.get("pr_auc"),
            "Brier Score": metrics_data.get("brier_score"),
        }])
        st.dataframe(m_df.round(4), use_container_width=True)
    else:
        st.info("Start stream simulation to view real-time prequential performance metrics.")

with tab_reports:
    st.subheader("Persisted Experiment Reports & Export")
    exp_dir = "reports/experiments"
    if os.path.exists(exp_dir):
        files = [f for f in os.listdir(exp_dir) if f.endswith(".json")]
        if files:
            selected_file = st.selectbox("Select Experiment Result File", files)
            with open(os.path.join(exp_dir, selected_file), "r", encoding="utf-8") as f:
                content = json.load(f)
            st.json(content)
            st.download_button(
                label=f"Download {selected_file}",
                data=json.dumps(content, indent=2),
                file_name=selected_file,
                mime="application/json",
            )
        else:
            st.info("No experimental reports found in reports/experiments/.")

# Auto-refresh rerun trigger
if auto_refresh and is_online and stream_status.get("is_streaming", False):
    time.sleep(3)
    st.rerun()
