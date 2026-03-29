"""
dashboard.py
------------
Streamlit monitoring dashboard for the Forex Anomaly Detection System.

Run:
    streamlit run app/dashboard.py

New in this version:
  - Alerts-over-time time series chart
  - User drill-down: click any alert to inspect full triggered rules
  - Graceful empty-state handling throughout
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Anomaly Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }

    .main { background: #0a0e1a; }
    .stApp { background: #0a0e1a; color: #e2e8f0; }

    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-val  { font-size: 2.4rem; font-weight: 800; color: #f8fafc; }
    .metric-lbl  { font-size: 0.8rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; }

    .alert-card  {
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        background: #111827;
    }
    .alert-CRITICAL { border-color: #ef4444; }
    .alert-HIGH     { border-color: #f97316; }
    .alert-MEDIUM   { border-color: #eab308; }
    .alert-LOW      { border-color: #22c55e; }
    .sev-badge {
        display:inline-block; padding:2px 8px; border-radius:4px;
        font-size:0.72rem; font-weight:700; letter-spacing:0.05em;
        font-family:'JetBrains Mono',monospace;
    }
    .sev-CRITICAL { background:#ef444420; color:#ef4444; }
    .sev-HIGH     { background:#f9731620; color:#f97316; }
    .sev-MEDIUM   { background:#eab30820; color:#eab308; }
    .sev-LOW      { background:#22c55e20; color:#22c55e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_alerts(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=10)
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA          = ROOT / "data"
ALERTS_PATH   = DATA / "alerts.jsonl"
STREAM_ALERTS = DATA / "stream_alerts.jsonl"
FEATURES_PATH = DATA / "features.parquet"
MODEL_PATH    = DATA / "isolation_forest.pkl"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    alert_file = st.radio(
        "Alert source",
        ["Batch (alerts.jsonl)", "Stream (stream_alerts.jsonl)"],
    )
    chosen_path = ALERTS_PATH if "Batch" in alert_file else STREAM_ALERTS
    sev_filter = st.multiselect(
        "Severity filter",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM"],
    )
    top_n = st.slider("Top N users", 5, 50, 20)
    auto_refresh = st.checkbox("Auto-refresh (5 s)", value=False)
    if auto_refresh:
        import time; time.sleep(5); st.rerun()
    st.markdown("---")
    st.markdown("**Model path:**")
    st.code(str(MODEL_PATH.relative_to(ROOT)), language="")
    if MODEL_PATH.exists():
        import datetime
        mtime = datetime.datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
        st.caption(f"Trained: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.caption("⚠️ Model not trained yet — run `python main.py run`")

# ── Load data ─────────────────────────────────────────────────────────────────
alerts_df   = load_alerts(chosen_path)
features_df = load_features(FEATURES_PATH)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne;font-weight:800;letter-spacing:-0.03em;"
    "color:#f8fafc;margin-bottom:0'>🔍 Forex Anomaly Monitor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#64748b;font-size:0.9rem;margin-top:4px'>"
    "Real-time unsupervised anomaly detection · Isolation Forest · RuleBasedExplainer</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def metric_card(col, val, label, color="#f8fafc"):
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-val' style='color:{color}'>{val}</div>"
        f"<div class='metric-lbl'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

total_alerts = len(alerts_df)
if not alerts_df.empty and "severity" in alerts_df.columns:
    critical_n   = int((alerts_df["severity"] == "CRITICAL").sum())
    high_n       = int((alerts_df["severity"] == "HIGH").sum())
    avg_score    = f"{alerts_df['risk_score'].mean():.3f}"
    unique_uids  = alerts_df["user_id"].nunique()
else:
    critical_n = high_n = 0
    avg_score  = "—"
    unique_uids = 0

total_users = len(features_df) if not features_df.empty else "—"

metric_card(col1, total_users,   "Total Users")
metric_card(col2, total_alerts,  "Total Alerts")
metric_card(col3, critical_n,    "Critical",   "#ef4444")
metric_card(col4, high_n,        "High Risk",  "#f97316")
metric_card(col5, avg_score,     "Avg Risk Score")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

# ─── LEFT : Alert feed ────────────────────────────────────────────────────────
with left:
    st.markdown("### 🚨 Alert Feed")

    if alerts_df.empty:
        st.info("No alerts found. Run `python main.py run` to generate data and alerts.")
    else:
        filtered = alerts_df[alerts_df["severity"].isin(sev_filter)] if sev_filter else alerts_df
        st.caption(f"Showing {min(30, len(filtered))} of {len(filtered)} alerts")

        for _, row in filtered.head(30).iterrows():
            sev       = row.get("severity", "LOW")
            uid       = row.get("user_id", "?")
            score     = row.get("risk_score", 0.0)
            alert_txt = row.get("alert", "")
            ts        = row.get("timestamp", "")
            feats     = row.get("features_triggered", [])
            if isinstance(feats, str):
                try:
                    feats = json.loads(feats)
                except Exception:
                    feats = []

            feat_badges = " ".join(
                f"<code style='background:#1e293b;color:#94a3b8;"
                f"border-radius:3px;padding:1px 5px;font-size:0.72rem'>{f}</code>"
                for f in feats[:4]
            )
            st.markdown(
                f"<div class='alert-card alert-{sev}'>"
                f"<span class='sev-badge sev-{sev}'>{sev}</span>&nbsp;"
                f"<strong style='color:#e2e8f0'>{uid}</strong>&nbsp;"
                f"<span style='color:#64748b;font-size:0.8rem;font-family:JetBrains Mono'>"
                f"score {score:.4f}</span><br>"
                f"<span style='color:#cbd5e1;font-size:0.85rem'>{alert_txt}</span><br>"
                f"<span style='font-size:0.75rem;color:#475569'>{ts}</span>&nbsp;"
                f"{feat_badges}"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─── RIGHT : Charts ───────────────────────────────────────────────────────────
with right:
    if not alerts_df.empty:
        import altair as alt

        # Risk score histogram
        st.markdown("### 📊 Risk Distribution")
        hist_data = alerts_df[["risk_score"]].copy()
        chart = (
            alt.Chart(hist_data)
            .mark_bar(color="#6366f1", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=20), title="Risk Score"),
                y=alt.Y("count():Q", title="Users"),
                tooltip=["count()"],
            )
            .properties(height=160)
            .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(chart, use_container_width=True)

        # Severity breakdown
        st.markdown("#### Severity Breakdown")
        sev_counts = alerts_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        sev_color_map = {
            "CRITICAL": "#ef4444", "HIGH": "#f97316",
            "MEDIUM": "#eab308",   "LOW":  "#22c55e",
        }
        for _, r in sev_counts.iterrows():
            sev_name = r["Severity"]
            cnt      = r["Count"]
            color    = sev_color_map.get(sev_name, "#94a3b8")
            pct      = cnt / len(alerts_df) * 100
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                f"<span style='width:80px;font-size:0.78rem;color:{color};font-weight:700'>{sev_name}</span>"
                f"<div style='flex:1;background:#1e293b;border-radius:4px;height:14px'>"
                f"<div style='width:{pct:.1f}%;background:{color};border-radius:4px;height:14px'></div></div>"
                f"<span style='width:40px;text-align:right;font-size:0.8rem;color:#94a3b8'>{cnt}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Top-N anomalous users
        st.markdown("### 🏆 Top Anomalous Users")
        top_users = (
            alerts_df.groupby("user_id")["risk_score"]
            .max()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        top_users.columns = ["User ID", "Max Risk Score"]
        top_users["Max Risk Score"] = top_users["Max Risk Score"].round(4)

        bar = (
            alt.Chart(top_users)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("User ID:N", sort="-x", axis=alt.Axis(labelLimit=120)),
                x=alt.X("Max Risk Score:Q", scale=alt.Scale(domain=[0, 1]), title="Risk Score"),
                color=alt.Color(
                    "Max Risk Score:Q",
                    scale=alt.Scale(domain=[0.5, 1.0], range=["#6366f1", "#ef4444"]),
                    legend=None,
                ),
                tooltip=["User ID", "Max Risk Score"],
            )
            .properties(height=max(200, top_n * 22))
            .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Run the pipeline to populate charts.")

# ── Alerts over time (NEW) ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Alerts Over Time")

if not alerts_df.empty and "timestamp" in alerts_df.columns:
    import altair as alt

    # Bin alerts by hour
    time_df = alerts_df.copy()
    time_df["hour"] = time_df["timestamp"].dt.floor("h")
    time_agg = (
        time_df.groupby(["hour", "severity"])
        .size()
        .reset_index(name="count")
    )
    sev_order   = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sev_colors  = ["#ef4444", "#f97316", "#eab308", "#22c55e"]

    time_chart = (
        alt.Chart(time_agg)
        .mark_area(opacity=0.75, interpolate="step-after")
        .encode(
            x=alt.X("hour:T", title="Time (hourly bins)"),
            y=alt.Y("count:Q", title="Alert Count", stack="zero"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=sev_order, range=sev_colors),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["hour:T", "severity:N", "count:Q"],
        )
        .properties(height=220)
        .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(time_chart, use_container_width=True)
else:
    st.info("Timestamp data not available — run `python main.py run` first.")

# ── Feature explorer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Feature Explorer")

if not features_df.empty:
    import altair as alt

    numeric_cols = [
        c for c in features_df.select_dtypes(include=[np.number]).columns
        if c != "user_id"
    ]
    sel_feat = st.selectbox("Select feature to explore", numeric_cols, index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{sel_feat} — Population Distribution**")
        feat_data = features_df[["user_id", sel_feat]].copy()
        hist = (
            alt.Chart(feat_data)
            .mark_bar(color="#6366f1", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(f"{sel_feat}:Q", bin=alt.Bin(maxbins=30), title=sel_feat),
                y=alt.Y("count():Q", title="Users"),
            )
            .properties(height=200)
            .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(hist, use_container_width=True)

    with col_b:
        st.markdown(f"**{sel_feat} — Summary Statistics**")
        stats = features_df[sel_feat].describe().rename("value").round(4)
        st.dataframe(stats, use_container_width=True)
else:
    st.info("Feature data not available. Run `python main.py run` first.")

# ── User drill-down (NEW) ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🕵️ User Drill-Down")

if not alerts_df.empty:
    user_ids = sorted(alerts_df["user_id"].unique().tolist())
    selected_user = st.selectbox("Select a user to inspect", user_ids)

    user_alerts = alerts_df[alerts_df["user_id"] == selected_user].sort_values(
        "risk_score", ascending=False
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts", len(user_alerts))
    c2.metric("Max Risk Score", f"{user_alerts['risk_score'].max():.4f}")
    c3.metric("Severity", user_alerts.iloc[0]["severity"])

    for _, row in user_alerts.iterrows():
        rules = row.get("triggered_rules", [])
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                rules = []
        with st.expander(
            f"[{row['severity']}] score={row['risk_score']:.4f} · {row['timestamp']}",
            expanded=False,
        ):
            st.markdown(f"**Alert:** {row['alert']}")
            if rules:
                rule_df = pd.DataFrame(rules)[["feature", "label", "value", "threshold", "excess_ratio"]]
                st.dataframe(rule_df, use_container_width=True)
            else:
                st.caption("No triggered rules recorded.")
else:
    st.info("No alerts loaded.")

# ── Raw alert table ───────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Raw Alert Table", expanded=False):
    if alerts_df.empty:
        st.info("No alerts yet.")
    else:
        display_cols = ["user_id", "severity", "risk_score", "alert", "features_triggered", "timestamp"]
        avail = [c for c in display_cols if c in alerts_df.columns]
        st.dataframe(alerts_df[avail].head(200), use_container_width=True)

# ── Model Comparison (NEW) ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚖️ Model Comparison — Isolation Forest vs Autoencoder")

COMPARISON_PATH = DATA / "model_comparison.csv"
IF_SCORES_PATH  = DATA / "if_scores.parquet"
AE_SCORES_PATH  = DATA / "ae_scores.parquet"

if COMPARISON_PATH.exists():
    import altair as alt

    comp_df = pd.read_csv(COMPARISON_PATH, index_col=0).reset_index()
    comp_df.columns = [c.replace("_", " ").title() for c in comp_df.columns]

    # Metrics table
    st.dataframe(comp_df, use_container_width=True)

    # Bar chart comparison
    metric_cols = ["Precision", "Recall", "F1 Score", "Roc Auc", "Pr Auc"]
    avail_metrics = [c for c in metric_cols if c in comp_df.columns]

    if avail_metrics:
        melted = comp_df.melt(
            id_vars=["Model"],
            value_vars=avail_metrics,
            var_name="Metric",
            value_name="Score",
        )
        bar = (
            alt.Chart(melted)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusTopLeft=4)
            .encode(
                x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(
                        domain=["Isolation Forest", "Autoencoder"],
                        range=["#6366f1", "#f97316"],
                    ),
                ),
                xOffset="Model:N",
                tooltip=["Model", "Metric", "Score"],
            )
            .properties(height=260)
            .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(bar, use_container_width=True)

    # Score distribution overlay
    if IF_SCORES_PATH.exists() and AE_SCORES_PATH.exists():
        st.markdown("#### Score Distribution Comparison")
        if_sc = pd.read_parquet(IF_SCORES_PATH)[["user_id", "anomaly_score"]].rename(
            columns={"anomaly_score": "score"}
        )
        if_sc["model"] = "Isolation Forest"
        ae_sc = pd.read_parquet(AE_SCORES_PATH)[["user_id", "anomaly_score"]].rename(
            columns={"anomaly_score": "score"}
        )
        ae_sc["model"] = "Autoencoder"
        combined = pd.concat([if_sc, ae_sc], ignore_index=True)

        overlay = (
            alt.Chart(combined)
            .mark_area(opacity=0.5, interpolate="step")
            .encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Anomaly Score"),
                y=alt.Y("count():Q", title="Users", stack=None),
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(
                        domain=["Isolation Forest", "Autoencoder"],
                        range=["#6366f1", "#f97316"],
                    ),
                ),
                tooltip=["model:N", "count():Q"],
            )
            .properties(height=200)
            .configure_axis(grid=False, labelColor="#64748b", titleColor="#64748b")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(overlay, use_container_width=True)
        st.caption("Run `python main.py evaluate` to refresh model comparison metrics.")
else:
    st.info("Model comparison not available yet. Run `python main.py run` to generate it.")

st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.75rem;margin-top:2rem'>"
    "Forex Anomaly Detection System · Isolation Forest + Autoencoder · Unsupervised ML</p>",
    unsafe_allow_html=True,
)
