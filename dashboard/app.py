"""Streamlit NOC-style dashboard for IT log anomaly detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_bgl_logs
from src.data.preprocessor import add_severity_score, train_test_split_temporal
from src.features.engineering import build_features, load_features
from src.llm.summarizer import LogSummarizer, build_anomaly_context
from src.models.detector import AnomalyDetector

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Anomaly Detection — NOC Dashboard",
    page_icon="🔍",
    layout="wide",
)

DATA_PATH = Path("data/raw/BGL.log")
FEATURES_PATH = Path("data/processed/features_train.parquet")
MODEL_PATH = Path("models/saved/lof_v1.joblib")
N_ROWS_DEMO = 500_000

SEVERITY_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
    "UNKNOWN": "#6b7280",
}

PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"


# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #0f172a;
}
[data-testid="block-container"] {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0d1b2a;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* ── Title ── */
h1 {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
}
h2, h3 {
    color: #cbd5e1 !important;
}
[data-testid="stCaptionContainer"] p {
    color: #64748b !important;
    font-size: 0.82rem;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2d5a8e;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700;
    color: #60a5fa !important;
}
[data-testid="stMetricLabel"] > div {
    color: #94a3b8 !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
    border-bottom: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-weight: 600;
    font-size: 0.88rem;
    border-radius: 6px 6px 0 0;
    padding: 0.6rem 1.4rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(96, 165, 250, 0.1) !important;
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.85;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Selectbox / Slider labels ── */
label[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.85rem;
}

/* ── Divider ── */
hr {
    border-color: #1e3a5f !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando logs BGL...")
def load_data() -> pd.DataFrame:
    df = load_bgl_logs(DATA_PATH, nrows=N_ROWS_DEMO)
    return add_severity_score(df)


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model() -> AnomalyDetector | None:
    if MODEL_PATH.exists():
        return AnomalyDetector.load(MODEL_PATH)
    return None


@st.cache_resource(show_spinner="Conectando a Ollama...")
def load_summarizer() -> LogSummarizer:
    return LogSummarizer(model="llama3.2")


@st.cache_data(show_spinner="Generando predicciones...")
def get_predictions(_df: pd.DataFrame, _model: AnomalyDetector) -> pd.DataFrame:
    if FEATURES_PATH.exists():
        feat_df = load_features(FEATURES_PATH)
    else:
        _, feat_df = build_features(_df, fit_scaler=True)

    feature_cols = [c for c in feat_df.columns if c not in {"timestamp", "node", "is_anomaly"}]
    _, test_df = train_test_split_temporal(feat_df, test_fraction=0.2)

    X = test_df[feature_cols].fillna(0).values
    scores_raw = _model.score_samples(X)
    preds = _model.predict(X)

    s_min, s_max = scores_raw.min(), scores_raw.max()
    scores_norm = (scores_raw - s_min) / (s_max - s_min + 1e-9)

    result = test_df[["timestamp", "node", "is_anomaly"]].copy()
    result["anomaly_score"] = scores_norm
    result["is_predicted_anomaly"] = preds.astype(bool)
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
st.sidebar.markdown("---")

severity_filter = st.sidebar.multiselect(
    "Severidad",
    options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    default=["MEDIUM", "HIGH", "CRITICAL"],
)

score_threshold = st.sidebar.slider(
    "Score mínimo de anomalía",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#475569'>Modelo: LOF · Dataset: BGL 4.7M<br>LLM: Ollama/llama3.2</small>",
    unsafe_allow_html=True,
)

demo_mode = not DATA_PATH.exists()
if demo_mode:
    st.sidebar.warning("Modo demo — dataset no encontrado")


def _show_demo_placeholder() -> None:
    st.markdown("### Demo — estructura del dashboard")
    st.markdown("""
    **Tab 1 — Resumen en tiempo real:**
    - Métricas: total eventos, anomalías detectadas, tasa, alertas CRITICAL
    - Serie temporal eventos normal vs anomalías
    - Heatmap de anomalías por nodo y hora

    **Tab 2 — Panel de Alertas:**
    - Tabla filtrable con severidad, nodo, score
    - Distribución de scores por severidad
    - Análisis LLM on-demand por anomalía

    **Tab 3 — Análisis de Modelos:**
    - Distribución de anomaly scores (normal vs anomalía real)
    - Top nodos problemáticos
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("IT Log Anomaly Detection — NOC Dashboard")
st.caption("BGL Supercomputer Logs · Local Outlier Factor + LLM Summarizer · Ollama/llama3.2")

if demo_mode:
    st.info("Ejecuta `make download-data` y luego los notebooks 01-03 para ver datos reales.")
    _show_demo_placeholder()
    st.stop()

df_raw = load_data()
model = load_model()

if model is None:
    st.error("Modelo no entrenado. Ejecuta el notebook `03_anomaly_detection.ipynb` primero.")
    st.stop()

df_pred = get_predictions(df_raw, model)


def _assign_severity(score: float) -> str:
    if score < 0.3:
        return "LOW"
    elif score < 0.5:
        return "MEDIUM"
    elif score < 0.7:
        return "HIGH"
    return "CRITICAL"


df_pred["severidad"] = df_pred["anomaly_score"].apply(_assign_severity)

anomalies = df_pred[
    df_pred["is_predicted_anomaly"] & (df_pred["anomaly_score"] >= score_threshold)
]
if severity_filter:
    anomalies = anomalies[anomalies["severidad"].isin(severity_filter)]

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Resumen en tiempo real", "Panel de Alertas", "Análisis de Modelos"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total eventos (test)", f"{len(df_pred):,}")
    with col2:
        st.metric("Anomalías detectadas", f"{df_pred['is_predicted_anomaly'].sum():,}")
    with col3:
        rate = df_pred["is_predicted_anomaly"].mean()
        st.metric("Tasa de anomalías", f"{rate:.1%}")
    with col4:
        critical = int((anomalies["severidad"] == "CRITICAL").sum())
        st.metric(
            "Alertas CRITICAL",
            f"{critical:,}",
            delta=f"+{critical}" if critical > 0 else None,
            delta_color="inverse",
        )

    st.markdown("---")

    df_ts = df_pred.set_index("timestamp").resample("1h")["is_predicted_anomaly"].agg(
        anomalias="sum", total="count"
    )
    df_ts["normales"] = df_ts["total"] - df_ts["anomalias"]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["normales"],
        fill="tozeroy", name="Normal",
        line=dict(color="#60a5fa", width=1.5),
        fillcolor="rgba(96,165,250,0.15)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["anomalias"],
        fill="tozeroy", name="Anomalía",
        line=dict(color="#ef4444", width=1.5),
        fillcolor="rgba(239,68,68,0.3)",
    ))
    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text="Serie temporal — eventos normales vs anomalías", font=dict(size=14)),
        xaxis_title="Fecha",
        yaxis_title="Eventos/hora",
        height=340,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    top_nodes = anomalies["node"].value_counts().head(15).index.tolist()
    if top_nodes:
        heat_df = anomalies[anomalies["node"].isin(top_nodes)].copy()
        heat_df["hour"] = heat_df["timestamp"].dt.hour
        heat_data = heat_df.groupby(["node", "hour"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="YlOrRd",
            title="Heatmap — anomalías por nodo y hora del día",
            labels={"x": "Hora", "y": "Nodo", "color": "Anomalías"},
            template=PLOTLY_TEMPLATE,
        )
        fig_heat.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Alert panel ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Alertas detectadas ({len(anomalies):,})")

    if anomalies.empty:
        st.info("No hay alertas con los filtros actuales.")
    else:
        display_df = anomalies.sort_values("anomaly_score", ascending=False).head(200).copy()
        display_df["Severidad"] = display_df["severidad"].map({
            "LOW": "🟢 LOW",
            "MEDIUM": "🟡 MEDIUM",
            "HIGH": "🟠 HIGH",
            "CRITICAL": "🔴 CRITICAL",
        })

        st.dataframe(
            display_df[["timestamp", "node", "anomaly_score", "Severidad", "is_anomaly"]]
            .rename(columns={
                "timestamp": "Timestamp",
                "node": "Nodo",
                "anomaly_score": "Score",
                "is_anomaly": "Real",
            }),
            use_container_width=True,
            height=380,
        )

        fig_scores = px.histogram(
            anomalies,
            x="anomaly_score",
            color="severidad",
            color_discrete_map=SEVERITY_COLORS,
            title="Distribución de anomaly scores por severidad",
            nbins=50,
            template=PLOTLY_TEMPLATE,
        )
        fig_scores.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=280,
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    # ── LLM on-demand ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Análisis LLM — on demand")
    st.caption("Selecciona una anomalía para que llama3.2 genere un diagnóstico en lenguaje natural.")

    top_for_llm = anomalies.nlargest(20, "anomaly_score").reset_index(drop=True)

    if top_for_llm.empty:
        st.info("Ajusta los filtros para ver anomalías disponibles para analizar.")
    else:
        options = [
            f"#{i + 1}  {row['node']}  |  {row['timestamp']}  |  score={row['anomaly_score']:.3f}"
            for i, row in top_for_llm.iterrows()
        ]
        selected_idx = st.selectbox(
            "Anomalía a analizar:",
            range(len(options)),
            format_func=lambda i: options[i],
        )

        if st.button("Analizar con LLM", type="primary"):
            row = top_for_llm.iloc[selected_idx]
            ts = row["timestamp"]
            node = str(row["node"])
            score_val = float(row["anomaly_score"])

            window = df_raw[
                (df_raw["timestamp"] >= ts - pd.Timedelta(minutes=5))
                & (df_raw["timestamp"] <= ts + pd.Timedelta(minutes=5))
            ]

            summarizer = load_summarizer()

            if not summarizer.is_available:
                st.warning("Ollama no disponible. Inicia el servidor con: `ollama serve`")
            elif window.empty:
                st.warning("No se encontraron logs en la ventana temporal ±5min para este evento.")
            else:
                with st.spinner(f"llama3.2 analizando {node} — {len(window)} eventos en ventana..."):
                    context = build_anomaly_context(
                        df_window=window,
                        anomaly_score=score_val,
                        node=node,
                        timestamp=str(ts),
                    )
                    result = summarizer.summarize(context)

                sev = result.get("severidad", "UNKNOWN")
                accent = SEVERITY_COLORS.get(sev, "#6b7280")
                resp_ms = result.get("response_time_ms", 0)

                st.markdown(
                    f"""
<div style="
  border-left: 5px solid {accent};
  padding: 1.2rem 1.5rem;
  background: #1e293b;
  border-radius: 8px;
  margin-top: 0.75rem;
  color: #e2e8f0;
  font-family: inherit;
">
  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem">
    <span style="
      background:{accent};
      color:#000;
      font-weight:700;
      font-size:0.75rem;
      padding:0.2rem 0.6rem;
      border-radius:4px;
      letter-spacing:0.06em;
    ">{sev}</span>
    <span style="color:#64748b;font-size:0.8rem">Nodo: {node}</span>
  </div>
  <p style="margin:0.5rem 0;color:#e2e8f0">
    <span style="color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em">Resumen</span><br>
    {result.get("resumen", "N/A")}
  </p>
  <p style="margin:0.5rem 0;color:#e2e8f0">
    <span style="color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em">Causa probable</span><br>
    {result.get("causa_probable", "N/A")}
  </p>
  <p style="margin:0.5rem 0;color:#e2e8f0">
    <span style="color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em">Acción recomendada</span><br>
    {result.get("accion_recomendada", "N/A")}
  </p>
  <p style="margin-top:1rem;color:#475569;font-size:0.78rem;border-top:1px solid #334155;padding-top:0.6rem">
    Tiempo LLM: {resp_ms:.0f}ms &nbsp;·&nbsp; Ventana: {len(window)} eventos &nbsp;·&nbsp; {ts}
  </p>
</div>
""",
                    unsafe_allow_html=True,
                )

# ── Tab 3: Model analysis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis del modelo — Local Outlier Factor")

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_pred[~df_pred["is_anomaly"]]["anomaly_score"],
            name="Normal (ground truth)",
            opacity=0.6,
            marker_color="#60a5fa",
            histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_pred[df_pred["is_anomaly"]]["anomaly_score"],
            name="Anomalía (ground truth)",
            opacity=0.75,
            marker_color="#ef4444",
            histnorm="probability density",
        ))
        fig_dist.update_layout(
            barmode="overlay",
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            title=dict(text="Distribución de scores — Normal vs Anomalía real", font=dict(size=13)),
            xaxis_title="Anomaly score (normalizado)",
            yaxis_title="Densidad",
            height=340,
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        node_scores = (
            df_pred[df_pred["is_predicted_anomaly"]]
            .groupby("node", observed=True)
            .agg(n_anomalies=("anomaly_score", "count"), avg_score=("anomaly_score", "mean"))
            .sort_values("n_anomalies", ascending=False)
            .head(15)
        )
        fig_nodes = px.bar(
            node_scores.reset_index(),
            x="node",
            y="n_anomalies",
            color="avg_score",
            color_continuous_scale="Reds",
            title="Top 15 nodos con más anomalías detectadas",
            template=PLOTLY_TEMPLATE,
        )
        fig_nodes.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=340,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_tickangle=45,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

    # Metrics summary
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    tp = int((df_pred["is_anomaly"] & df_pred["is_predicted_anomaly"]).sum())
    fp = int((~df_pred["is_anomaly"] & df_pred["is_predicted_anomaly"]).sum())
    fn = int((df_pred["is_anomaly"] & ~df_pred["is_predicted_anomaly"]).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    with mc1:
        st.metric("Precision", f"{precision:.4f}")
    with mc2:
        st.metric("Recall", f"{recall:.4f}")
    with mc3:
        st.metric("F1", f"{f1:.4f}")
    with mc4:
        st.metric("Verdaderos positivos", f"{tp:,}")
