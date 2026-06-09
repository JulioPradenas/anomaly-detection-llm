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
    "LOW": "#06d6a0",
    "MEDIUM": "#f4a261",
    "HIGH": "#e76f51",
    "CRITICAL": "#ef233c",
    "UNKNOWN": "#adb5bd",
}

# KPI card gradients (matching image: blue / teal / purple / red)
KPI_GRADIENTS = [
    ("135deg, #4361ee 0%, #3a0ca3 100%", "#fff"),   # Total events — blue
    ("135deg, #2ec4b6 0%, #0096a0 100%", "#fff"),   # Anomalías — teal
    ("135deg, #7c3aed 0%, #4c1d95 100%", "#fff"),   # Tasa — purple
    ("135deg, #ef233c 0%, #b5001e 100%", "#fff"),   # Critical — red
]

PLOTLY_TEMPLATE = "simple_white"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#e8ecf4"
FONT_COLOR = "#2d3748"
HOVER_LABEL = dict(bgcolor="#ffffff", font_size=12, font_color="#2d3748", bordercolor="#e2e8f0")


# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
    background-color: #f0f3fa !important;
    font-family: 'Inter', sans-serif;
    color: #2d3748;
}
[data-testid="block-container"] {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #1e2d40 !important;
    border-right: none;
    box-shadow: 3px 0 15px rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stMarkdown hr {
    border-color: #2d4a6a !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarTitle"] {
    color: #e2e8f0 !important;
    font-weight: 700;
    font-size: 1.1rem;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    background-color: #253b52 !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] * {
    color: #e2e8f0 !important;
}

/* ── Title area ── */
h1 {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #1e2d40 !important;
    -webkit-text-fill-color: #1e2d40 !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    color: #2d3748 !important;
    font-weight: 700 !important;
}
[data-testid="stCaptionContainer"] p {
    color: #718096 !important;
    font-size: 0.82rem;
}

/* ── Metric cards (native st.metric) ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(30,45,64,0.08) !important;
    border: 1px solid #e8ecf4 !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricValue"] {
    color: #1e2d40 !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
}
[data-testid="stMetricLabel"] > div {
    color: #718096 !important;
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 2px;
    border-bottom: 2px solid #e8ecf4;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #718096 !important;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.65rem 1.4rem;
    border: none !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #4361ee !important;
    border-bottom: 2px solid #4361ee !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%) !important;
    border: none !important;
    color: white !important;
    padding: 0.5rem 1.8rem !important;
    box-shadow: 0 4px 14px rgba(67,97,238,0.35) !important;
    transition: box-shadow 0.2s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(67,97,238,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(30,45,64,0.06);
    border: 1px solid #e8ecf4 !important;
}

/* ── Chart containers ── */
[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(30,45,64,0.07);
    border: 1px solid #e8ecf4;
    padding: 0.5rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #4361ee !important;
}

/* ── Plotly chart text — SVG uses fill, not color ── */
[data-testid="stPlotlyChart"] text,
[data-testid="stPlotlyChart"] .gtitle,
[data-testid="stPlotlyChart"] .xtitle,
[data-testid="stPlotlyChart"] .ytitle,
[data-testid="stPlotlyChart"] .xtick text,
[data-testid="stPlotlyChart"] .ytick text,
[data-testid="stPlotlyChart"] .legendtext,
[data-testid="stPlotlyChart"] .g-xtitle text,
[data-testid="stPlotlyChart"] .g-ytitle text {
    fill: #2d3748 !important;
    color: #2d3748 !important;
}

/* ── Selectbox main content — dropdown list ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #2d3748 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #2d3748 !important;
}
[data-baseweb="popover"] {
    background: #ffffff !important;
}
[data-baseweb="popover"] ul {
    background: #ffffff !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="list-item"] {
    background: #ffffff !important;
    color: #2d3748 !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background: #f0f3fa !important;
}
[data-baseweb="option"] {
    background: #ffffff !important;
    color: #2d3748 !important;
}

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #4361ee !important;
    color: #fff !important;
}

/* ── Divider ── */
hr {
    border-color: #e8ecf4 !important;
    margin: 1rem 0 !important;
}

/* ── Subheader spacing ── */
[data-testid="stHeadingWithActionElements"] {
    margin-top: 0.5rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _kpi_card(label: str, value: str, subtitle: str, gradient: str, text_color: str) -> str:
    return f"""
<div style="
  background: linear-gradient({gradient});
  border-radius: 14px;
  padding: 1.3rem 1.5rem;
  color: {text_color};
  box-shadow: 0 4px 18px rgba(0,0,0,0.12);
  height: 100%;
  min-height: 110px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
">
  <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;opacity:0.85">{label}</div>
  <div style="font-size:2rem;font-weight:800;letter-spacing:-0.02em;margin:0.3rem 0">{value}</div>
  <div style="font-size:0.78rem;opacity:0.75">{subtitle}</div>
</div>
"""


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
st.sidebar.markdown(
    "<h2 style='color:#e2e8f0!important;font-weight:800;font-size:1.1rem;margin:0'>NOC Dashboard</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

severity_filter = st.sidebar.multiselect(
    "Severidad",
    options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    default=["MEDIUM", "HIGH", "CRITICAL"],
)

score_threshold = st.sidebar.slider(
    "Score mínimo",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """<div style='color:#94a3b8;font-size:0.78rem;line-height:1.6'>
    <b style='color:#cbd5e1'>Modelo</b><br>Local Outlier Factor<br>
    <b style='color:#cbd5e1'>Dataset</b><br>BGL 4.7M eventos<br>
    <b style='color:#cbd5e1'>LLM</b><br>Ollama / llama3.2
    </div>""",
    unsafe_allow_html=True,
)

demo_mode = not DATA_PATH.exists()
if demo_mode:
    st.sidebar.warning("Modo demo — dataset no encontrado")


def _show_demo_placeholder() -> None:
    st.markdown("### Demo — estructura del dashboard")
    st.markdown("""
    **Tab 1 — Resumen en tiempo real:** KPIs, serie temporal, heatmap por nodo
    **Tab 2 — Panel de Alertas:** tabla de alertas, histograma, análisis LLM on-demand
    **Tab 3 — Análisis de Modelos:** distribución de scores, top nodos, métricas F1
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("IT Log Anomaly Detection — NOC Dashboard")
st.caption("BGL Supercomputer Logs  ·  Local Outlier Factor  ·  LLM Summarizer (Ollama/llama3.2)")

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
    total_ev = len(df_pred)
    n_anomalies = int(df_pred["is_predicted_anomaly"].sum())
    rate = df_pred["is_predicted_anomaly"].mean()
    critical = int((anomalies["severidad"] == "CRITICAL").sum())

    kpi_data = [
        ("Total eventos (test)", f"{total_ev:,}", "Holdout temporal 20%", *KPI_GRADIENTS[0]),
        ("Anomalías detectadas", f"{n_anomalies:,}", f"de {total_ev:,} eventos", *KPI_GRADIENTS[1]),
        ("Tasa de anomalías", f"{rate:.1%}", "Local Outlier Factor", *KPI_GRADIENTS[2]),
        ("Alertas CRITICAL", f"{critical:,}", "Score ≥ 0.70", *KPI_GRADIENTS[3]),
    ]

    cols = st.columns(4)
    for col, (label, value, subtitle, gradient, text_color) in zip(cols, kpi_data):
        with col:
            st.markdown(_kpi_card(label, value, subtitle, gradient, text_color), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    df_ts = df_pred.set_index("timestamp").resample("1h")["is_predicted_anomaly"].agg(
        anomalias="sum", total="count"
    )
    df_ts["normales"] = df_ts["total"] - df_ts["anomalias"]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["normales"],
        fill="tozeroy", name="Normal",
        line=dict(color="#4361ee", width=2),
        fillcolor="rgba(67,97,238,0.08)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["anomalias"],
        fill="tozeroy", name="Anomalía",
        line=dict(color="#ef233c", width=2),
        fillcolor="rgba(239,35,60,0.12)",
    ))
    _ax = dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR))
    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text="Serie temporal — eventos normales vs anomalías", font=dict(size=13, color=FONT_COLOR)),
        xaxis=dict(title="Fecha", showgrid=True, **_ax),
        yaxis=dict(title="Eventos/hora", showgrid=True, **_ax),
        height=310,
        margin=dict(t=50, b=30, l=30, r=20),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(color=FONT_COLOR)),
        font=dict(color=FONT_COLOR),
        hoverlabel=HOVER_LABEL,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    top_nodes = anomalies["node"].value_counts().head(15).index.tolist()
    if top_nodes:
        heat_df = anomalies[anomalies["node"].isin(top_nodes)].copy()
        heat_df["hour"] = heat_df["timestamp"].dt.hour
        heat_data = heat_df.groupby(["node", "hour"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="Blues",
            title="Heatmap — anomalías por nodo y hora del día",
            labels={"x": "Hora", "y": "Nodo", "color": "Anomalías"},
        )
        fig_heat.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=360,
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(color=FONT_COLOR),
            title_font=dict(size=13, color=FONT_COLOR),
            xaxis=dict(tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            yaxis=dict(tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            coloraxis_colorbar=dict(tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            hoverlabel=HOVER_LABEL,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Alert panel ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Alertas detectadas — {len(anomalies):,}")

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
            height=360,
        )

        fig_scores = px.histogram(
            anomalies,
            x="anomaly_score",
            color="severidad",
            color_discrete_map=SEVERITY_COLORS,
            title="Distribución de anomaly scores por severidad",
            nbins=50,
        )
        fig_scores.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis=dict(title="Score", gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            yaxis=dict(title="Cantidad", gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            height=300,
            margin=dict(t=45, b=70, l=40, r=20),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(color=FONT_COLOR)),
            font=dict(color=FONT_COLOR),
            title_font=dict(size=13, color=FONT_COLOR),
            hoverlabel=HOVER_LABEL,
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
                accent = SEVERITY_COLORS.get(sev, "#adb5bd")
                resp_ms = result.get("response_time_ms", 0)

                st.markdown(
                    f"""
<div style="
  border-left: 5px solid {accent};
  padding: 1.4rem 1.6rem;
  background: #ffffff;
  border-radius: 12px;
  margin-top: 0.75rem;
  box-shadow: 0 2px 12px rgba(30,45,64,0.08);
  border: 1px solid #e8ecf4;
  border-left: 5px solid {accent};
  color: #2d3748;
">
  <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:1rem">
    <span style="
      background:{accent};
      color:#fff;
      font-weight:700;
      font-size:0.72rem;
      padding:0.25rem 0.7rem;
      border-radius:20px;
      letter-spacing:0.07em;
    ">{sev}</span>
    <span style="color:#718096;font-size:0.82rem;font-weight:500">Nodo: {node}</span>
  </div>
  <div style="display:grid;gap:0.8rem">
    <div>
      <div style="color:#a0aec0;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem">Resumen</div>
      <div style="color:#2d3748;font-size:0.92rem;line-height:1.5">{result.get("resumen","N/A")}</div>
    </div>
    <div>
      <div style="color:#a0aec0;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem">Causa probable</div>
      <div style="color:#2d3748;font-size:0.92rem;line-height:1.5">{result.get("causa_probable","N/A")}</div>
    </div>
    <div>
      <div style="color:#a0aec0;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem">Acción recomendada</div>
      <div style="color:#2d3748;font-size:0.92rem;line-height:1.5">{result.get("accion_recomendada","N/A")}</div>
    </div>
  </div>
  <div style="margin-top:1rem;padding-top:0.7rem;border-top:1px solid #e8ecf4;color:#a0aec0;font-size:0.76rem">
    Tiempo LLM: <b style="color:#718096">{resp_ms:.0f}ms</b>
    &nbsp;·&nbsp; Ventana: <b style="color:#718096">{len(window)} eventos</b>
    &nbsp;·&nbsp; {ts}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

# ── Tab 3: Model analysis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis del modelo — Local Outlier Factor")

    col1, col2 = st.columns(2)

    with col1:
        normal_scores = df_pred.loc[~df_pred["is_anomaly"], "anomaly_score"]
        anom_scores   = df_pred.loc[df_pred["is_anomaly"],  "anomaly_score"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=normal_scores,
            name=f"Normal ({len(normal_scores):,})",
            opacity=0.75,
            marker=dict(color="#4361ee", line=dict(width=0)),
            nbinsx=50,
        ))
        fig_dist.add_trace(go.Histogram(
            x=anom_scores,
            name=f"Anomalía ({len(anom_scores):,})",
            opacity=0.65,
            marker=dict(color="#ef233c", line=dict(width=0)),
            nbinsx=50,
        ))
        fig_dist.update_layout(
            barmode="overlay",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            title=dict(text="Distribución de scores — Normal vs Anomalía (escala log)", font=dict(size=13, color=FONT_COLOR)),
            xaxis=dict(title="Anomaly score (normalizado)", gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR), showgrid=True),
            yaxis=dict(title="Cantidad (log)", type="log", gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR), showgrid=True),
            height=320,
            margin=dict(t=50, b=60, l=50, r=20),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(color=FONT_COLOR)),
            font=dict(color=FONT_COLOR),
            hoverlabel=HOVER_LABEL,
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
            color_continuous_scale=["#4361ee", "#7c3aed", "#ef233c"],
            title="Top 15 nodos — anomalías detectadas",
        )
        fig_nodes.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            xaxis=dict(gridcolor=GRID_COLOR, tickangle=45, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            height=320,
            margin=dict(t=50, b=60, l=30, r=20),
            coloraxis_showscale=False,
            font=dict(color=FONT_COLOR),
            title_font=dict(size=13, color=FONT_COLOR),
            hoverlabel=HOVER_LABEL,
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

    # Metrics summary
    st.markdown("---")
    tp = int((df_pred["is_anomaly"] & df_pred["is_predicted_anomaly"]).sum())
    fp = int((~df_pred["is_anomaly"] & df_pred["is_predicted_anomaly"]).sum())
    fn = int((df_pred["is_anomaly"] & ~df_pred["is_predicted_anomaly"]).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Precision", f"{precision:.4f}")
    with mc2:
        st.metric("Recall", f"{recall:.4f}")
    with mc3:
        st.metric("F1 Score", f"{f1:.4f}")
    with mc4:
        st.metric("Verdaderos positivos", f"{tp:,}")
