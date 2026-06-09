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
    "LOW": "#4CAF50",
    "MEDIUM": "#FF9800",
    "HIGH": "#F44336",
    "CRITICAL": "#212121",
    "UNKNOWN": "#9E9E9E",
}


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

    # Normalize scores to [0,1] for consistent display across model types
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
demo_mode = not DATA_PATH.exists()
if demo_mode:
    st.sidebar.warning("Modo demo — dataset no encontrado")


def _show_demo_placeholder():
    st.markdown("### Demo — estructura del dashboard")
    st.markdown("""
    **Tab 1 — Resumen en tiempo real:**
    - Métricas: total eventos, anomalías detectadas, tasa, alertas CRITICAL
    - Serie temporal events normal vs anomalías
    - Heatmap de anomalías por nodo y hora

    **Tab 2 — Panel de Alertas:**
    - Tabla filtrable con severidad, nodo, score
    - Distribución de scores por severidad

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

# Asignar severidad heurística basada en score (el LLM no está corriendo en el dashboard)
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
        st.metric("Total eventos", f"{len(df_pred):,}")
    with col2:
        st.metric("Anomalías detectadas", f"{df_pred['is_predicted_anomaly'].sum():,}")
    with col3:
        rate = df_pred["is_predicted_anomaly"].mean()
        st.metric("Tasa de anomalías", f"{rate:.1%}")
    with col4:
        critical = (anomalies["severidad"] == "CRITICAL").sum()
        st.metric("Alertas CRITICAL", f"{critical:,}", delta=f"+{critical}" if critical > 0 else None,
                  delta_color="inverse")

    st.markdown("---")

    # Serie temporal
    df_ts = df_pred.set_index("timestamp").resample("1h")["is_predicted_anomaly"].agg(
        anomalias="sum", total="count"
    )
    df_ts["normales"] = df_ts["total"] - df_ts["anomalias"]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["normales"],
        fill="tozeroy", name="Normal",
        line=dict(color="#2196F3"), fillcolor="rgba(33,150,243,0.3)"
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, y=df_ts["anomalias"],
        fill="tozeroy", name="Anomalía",
        line=dict(color="#F44336"), fillcolor="rgba(244,67,54,0.5)"
    ))
    fig_ts.update_layout(
        title="Serie temporal: eventos normales vs anomalías",
        xaxis_title="Fecha", yaxis_title="Eventos/hora",
        height=350, margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Heatmap por nodo
    top_nodes = anomalies["node"].value_counts().head(15).index.tolist()
    if top_nodes:
        heat_df = anomalies[anomalies["node"].isin(top_nodes)].copy()
        heat_df["hour"] = heat_df["timestamp"].dt.hour
        heat_data = heat_df.groupby(["node", "hour"]).size().unstack(fill_value=0)
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="YlOrRd",
            title="Heatmap: anomalías por nodo y hora del día",
            labels={"x": "Hora", "y": "Nodo", "color": "Anomalías"},
        )
        fig_heat.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Alert panel ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Alertas detectadas ({len(anomalies):,})")

    if anomalies.empty:
        st.info("No hay alertas con los filtros actuales.")
    else:
        display_df = anomalies.sort_values("anomaly_score", ascending=False).head(200)
        display_df["severidad_icon"] = display_df["severidad"].map({
            "LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM",
            "HIGH": "🔴 HIGH", "CRITICAL": "⚫ CRITICAL",
        })

        st.dataframe(
            display_df[["timestamp", "node", "anomaly_score", "severidad_icon", "is_anomaly"]]
            .rename(columns={
                "timestamp": "Timestamp",
                "node": "Nodo",
                "anomaly_score": "Score",
                "severidad_icon": "Severidad",
                "is_anomaly": "Anomalía real",
            }),
            use_container_width=True,
            height=400,
        )

        # Score distribution
        fig_scores = px.histogram(
            anomalies, x="anomaly_score", color="severidad",
            color_discrete_map=SEVERITY_COLORS,
            title="Distribución de anomaly scores por severidad",
            nbins=50,
        )
        st.plotly_chart(fig_scores, use_container_width=True)

# ── Tab 3: Model analysis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis del modelo — Local Outlier Factor")

    col1, col2 = st.columns(2)

    with col1:
        # Score distribution: anomalías reales vs falsas
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_pred[~df_pred["is_anomaly"]]["anomaly_score"],
            name="Normal (ground truth)", opacity=0.6,
            marker_color="#2196F3", histnorm="probability density"
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_pred[df_pred["is_anomaly"]]["anomaly_score"],
            name="Anomalía (ground truth)", opacity=0.7,
            marker_color="#F44336", histnorm="probability density"
        ))
        fig_dist.update_layout(
            barmode="overlay",
            title="Distribución de scores: Normal vs Anomalía real",
            xaxis_title="Anomaly score", yaxis_title="Densidad",
            height=350,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Top nodes por score promedio
        node_scores = (
            df_pred[df_pred["is_predicted_anomaly"]]
            .groupby("node", observed=True)
            .agg(n_anomalies=("anomaly_score", "count"), avg_score=("anomaly_score", "mean"))
            .sort_values("n_anomalies", ascending=False)
            .head(15)
        )
        fig_nodes = px.bar(
            node_scores.reset_index(),
            x="node", y="n_anomalies", color="avg_score",
            color_continuous_scale="Reds",
            title="Top 15 nodos con más anomalías detectadas",
        )
        fig_nodes.update_layout(height=350, xaxis_tickangle=45)
        st.plotly_chart(fig_nodes, use_container_width=True)
