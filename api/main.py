"""FastAPI app for IT log anomaly detection."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.predictor import AnomalyPipeline
from api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ModelHealthResponse,
    RetrainRequest,
    RetrainResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from src.features.engineering import load_features
from src.llm.agent import LogAgent
from src.models.active_learner import ActiveLearner
from src.models.evaluator import evaluate


pipeline = AnomalyPipeline()
agent = LogAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_model()
    yield


app = FastAPI(
    title="IT Log Anomaly Detection API",
    description="Detect anomalies in IT logs using Isolation Forest + LLM summarizer",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.is_loaded,
        llm_available=pipeline.summarizer.is_available,
        model_type="lof",
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — train it first.")

    anomalies = pipeline.detect(request)
    total = len(request.logs)
    n_anomalies = len(anomalies)

    return DetectResponse(
        anomalies=anomalies,
        total_logs=total,
        total_anomalies=n_anomalies,
        anomaly_rate=n_anomalies / total if total > 0 else 0.0,
        summary_available=pipeline.summarizer.is_available,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    from src.data.preprocessor import add_severity_score
    import pandas as pd

    if not request.context_window:
        raise HTTPException(status_code=400, detail="context_window cannot be empty.")

    rows = [
        {
            "timestamp": e.timestamp,
            "node": e.node,
            "level": e.level,
            "component": e.component,
            "content": e.content,
        }
        for e in request.context_window
    ]
    df_window = pd.DataFrame(rows)
    df_window = add_severity_score(df_window)

    node = request.context_window[0].node
    timestamp = str(request.context_window[0].timestamp)

    from src.llm.summarizer import build_anomaly_context
    context = build_anomaly_context(
        df_window=df_window,
        anomaly_score=request.anomaly_score,
        node=node,
        timestamp=timestamp,
    )

    result = pipeline.summarizer.summarize(context)

    return SummarizeResponse(
        anomaly_id=request.anomaly_id,
        resumen=result.get("resumen", ""),
        severidad=result.get("severidad", "UNKNOWN"),
        causa_probable=result.get("causa_probable", ""),
        accion_recomendada=result.get("accion_recomendada", ""),
        response_time_ms=result.get("response_time_ms", 0.0),
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest):
    """Retrain using LightGBM on LLM-generated labels (active learning)."""
    import time

    import numpy as np

    from src.data.preprocessor import train_test_split_temporal

    FEATURES_PATH = Path("data/processed/features_train.parquet")
    LABELS_PATH = Path("data/labels/llm_confirmed.parquet")
    AL_MODEL_PATH = Path("models/saved/active_learner_v1.joblib")

    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=503, detail="Features not found — run notebook 02 first.")

    feat_df = load_features(FEATURES_PATH)
    feature_cols = [c for c in feat_df.columns if c not in {"timestamp", "node", "is_anomaly"}]
    train_df, test_df = train_test_split_temporal(feat_df, test_fraction=0.2)

    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["is_anomaly"].values

    # Baseline: current LOF model
    if pipeline.is_loaded:
        y_pred_lof = pipeline.model.predict(X_test)
        scores_lof = pipeline.model.score_samples(X_test)
        baseline = evaluate(y_test, y_pred_lof, scores_lof, model_name="LOF baseline")
        f1_before = baseline.f1
    else:
        f1_before = 0.0

    # Labels source
    if request.use_llm_labels and LABELS_PATH.exists():
        labels_df = __import__("pandas").read_parquet(LABELS_PATH)
        merged = train_df.merge(labels_df[["timestamp", "node", "llm_label"]], on=["timestamp", "node"], how="inner")
        merged = merged.dropna(subset=["llm_label"])
        labels_source = f"llm ({len(merged)} samples)"
    else:
        # Fallback: use ground-truth is_anomaly from features
        merged = train_df.copy()
        merged["llm_label"] = merged["is_anomaly"].astype(int)
        labels_source = f"ground_truth ({len(merged)} samples)"

    if len(merged) < request.min_samples:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(merged)} labeled samples — need at least {request.min_samples}.",
        )

    X_train_al = merged[feature_cols].fillna(0).values
    y_train_al = merged["llm_label"].values.astype(int)

    t0 = time.monotonic()
    learner = ActiveLearner()
    learner.fit(X_train_al, y_train_al, feature_names=feature_cols)
    elapsed_ms = (time.monotonic() - t0) * 1000

    y_pred_al = learner.predict(X_test)
    scores_al = learner.score_samples(X_test)
    al_result = evaluate(y_test, y_pred_al, scores_al, model_name="ActiveLearner")

    AL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    learner.save(AL_MODEL_PATH)

    return RetrainResponse(
        model_version="active_learner_v1",
        f1_before=round(f1_before, 4),
        f1_after=round(al_result.f1, 4),
        n_samples_used=len(merged),
        retrain_time_ms=round(elapsed_ms, 1),
        labels_source=labels_source,
    )


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """Conversational agent with memory — investigates anomalies using LLM + tools."""
    result = agent.chat(
        message=request.message,
        session_id=request.session_id,
        anomaly_context=request.anomaly_context,
    )
    return AgentChatResponse(
        response=result["response"],
        tools_used=result["tools_used"],
        session_id=result["session_id"],
    )


@app.delete("/agent/session/{session_id}")
async def delete_agent_session(session_id: str):
    """Clear conversation history for a session."""
    existed = agent.clear_session(session_id)
    return {"session_id": session_id, "cleared": existed}


@app.get("/model/health", response_model=ModelHealthResponse)
async def model_health():
    """Report model version, drift score, and whether re-training is recommended."""
    import pandas as pd

    from src.monitoring.drift_detector import DriftDetector

    FEATURES_PATH = Path("data/processed/features_train.parquet")
    LOF_PATH = Path("models/saved/lof_v1.joblib")

    if not FEATURES_PATH.exists():
        return ModelHealthResponse(
            model_version="lof_v1",
            last_trained="unknown",
            drift_score=0.0,
            drift_detected=False,
            features_drifted=[],
            recommendation="Features not found — run notebook 02 first.",
        )

    feat_df = load_features(FEATURES_PATH)
    feature_cols = [c for c in feat_df.columns if c not in {"timestamp", "node", "is_anomaly"}]
    n = len(feat_df)
    split = int(n * 0.8)
    reference_df = feat_df.iloc[:split][feature_cols].fillna(0)
    current_df = feat_df.iloc[split:][feature_cols].fillna(0)

    detector = DriftDetector()
    detector.fit_reference(reference_df)
    report = detector.detect(current_df)

    last_trained = "unknown"
    if LOF_PATH.exists():
        import datetime

        mtime = LOF_PATH.stat().st_mtime
        last_trained = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    return ModelHealthResponse(
        model_version="lof_v1",
        last_trained=last_trained,
        drift_score=report.drift_score,
        drift_detected=report.drift_detected,
        features_drifted=report.features_drifted,
        recommendation=report.recommendation,
    )
