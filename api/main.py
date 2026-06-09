"""FastAPI app for IT log anomaly detection."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predictor import AnomalyPipeline
from api.schemas import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
)


pipeline = AnomalyPipeline()


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
