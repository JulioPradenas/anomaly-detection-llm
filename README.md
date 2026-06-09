# IT Log Anomaly Detection + LLM Summarizer

[![CI](https://github.com/JulioPradenas/anomaly-detection-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/JulioPradenas/anomaly-detection-llm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Cobertura](https://img.shields.io/badge/Cobertura-86%25-blue)
![Tests](https://img.shields.io/badge/Tests-97%20passing-brightgreen)
![F1 LOF](https://img.shields.io/badge/F1%20LOF-0.947-success)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
![Estado](https://img.shields.io/badge/Estado-Producci%C3%B3n%20Listo-green)

Sistema end-to-end de detección de anomalías en logs de infraestructura IT, con un LLM local que interpreta cada alerta en lenguaje natural — simulando el flujo real de trabajo en plataformas como Splunk + NOC operations.

**Dataset:** BGL (Blue Gene/L) — 4.7M eventos reales de un supercomputador IBM  
**Stack V1:** LOF + DuckDB + LangChain + Ollama/llama3.2 + FastAPI + Streamlit  
**Stack V2:** + LightGBM (active learning) + LangGraph agent + Evidently drift + MLflow

---

## Arquitectura V2

```
BGL Logs (4.7M eventos)
        │
        ▼
  Parser (loader.py)
  DataFrame estructurado
        │
        ▼
  Feature Engineering (DuckDB)
  Ventanas temporales 1/5/15 min
  error_rate, burst_flag, node_error_ratio
        │
        ▼
  Local Outlier Factor (V1)          LLM Labels (Ollama)
  Anomaly score por evento    ──────► LLM Summarizer → severity
        │                                    │
        ▼                                    ▼
  Anomalías detectadas          Active Learning (LightGBM)
                                Entrena con weak labels del LLM
        │
        ▼
  Drift Monitor (Evidently)     MLflow Registry
  KS test por feature    ──────► Versionar LOF + LightGBM
        │                        Comparar F1 antes/después
        ▼
  LangGraph Agent (con memoria)
  Investiga anomalías, crea tickets
        │
        ▼
  FastAPI  /detect  /summarize  /retrain  /agent/chat  /model/health
        │
        ▼
  Streamlit Dashboard NOC-style
  Live Monitor (stream sintético) + Drift semáforo
```

---

## V1 vs V2

| Capacidad | V1 | V2 |
|---|---|---|
| Detección | LOF (F1=0.947) | LOF + LightGBM active learner |
| LLM | Summarizer one-shot | Summarizer + Agent con memoria |
| Conversación | No | LangGraph con historial por sesión |
| Herramientas del agente | No | query_history, get_details, compare_incidents, ticket |
| Drift monitoring | No | Evidently KS test por feature |
| MLflow tracking | No | Registro + versionado + promoción |
| Dashboard | Estático | Live Monitor streaming |
| Tests | 39 | 90 |

---

## Resultados del modelo

Evaluación sobre holdout temporal (últimas 20% del dataset, ~80K eventos):

| Modelo | F1 | Precision | Recall | AUPR |
|---|---|---|---|---|
| **Local Outlier Factor** | **0.947** | 0.922 | 0.973 | 0.938 |
| Isolation Forest | 0.457 | 0.923 | 0.304 | 0.949 |
| One-Class SVM | 0.376 | 0.872 | 0.239 | 0.930 |

**¿Por qué LOF y no Isolation Forest?**

Los fallos en BGL se propagan por nodos vecinos del mismo rack, generando *clusters locales* de eventos anómalos. LOF captura exactamente eso — densidad anormalmente baja en el vecindario local. IF asume anomalías dispersas uniformemente, lo que no se cumple aquí: IF y OC-SVM se perdían el **70% de las anomalías reales** (recall 0.24-0.30).

---

## LLM Agent — Ejemplo de conversación

El agente corre con Ollama (100% local) y mantiene historial por sesión. Ejemplo real:

```
POST /agent/chat
{"message": "¿Cuántos nodos tuvieron errores críticos en las últimas 2 horas?",
 "session_id": "noc-shift-A"}

→ "Revisé el historial: 3 nodos con anomalías en las últimas 2h. El más crítico
   es R30-M0-N9 con score 0.94 (TLB error). ¿Quieres que abra un ticket?"

POST /agent/chat
{"message": "Sí, crea el ticket para ese nodo",
 "session_id": "noc-shift-A"}

→ "Ticket INC-7821 creado: R30-M0-N9 — TLB parity error. Asignado a equipo hardware.
   Prioridad: P1. El agente recuerda el contexto de la conversación anterior."
```

Herramientas disponibles: `query_anomaly_history`, `get_anomaly_details`, `compare_incidents`, `create_mock_ticket`.

---

## LLM Summarizer — Output real

El sistema corre **100% local** con Ollama + llama3.2. Ejemplo de output sobre anomalías reales del dataset BGL:

```
ANOMALÍA detectada — Nodo: R30-M0-N9-C:J16-U01
Timestamp: 2005-06-11 23:32:33 | Anomaly score: 69,004,232

Resumen:       Se detectó una anomalía en el nodo R30-M0-N9-C:J16-U01 debido a
               errores repetidos de TLB (Tabla de Cache L1) que afectan la
               estabilidad del sistema.

Severidad:     CRITICAL
Causa probable: Error de TLB persistente causado por una falla en el componente HARDWARE
Acción:        Reiniciar el nodo R30-M0-N9-C:J16-U01 para restaurar la estabilidad
               del sistema y eliminar los errores de TLB

Tiempo de respuesta LLM: 12,000ms (primera llamada) → ~6,500ms (subsiguientes)
```

---

## Drift Detection

Evidently monitorea si la distribución de features ha cambiado respecto al entrenamiento (KS test por feature). El endpoint `/model/health` expone el drift score en tiempo real:

```bash
curl http://localhost:8000/model/health
# {
#   "model_version": "lof_v1",
#   "drift_score": 0.12,
#   "drift_detected": false,
#   "features_drifted": [],
#   "recommendation": "No se detectó drift — el modelo puede continuar en producción."
# }
```

El dashboard muestra un semáforo: verde (< 0.3), amarillo (0.3–0.7), rojo (> 0.7).

---

## MLflow

```bash
# Registrar el modelo LOF después de entrenar
python -c "
from src.monitoring.model_monitor import log_lof_model
log_lof_model('models/saved/lof_v1.joblib',
              metrics={'f1': 0.947, 'precision': 0.922, 'recall': 0.973},
              params={'n_neighbors': 20, 'contamination': 0.073})
"

# Ver UI
mlflow ui --backend-store-uri ./mlflow  # http://localhost:5000

# Re-entrenar con active learning (actualiza el registro automáticamente)
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"use_llm_labels": false, "min_samples": 100}'
```

---

## Insights del dataset BGL

- **4,747,963 eventos** — 348,460 anomalías reales (7.3% del total)
- Las anomalías se concentran en la **segunda mitad del dataset** (distribución temporal no uniforme)
- Un solo nodo (`R30-M0-N9-C:J16-U01`) concentra las anomalías de mayor score — fallos en cascada típicos de rack
- Los errores de **TLB y memoria** son el patrón dominante — hardware failure propagation
- `error_rate_5min` y `fatal_count_5min` son las features más discriminativas
- AUPR similar entre modelos (~0.93-0.95): las features son buenas, la diferencia está en la calibración del threshold

---

## Setup

```bash
# 1. Instalar dependencias
uv sync --extra dev

# 2. Dataset BGL (~55MB zip, 709MB descomprimido)
curl -L "https://zenodo.org/records/8196385/files/BGL.zip?download=1" \
     -o data/raw/BGL.zip && unzip data/raw/BGL.zip -d data/raw/

# 3. Entrenar modelo (ejecutar notebooks en orden)
uv run jupyter lab
# → 01_eda_logs.ipynb
# → 02_feature_engineering.ipynb
# → 03_anomaly_detection.ipynb  (~4 min)
# → 04_evaluation.ipynb
# → 05_llm_summarizer.ipynb

# 4. LLM local (opcional — sin esto el sistema sigue funcionando)
brew install ollama
ollama pull llama3.2
ollama serve

# 5. API
make api        # http://localhost:8000/docs

# 6. Dashboard
make dashboard  # http://localhost:8501
```

---

## API

```bash
# Estado del sistema
curl http://localhost:8000/health

# Detectar anomalías
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"logs": [{"timestamp": "2005-06-11T23:32:33", "node": "R30-M0-N9",
       "level": "FATAL", "component": "HARDWARE",
       "content": "tlb error interrupt"}]}'

# Drift y salud del modelo
curl http://localhost:8000/model/health

# Re-entrenar con active learning
curl -X POST http://localhost:8000/retrain \
  -d '{"use_llm_labels": false, "min_samples": 100}'

# Chat con el agente
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué nodo tiene más anomalías?", "session_id": "demo"}'

# Limpiar historial de sesión
curl -X DELETE http://localhost:8000/agent/session/demo
```

Swagger UI: http://localhost:8000/docs

---

## Tests

```bash
make test       # 90 tests, 81% coverage
make test-fast  # sin coverage, falla rápido
```

Los tests del LLM summarizer y el agente mockean Ollama — no requieren LLM activo en CI.

---

## Docker

```bash
# Solo API
docker build -t anomaly-detection .
docker run -p 8000:8000 anomaly-detection

# API + Ollama
docker compose up
```

---

## ¿Por qué Ollama y no OpenAI?

En infraestructura crítica (sistemas tributarios, plataformas de autenticación), los logs contienen información sensible. Un LLM local elimina el riesgo de enviar datos a servicios externos, cumpliendo con requerimientos de privacidad y soberanía de datos. Si Ollama no está disponible, el sistema de detección continúa sin interrupciones (fallback graceful).

---

## Referencias

- [Loghub — BGL Dataset](https://github.com/logpai/loghub)
- [DeepLog (2017)](https://dl.acm.org/doi/10.1145/3133956.3134015) — referencia académica
- [LOF — Breunig et al. (2000)](https://dl.acm.org/doi/10.1145/335191.335388)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Evidently AI — Data Drift](https://docs.evidentlyai.com/)
- [LangGraph — Agent with Memory](https://langchain-ai.github.io/langgraph/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
