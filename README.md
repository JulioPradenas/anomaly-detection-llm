# IT Log Anomaly Detection + LLM Summarizer

Sistema end-to-end de detección de anomalías en logs de infraestructura IT, con un LLM local que interpreta cada alerta en lenguaje natural — simulando el flujo real de trabajo en plataformas como Splunk + NOC operations.

**Dataset:** BGL (Blue Gene/L) — 4.7M eventos reales de un supercomputador IBM  
**Stack:** LOF + DuckDB + LangChain + Ollama/llama3.2 + FastAPI + Streamlit

---

## Arquitectura

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
  Local Outlier Factor
  Anomaly score por evento
        │
        ▼
  Anomalías detectadas ──────► LLM Summarizer (Ollama/llama3.2)
                                Resumen + Severidad + Acción
        │
        ▼
  FastAPI  /detect  /summarize
        │
        ▼
  Streamlit Dashboard (NOC-style)
```

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

El LLM identifica correctamente la causa técnica (TLB cache errors), asigna severidad CRITICAL y proporciona una acción concreta — exactamente lo que necesita un analista NOC para triaje inmediato.

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

# 4. LLM local (opcional)
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
# {"status":"ok","model_loaded":true,"llm_available":true,"model_type":"lof"}

# Detectar anomalías
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"logs": [{"timestamp": "2005-06-11T23:32:33", "node": "R30-M0-N9",
       "level": "FATAL", "component": "HARDWARE",
       "content": "tlb error interrupt"}]}'

# Generar resumen LLM
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"anomaly_id": "abc-123", "context_window": [...], "anomaly_score": 0.9}'
```

Swagger UI: http://localhost:8000/docs

---

## Tests

```bash
make test       # 39 tests, 91% coverage
make test-fast  # sin coverage, falla rápido
```

Los tests del LLM summarizer mockean Ollama — no requieren LLM activo en CI.

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
