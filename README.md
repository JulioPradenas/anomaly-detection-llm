# IT Log Anomaly Detection con LLM Summarizer

Sistema end-to-end de detección de anomalías en logs de sistemas IT, con un LLM local (Ollama/llama3.2) que interpreta y resume las anomalías detectadas en lenguaje natural.

**Dataset:** BGL (Blue Gene/L) Supercomputer Logs — 4.7M eventos reales  
**Stack:** Isolation Forest + LangChain + Ollama + FastAPI + Streamlit  

---

## Arquitectura

```
BGL Logs (4.7M eventos)
    ↓
Parser (src/data/loader.py)
    ↓
Feature Engineering — ventanas temporales DuckDB (1/5/15min)
    ↓
Isolation Forest (scikit-learn)
    ↓
Anomalías detectadas
    ↓
LLM Summarizer (Ollama + llama3.2) ← resumen accionable en español
    ↓
FastAPI /detect + /summarize
    ↓
Streamlit Dashboard (NOC-style)
```

---

## Setup rápido

```bash
# 1. Instalar dependencias
uv sync --extra dev

# 2. Descargar dataset BGL (~55MB comprimido, 709MB descomprimido)
# Colocar BGL.log en data/raw/BGL.log
# Fuente: https://zenodo.org/records/8196385/files/BGL.zip

# 3. (Opcional) Instalar Ollama + modelo
brew install ollama
ollama pull llama3.2
ollama serve  # en otra terminal

# 4. Ejecutar notebooks en orden
# notebooks/01_eda_logs.ipynb
# notebooks/02_feature_engineering.ipynb
# notebooks/03_anomaly_detection.ipynb

# 5. Iniciar API
make api

# 6. Iniciar dashboard
make dashboard
```

---

## Métricas del modelo (Isolation Forest)

| Métrica | Valor | Umbral |
|---|---|---|
| Precision | TBD* | > 0.70 |
| Recall | TBD* | > 0.65 |
| F1-score | TBD* | > 0.68 |
| AUPR | TBD* | > 0.75 |

*Ejecutar `notebooks/03_anomaly_detection.ipynb` para obtener métricas reales.

---

## API

```bash
# Verificar estado
curl http://localhost:8000/health

# Detectar anomalías
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"logs": [{"timestamp": "2005-06-04T00:24:32", "node": "R23-M1-N8", "level": "FATAL", "component": "APP", "content": "ciod: failed to read"}]}'

# Generar resumen LLM
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"anomaly_id": "test-1", "context_window": [...], "anomaly_score": 0.85}'
```

Documentación interactiva: http://localhost:8000/docs

---

## Tests

```bash
make test         # con coverage
make test-fast    # sin coverage, falla rápido
```

Los tests de `summarizer.py` mockean Ollama — no requieren LLM activo en CI.

---

## Docker

```bash
# Solo la API (sin Ollama)
docker build -t anomaly-detection .
docker run -p 8000:8000 anomaly-detection

# API + Ollama juntos
docker compose up
```

---

## ¿Por qué Ollama y no OpenAI?

En contextos de infraestructura crítica (sistemas tributarios, plataformas de autenticación), los logs contienen información sensible. Un LLM local elimina el riesgo de enviar datos a servicios externos, cumpliendo con requerimientos de privacidad y soberanía de datos.

---

## Referencias

- [Loghub — BGL Dataset](https://github.com/logpai/loghub)
- [DeepLog (2017)](https://dl.acm.org/doi/10.1145/3133956.3134015) — referencia académica
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Isolation Forest — sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
