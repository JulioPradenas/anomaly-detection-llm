# 🔍 IT Log Anomaly Detection con LLM Summarizer
## Plan de Proyecto para Portafolio de Data Scientist

> **Objetivo:** Construir un sistema end-to-end de detección de anomalías en logs de sistemas IT, con un diferenciador único: un LLM local (Ollama/llama3.2) que interpreta y resume las anomalías detectadas en lenguaje natural, simulando el flujo real de trabajo en plataformas como Splunk + NOC operations.

> **Contexto de negocio simulado:** Sistema de monitoreo para infraestructura crítica de servicios digitales (trazabilidad, autenticación, plataformas tributarias) — directamente alineado con el stack real de SICPA/SII.

---

## 📁 Estructura del Proyecto

```
anomaly-detection-llm/
│
├── data/
│   ├── raw/                          # Logs originales (HDFS / BGL)
│   ├── processed/                    # Features tabulares extraídas
│   └── samples/                      # Muestras pequeñas para tests
│
├── notebooks/
│   ├── 01_eda_logs.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_llm_summarizer.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py                 # Parseo de logs raw
│   │   └── preprocessor.py          # Extracción de features de logs
│   ├── features/
│   │   └── engineering.py           # Ventanas temporales, conteos, ratios
│   ├── models/
│   │   ├── detector.py              # Isolation Forest + LOF wrapper
│   │   └── evaluator.py             # Métricas: precision, recall, F1, AUPR
│   └── llm/
│       ├── summarizer.py            # LangChain + Ollama integration
│       └── prompts.py               # Prompt templates para resúmenes
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── schemas.py                   # Pydantic v2 models
│   └── predictor.py                 # Inference pipeline
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── models/
│   └── saved/                       # Modelos serializados (.joblib)
│
├── reports/
│   └── figures/                     # Gráficas exportadas
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_detector.py
│   ├── test_summarizer.py
│   └── test_api.py
│
├── pyproject.toml
├── Makefile
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```

---

## 📊 Dataset

**Dataset principal:** [BGL (Blue Gene/L) Supercomputer Logs — Loghub](https://github.com/logpai/loghub)

| Campo | Descripción |
|---|---|
| `timestamp` | Marca de tiempo del evento |
| `node` | Nodo del sistema que generó el log |
| `type` | Tipo de mensaje (INFO, WARNING, ERROR, FATAL) |
| `component` | Componente del sistema |
| `content` | Texto del mensaje de log |
| `label` | **Target:** `-` = normal, cualquier otro = anomalía |

**¿Por qué BGL?**
- Tiene labels reales (anomalías marcadas manualmente) → permite evaluación supervisada honesta
- ~4.7M líneas — suficientemente grande para simular operaciones reales
- Usado en papers de referencia (DeepLog, LogBERT) → tu trabajo es comparable
- Representa exactamente el tipo de datos que maneja Splunk/Nagios en producción

**Alternativa ligera para desarrollo:** usar los primeros 500K eventos para iteración rápida, modelo final sobre el dataset completo.

---

## 🛠️ Stack Tecnológico

| Capa | Herramienta | Versión |
|---|---|---|
| Runtime | Python | 3.11 |
| Package mgmt | uv | latest |
| Feature engineering | pandas, DuckDB | 2.2+, 0.10+ |
| Detección de anomalías | scikit-learn (Isolation Forest, LOF) | 1.5+ |
| Detección complementaria | PyOD | 2.0+ |
| LLM local | Ollama + llama3.2 | latest |
| LLM framework | LangChain | 0.3+ |
| API | FastAPI + Pydantic v2 | 0.115+, 2.7+ |
| Dashboard | Streamlit | 1.40+ |
| Logging estructurado | structlog | 24+ |
| Testing | pytest + pytest-cov | 8+, 5+ |
| Linting/formato | Ruff | 0.8+ |
| Type checking | mypy | 1.11+ |
| CI/CD | GitHub Actions | — |
| Containerización | Docker | — |
| Visualización | matplotlib, plotly | — |

---

## 🎯 Métricas de Éxito

| Métrica | Umbral mínimo | Objetivo |
|---|---|---|
| Precision (anomalías) | > 0.70 | > 0.80 |
| Recall (anomalías) | > 0.65 | > 0.75 |
| F1-score | > 0.68 | > 0.78 |
| AUPR (Area Under PR Curve) | > 0.75 | > 0.85 |
| Cobertura de tests | > 80% | > 85% |
| Tiempo respuesta API (p95) | < 500ms | < 200ms |
| Calidad resumen LLM | coherente | accionable |

> **Nota:** En anomaly detection no supervisado puro, precision > recall es la prioridad operacional — false positives fatigan al equipo NOC. El LLM summarizer agrega valor exactamente aquí: reduce el costo cognitivo de investigar cada alerta.

---

## 📋 Fases del Proyecto

### Fase 1 — Setup y EDA de Logs
**Objetivo:** Entender la estructura de los logs, distribución temporal de anomalías, y definir la estrategia de features.

**Tareas:**
- Configurar proyecto con `uv`, estructura de carpetas, `pyproject.toml`, `Makefile` (targets: `fix`, `test`, `run`)
- Descargar y cargar dataset BGL (primeros 500K eventos para desarrollo)
- Parsear logs raw → DataFrame estructurado con `loader.py`
- EDA: distribución de tipos de mensaje, frecuencia por nodo, ratio anomalías/normal, distribución temporal
- Identificar patrones visuales: ¿las anomalías se agrupan temporalmente? ¿por nodo?

**Outputs:**
- `notebooks/01_eda_logs.ipynb` con visualizaciones y conclusiones
- `src/data/loader.py` con función `load_bgl_logs(path) -> pd.DataFrame`
- Reporte de class imbalance (esperado: ~2-5% anomalías)

**Criterio de aceptación:** DataFrame limpio con columnas tipadas, notebook con al menos 6 visualizaciones, ratio anomalía/normal documentado.

**Commit:** `feat: phase 1 - EDA and log parsing complete`

---

### Fase 2 — Feature Engineering sobre Logs
**Objetivo:** Transformar logs de texto en features numéricas que capturen comportamiento anómalo.

**Tareas:**
- Implementar ventanas temporales deslizantes (1min, 5min, 15min) con DuckDB:
  - `error_count_1min`, `warning_count_1min`, `fatal_count_1min`
  - `error_rate_5min` (errores / total eventos)
  - `unique_nodes_15min` (diversidad de nodos activos)
  - `burst_flag` (spike de eventos > 2 std sobre media móvil)
- Features por nodo: `node_error_ratio`, `node_avg_severity`
- Encoding de `component` con frecuencia (top-20 componentes)
- Normalización con `RobustScaler` (robusto a outliers — apropiado aquí)

**Outputs:**
- `src/features/engineering.py` con `build_features(df) -> pd.DataFrame`
- `notebooks/02_feature_engineering.ipynb` con análisis de importancia de features
- `data/processed/features_train.parquet`

**Criterio de aceptación:** Pipeline reproducible, sin data leakage temporal (ventanas calculadas solo con datos pasados), shapes documentados.

**Commit:** `feat: phase 2 - temporal window features with DuckDB`

---

### Fase 3 — Detección de Anomalías
**Objetivo:** Entrenar y comparar modelos de detección. Justificar selección final.

**Modelos a comparar:**

| Modelo | Tipo | Razón de inclusión |
|---|---|---|
| Isolation Forest | Ensemble no supervisado | Estándar industria, interpretable, rápido |
| Local Outlier Factor | Densidad | Captura anomalías locales (por nodo) |
| One-Class SVM | Frontera de decisión | Baseline clásico |

**Tareas:**
- Implementar `AnomalyDetector` con interfaz unificada (`fit`, `predict`, `score`)
- Tuning de `contamination` parameter con Optuna (usando labels como validación)
- Comparativa en tabla: Precision, Recall, F1, AUPR por modelo
- Análisis de falsos positivos: ¿qué tipo de eventos generan más FP?
- **Selección final:** Isolation Forest (justificación documentada)

**Outputs:**
- `src/models/detector.py`
- `src/models/evaluator.py` con métricas y curvas PR
- `notebooks/03_anomaly_detection.ipynb`
- `models/saved/isolation_forest_v1.joblib`

**Criterio de aceptación:** Modelo final con F1 > 0.68 en holdout temporal (últimas 2 semanas del dataset).

**Commit:** `feat: phase 3 - anomaly detection models with comparative evaluation`

---

### Fase 4 — LLM Summarizer (Diferenciador Central)
**Objetivo:** Construir el pipeline que toma anomalías detectadas y genera resúmenes accionables en lenguaje natural usando Ollama + llama3.2.

**Arquitectura del summarizer:**

```
Anomalía detectada
    ↓
Contexto estructurado (ventana de eventos ±5min)
    ↓
Prompt template (LangChain)
    ↓
Ollama / llama3.2 (local)
    ↓
Resumen accionable + severidad estimada + acción sugerida
```

**Prompt template base:**
```
Eres un ingeniero experto en operaciones IT analizando logs de sistema.
Se detectó una anomalía en el nodo {node} a las {timestamp}.

Contexto de los últimos 5 minutos:
- Eventos totales: {total_events}
- Errores: {error_count} ({error_rate:.1%} del total)
- Warnings: {warning_count}
- Componentes afectados: {components}
- Mensajes representativos: {sample_messages}

Anomaly score: {anomaly_score:.3f} (umbral: {threshold:.3f})

Proporciona:
1. Resumen ejecutivo (2 oraciones máximo)
2. Severidad estimada (LOW/MEDIUM/HIGH/CRITICAL)
3. Causa probable
4. Acción recomendada inmediata
Responde en español.
```

**Tareas:**
- Instalar y configurar Ollama con llama3.2 (documentar en README)
- Implementar `LogSummarizer` con LangChain `ChatOllama`
- Fallback graceful si Ollama no está disponible (modo sin LLM)
- Evaluar calidad cualitativamente: 20 anomalías → revisar coherencia de resúmenes
- Agregar `response_time_ms` como métrica de logging

**Outputs:**
- `src/llm/summarizer.py`
- `src/llm/prompts.py`
- `notebooks/05_llm_summarizer.ipynb` con ejemplos reales de output

**Criterio de aceptación:** Pipeline funciona end-to-end, fallback implementado, 3 ejemplos de resúmenes documentados en el notebook.

**Commit:** `feat: phase 4 - LLM summarizer with Ollama and LangChain`

---

### Fase 5 — FastAPI + Schemas
**Objetivo:** Exponer el sistema como API REST con dos endpoints principales.

**Endpoints:**

```
POST /detect
  Input:  { logs: List[LogEntry], window_minutes: int = 5 }
  Output: { anomalies: List[AnomalyResult], summary_available: bool }

POST /summarize
  Input:  { anomaly_id: str, context_window: List[LogEntry] }
  Output: { summary: str, severity: str, probable_cause: str,
            recommended_action: str, response_time_ms: float }

GET  /health
  Output: { status: str, model_loaded: bool, llm_available: bool }
```

**Tareas:**
- Implementar schemas Pydantic v2 (`LogEntry`, `AnomalyResult`, `SummaryResponse`)
- Cargar modelo en `lifespan` context manager (no en `startup` deprecated)
- Verificar disponibilidad de Ollama en `/health`
- Manejo de errores con códigos HTTP apropiados
- Documentación automática en `/docs` (Swagger)

**Outputs:**
- `api/main.py`, `api/schemas.py`, `api/predictor.py`

**Criterio de aceptación:** Ambos endpoints responden correctamente, `/health` reporta estado real de Ollama.

**Commit:** `feat: phase 5 - FastAPI with detect and summarize endpoints`

---

### Fase 6 — Dashboard Streamlit (NOC-style)
**Objetivo:** Dashboard operacional que simula un Network Operations Center — el tipo de herramienta que usaría SICPA para monitorear sistemas del SII.

**Secciones del dashboard:**

**1. 📊 Overview en tiempo real**
- Métricas del período: total eventos, anomalías detectadas, tasa de anomalías
- Gráfico de serie temporal: eventos normales vs anomalías (colores distintos)
- Heatmap por nodo y hora del día

**2. 🚨 Panel de Alertas**
- Tabla de anomalías detectadas con columnas: timestamp, nodo, score, severidad (del LLM)
- Click en una fila → expande el resumen del LLM
- Filtros: por severidad, por nodo, por rango de tiempo

**3. 🔬 Análisis de Modelos**
- Curva PR comparativa (IF vs LOF vs OC-SVM)
- Distribución de anomaly scores
- Top features por importancia (permutation importance)

**Tareas:**
- Implementar con `st.session_state` para simulación de streaming
- Colores de severidad: 🟢 LOW, 🟡 MEDIUM, 🔴 HIGH, ⚫ CRITICAL
- Modo demo que carga datos pre-procesados (no requiere Ollama activo)

**Outputs:**
- `dashboard/app.py`
- Deployed en Streamlit Cloud

**Criterio de aceptación:** Dashboard carga en < 3s, modo demo funcional sin dependencias externas, alertas expandibles con resumen LLM.

**Commit:** `feat: phase 6 - NOC-style Streamlit dashboard`

---

### Fase 7 — Testing, CI/CD y Cleanup
**Objetivo:** Calidad de producción: tests, CI verde, Docker, README.

**Tareas:**
- Tests unitarios: `test_preprocessing.py`, `test_detector.py`, `test_summarizer.py`
- Tests de integración: `test_api.py` con `TestClient` como context manager
- Mock de Ollama en tests (no requiere LLM activo en CI)
- GitHub Actions: lint (Ruff) → type check (mypy) → tests → coverage badge
- Dockerfile multi-stage: builder + runtime
- `docker-compose.yml` con servicio API + Ollama
- README con: badges, arquitectura, ejemplos de output LLM, instrucciones de setup

**Outputs:**
- CI verde en `main`
- Cobertura > 80%
- README con GIF del dashboard y ejemplos reales de resúmenes LLM

**Criterio de aceptación:** `make fix && make test` pasan localmente y en CI. Docker build exitoso.

**Commit:** `feat: phase 7 - tests, CI/CD, Docker, README complete`

---

## 🗓️ Estimación de Tiempo (1-2 semanas)

| Fase | Descripción | Días estimados |
|---|---|---|
| 1 | Setup + EDA | 1.5 |
| 2 | Feature Engineering | 1.5 |
| 3 | Modelos de detección | 2.0 |
| 4 | LLM Summarizer ⭐ | 2.0 |
| 5 | FastAPI | 1.0 |
| 6 | Dashboard | 1.5 |
| 7 | Tests + CI/CD | 1.5 |
| **Total** | | **11 días** |

> Con dedicación de ~3-4h/día, completable en 2 semanas cómodamente.

---

## 💼 Narrativa para Entrevista SICPA

**Pregunta:** *"¿Tienes experiencia con detección de anomalías en sistemas IT?"*

**Respuesta guiada por el proyecto:**
> "Construí un sistema end-to-end de detección de anomalías sobre logs reales del dataset BGL (Blue Gene/L), que contiene 4.7 millones de eventos de un supercomputador. Comparé Isolation Forest, LOF y One-Class SVM, logrando un F1 de X. El diferenciador fue agregar un LLM local con Ollama y LangChain que interpreta cada anomalía detectada y genera un resumen accionable — severidad, causa probable y acción recomendada — en lenguaje natural. Esto simula directamente cómo un analista NOC usaría una herramienta sobre datos de Splunk: no solo ver qué es anómalo, sino entender por qué y qué hacer."

**Pregunta:** *"¿Por qué Ollama y no OpenAI?"*
> "En contextos de infraestructura crítica como los sistemas del SII, los logs contienen información sensible. Un LLM local elimina el riesgo de enviar datos a servicios externos, cumpliendo con requerimientos de privacidad y soberanía de datos. Ollama corre el modelo completamente en la máquina local."

---

## 🔗 Referencias

- [Loghub — BGL Dataset](https://github.com/logpai/loghub)
- [DeepLog paper (2017)](https://dl.acm.org/doi/10.1145/3133956.3134015) — referencia académica para contextualizar
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [LangChain + Ollama](https://python.langchain.com/docs/integrations/llms/ollama)
- [Isolation Forest — sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

---

## 📌 Notas para Claude Code

- Comenzar siempre por **Fase 1** antes de tocar modelos
- El parseo de BGL requiere manejo cuidadoso: las líneas tienen formato fijo pero irregular — documentar el parser
- El `LogSummarizer` debe tener **fallback graceful**: si Ollama no está disponible, retornar `{"summary": "LLM not available", "severity": "UNKNOWN"}` sin levantar excepción
- Tests de `summarizer.py` deben mockear Ollama con `unittest.mock.patch` — nunca requerir LLM activo en CI
- El `lifespan` context manager en FastAPI es el patrón correcto (no `@app.on_event("startup")` que está deprecated en FastAPI moderno)
- Notebooks: cerrar en VS Code antes de editar desde terminal
- `make fix` antes de cada commit (Ruff + mypy + pytest)
- Commits en inglés, conventional commits format
- Markdown de notebooks en español, código en inglés, labels de plots en español
- El `docker-compose.yml` debe incluir el servicio de Ollama como dependencia del servicio API
