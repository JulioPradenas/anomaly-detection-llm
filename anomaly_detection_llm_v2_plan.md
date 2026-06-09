# 🔍 IT Log Anomaly Detection — V2 Extension Plan
## Continuación post-V1: Active Learning + LLM Agent + Drift Detection + Streaming

> **Prerequisito:** V1 completamente terminada. CI verde en `main`, todos los tests pasando, dashboard deployado en Streamlit Cloud, API funcional con `/detect` y `/summarize`.

> **Objetivo V2:** Transformar el sistema de detección reactiva a un sistema inteligente con aprendizaje continuo, agente LLM con memoria, monitoreo de drift y dashboard en tiempo real. Cada fase es independiente y mergeable por separado.

---

## 🗂️ Cambios de Estructura respecto a V1

Solo se agregan archivos nuevos — no se modifica la estructura existente, para evitar romper V1.

```
anomaly-detection-llm/
│
├── src/
│   ├── models/
│   │   ├── detector.py              # ✅ V1 — sin cambios
│   │   ├── evaluator.py             # ✅ V1 — sin cambios
│   │   └── active_learner.py        # 🆕 V2 — LightGBM con labels del LLM
│   ├── llm/
│   │   ├── summarizer.py            # ✅ V1 — sin cambios
│   │   ├── prompts.py               # ✅ V1 — se extiende con prompts nuevos
│   │   └── agent.py                 # 🆕 V2 — LangChain agent con memoria y tools
│   ├── monitoring/
│   │   ├── drift_detector.py        # 🆕 V2 — Evidently drift detection
│   │   └── model_monitor.py         # 🆕 V2 — wrapper de métricas de monitoreo
│   └── streaming/
│       └── event_generator.py       # 🆕 V2 — generador de eventos para dashboard
│
├── api/
│   ├── main.py                      # 🔧 V2 — agregar endpoints nuevos
│   ├── schemas.py                   # 🔧 V2 — agregar schemas nuevos
│   └── predictor.py                 # ✅ V1 — sin cambios
│
├── dashboard/
│   └── app.py                       # 🔧 V2 — agregar tab de streaming y drift
│
├── notebooks/
│   ├── 06_active_learning.ipynb     # 🆕 V2
│   ├── 07_llm_agent.ipynb           # 🆕 V2
│   └── 08_drift_detection.ipynb     # 🆕 V2
│
├── data/
│   └── labels/
│       └── llm_confirmed.parquet    # 🆕 V2 — labels generados por LLM
│
├── mlflow/                          # 🆕 V2 — tracking local
│
└── tests/
    ├── test_active_learner.py        # 🆕 V2
    ├── test_agent.py                 # 🆕 V2
    └── test_drift_detector.py        # 🆕 V2
```

---

## 📦 Dependencias Nuevas V2

Agregar a `pyproject.toml`:

```toml
# V2 additions
evidently = ">=0.4.30"
mlflow = ">=2.15"
langchain-community = ">=0.3"      # para ConversationBufferMemory y tools
lightgbm = ">=4.5"                 # ya puede estar de proyectos anteriores
```

Instalar con:
```bash
uv add evidently mlflow langchain-community lightgbm
```

---

## 📋 Fases V2

---

### Fase 8 — Active Learning con LightGBM
**Objetivo:** Usar los labels de severidad generados por el LLM en V1 como supervisión débil para entrenar un clasificador LightGBM que mejore la detección iterativamente.

**Contexto:**
En V1, el LLM clasifica cada anomalía detectada con severidad LOW/MEDIUM/HIGH/CRITICAL. Esas etiquetas son señal supervisada — imperfecta, pero consistente. En esta fase las usamos para entrenar LightGBM y comparar su performance contra Isolation Forest.

**Flujo:**

```
Isolation Forest (V1) → candidatos a anomalía
        ↓
LLM Summarizer (V1) → severidad por anomalía
        ↓
Label encoding: HIGH/CRITICAL = 1, LOW/MEDIUM = 0
        ↓
LightGBM entrenado con esos labels → predictor supervisado
        ↓
Evaluación comparativa: IF vs LightGBM en holdout temporal
```

**Tareas:**
- Implementar `src/models/active_learner.py`:
  - `ActiveLearner` con método `fit(features_df, llm_labels)` → LightGBM
  - `predict_proba(features_df)` → scores de anomalía
  - `get_feature_importance()` → qué features el modelo supervisado prioriza
- Pipeline de generación de labels: correr summarizer sobre anomalías V1 → guardar en `data/labels/llm_confirmed.parquet`
- Comparativa en `notebooks/06_active_learning.ipynb`:
  - Precision/Recall/F1/AUPR: Isolation Forest vs LightGBM supervisado
  - Curvas PR superpuestas
  - Feature importance del modelo supervisado vs permutation importance de IF
- Documentar en notebook: ¿el LightGBM supervisado mejora? ¿en qué casos falla?
- Registrar experimento en MLflow: parámetros, métricas, artefactos

**MLflow tracking:**
```python
with mlflow.start_run(run_name="active_learner_v1"):
    mlflow.log_params({"n_estimators": 300, "learning_rate": 0.05, ...})
    mlflow.log_metrics({"f1": f1, "aupr": aupr, "precision": precision})
    mlflow.sklearn.log_model(model, "active_learner")
```

**Nuevo endpoint API:**
```
POST /retrain
  Input:  { use_llm_labels: bool = True, min_samples: int = 100 }
  Output: { model_version: str, f1_before: float, f1_after: float,
            n_samples_used: int, retrain_time_ms: float }
```

**Outputs:**
- `src/models/active_learner.py`
- `notebooks/06_active_learning.ipynb`
- `data/labels/llm_confirmed.parquet`
- MLflow run registrado en `mlflow/` local
- Tests en `tests/test_active_learner.py`

**Criterio de aceptación:** LightGBM entrenado con labels LLM. Comparativa documentada. `POST /retrain` funcional. MLflow run visible en `mlflow ui`.

**Commit:** `feat: phase 8 - active learning with LightGBM on LLM-generated labels`

---

### Fase 9 — LLM Agent con Memoria y Herramientas
**Objetivo:** Transformar el summarizer one-shot de V1 en un agente conversacional que recuerda incidentes anteriores, puede consultar el historial y ejecutar herramientas.

**Arquitectura del agente:**

```
Usuario / Sistema
      ↓
LangChain AgentExecutor
      ├── ConversationBufferWindowMemory (últimas N interacciones)
      ├── Tool: query_anomaly_history(node, date_range) → DuckDB
      ├── Tool: get_anomaly_details(anomaly_id) → parquet
      ├── Tool: compare_incidents(id_1, id_2) → análisis comparativo
      └── Tool: create_mock_ticket(severity, summary) → ServiceNow mock
      ↓
Ollama / llama3.2
      ↓
Respuesta con contexto histórico
```

**Herramientas (LangChain Tools):**

```python
@tool
def query_anomaly_history(node: str, days_back: int = 7) -> str:
    """Consulta anomalías históricas de un nodo en los últimos N días.
    Retorna resumen con conteo, severidades y timestamps."""

@tool
def get_anomaly_details(anomaly_id: str) -> str:
    """Retorna detalles completos de una anomalía específica:
    contexto de eventos, score, resumen LLM previo."""

@tool
def compare_incidents(anomaly_id_1: str, anomaly_id_2: str) -> str:
    """Compara dos incidentes: similitudes en nodo, componente,
    patrón temporal y severidad."""

@tool
def create_mock_ticket(severity: str, summary: str, node: str) -> str:
    """Simula creación de ticket en ServiceNow.
    Retorna ticket_id mock con timestamp."""
```

**Prompts extendidos en `src/llm/prompts.py`:**
- `AGENT_SYSTEM_PROMPT`: rol del agente, contexto de la infraestructura, instrucciones de uso de tools
- `TICKET_CREATION_PROMPT`: cuándo escalar automáticamente a CRITICAL

**Casos de uso documentados en `notebooks/07_llm_agent.ipynb`:**
1. *"¿El nodo R15 tuvo anomalías similares esta semana?"*
2. *"Compara el incidente de hoy con el del martes pasado"*
3. *"Este incidente es CRITICAL, crea un ticket"*
4. *"¿Cuáles son los 3 nodos más problemáticos del último mes?"*

**Nuevos endpoints API:**
```
POST /agent/chat
  Input:  { message: str, session_id: str, anomaly_context: AnomalyResult | None }
  Output: { response: str, tools_used: List[str], session_id: str }

DELETE /agent/session/{session_id}
  Output: { cleared: bool }
```

**Gestión de sesiones:** `Dict[str, ConversationBufferWindowMemory]` en memoria del proceso (simple, sin Redis — documentar limitación).

**Mock ServiceNow:**
- No requiere API real
- Retorna `{"ticket_id": "INC0001234", "created_at": "...", "status": "open"}`
- Documentar en README: *"En producción, reemplazar con llamada real a ServiceNow API"*

**Outputs:**
- `src/llm/agent.py`
- `src/llm/prompts.py` extendido
- `notebooks/07_llm_agent.ipynb` con 4 casos de uso documentados
- Tests en `tests/test_agent.py` (Ollama mockeado)

**Criterio de aceptación:** Agente responde usando herramientas correctamente en los 4 casos de uso. Memoria persiste dentro de la sesión. Mock ticket se crea para CRITICAL.

**Commit:** `feat: phase 9 - LLM agent with memory, tools and mock ServiceNow integration`

---

### Fase 10 — Drift Detection con Evidently
**Objetivo:** Monitorear si la distribución de los logs cambia con el tiempo, indicando que el modelo necesita re-entrenamiento.

**Contexto:**
En producción, la infraestructura cambia — nuevos servidores, nuevos componentes, updates de software. El modelo entrenado sobre logs históricos puede degradarse sin que nadie lo note. Evidently detecta ese drift automáticamente.

**Tipos de drift a monitorear:**

| Tipo | Qué mide | Tool Evidently |
|---|---|---|
| Data drift | Cambio en distribución de features (error_rate, burst_flag, etc.) | `DataDriftPreset` |
| Target drift | Cambio en ratio de anomalías detectadas | `TargetDriftPreset` |
| Model performance | Degradación de métricas si hay labels disponibles | `ClassificationPreset` |

**Pipeline de drift:**

```
Referencia: datos de entrenamiento (semanas 1-4)
Producción: datos recientes (última semana)
        ↓
Evidently Report → drift_score por feature
        ↓
Si drift_score > umbral → alerta en dashboard + log
        ↓
Si drift persistente > 3 días → trigger /retrain automático (opcional)
```

**Implementación en `src/monitoring/drift_detector.py`:**
```python
class DriftDetector:
    def fit_reference(self, reference_df: pd.DataFrame) -> None: ...
    def detect(self, current_df: pd.DataFrame) -> DriftReport: ...
    def get_drift_score(self) -> float: ...  # 0.0 = sin drift, 1.0 = drift total
    def generate_html_report(self, path: str) -> None: ...
```

**Notebook `notebooks/08_drift_detection.ipynb`:**
- Simular drift artificialmente: tomar logs de período distinto como "producción"
- Mostrar reporte Evidently con visualizaciones
- Analizar qué features driftan más (hipótesis: `error_rate` y `unique_nodes` serán las más sensibles)
- Decisión documentada: ¿cuándo re-entrenar? (umbral justificado)

**Nuevo endpoint API:**
```
GET /model/health
  Output: { model_version: str, last_trained: str,
            drift_score: float, drift_detected: bool,
            features_drifted: List[str], recommendation: str }
```

**Integración con dashboard:** nueva sección en Streamlit con semáforo de drift por feature.

**Outputs:**
- `src/monitoring/drift_detector.py`
- `src/monitoring/model_monitor.py`
- `notebooks/08_drift_detection.ipynb`
- `GET /model/health` funcional
- Tests en `tests/test_drift_detector.py`

**Criterio de aceptación:** Drift detectado correctamente cuando se usan datos de período distinto. Reporte HTML generado. Endpoint reporta estado real.

**Commit:** `feat: phase 10 - drift detection with Evidently and model health endpoint`

---

### Fase 11 — Dashboard Streaming en Tiempo Real
**Objetivo:** Transformar el dashboard estático de V1 en un panel operacional con eventos apareciendo en vivo — el salto visual más impactante para una demo.

**Arquitectura de streaming:**

```
event_generator.py
  → genera eventos sintéticos basados en distribución real del dataset
  → mezcla eventos normales (95%) con anomalías inyectadas (5%)
  → emite 1 evento cada 0.5s (configurable)
        ↓
Streamlit con st.empty() + st.rerun()
        ↓
Pipeline: evento → features → Isolation Forest → ¿anomalía?
        ↓ (si anomalía)
LLM Summarizer → severidad + resumen
        ↓
Panel de alertas se actualiza en vivo
```

**Generador de eventos en `src/streaming/event_generator.py`:**
```python
class EventGenerator:
    def __init__(self, reference_df: pd.DataFrame, anomaly_rate: float = 0.05): ...
    def stream(self) -> Generator[LogEvent, None, None]: ...
    def inject_anomaly(self) -> LogEvent: ...  # anomalía sintética controlada
```

**Cambios en `dashboard/app.py`:**

Nueva pestaña: **🔴 Live Monitor**
- Contador en tiempo real: eventos procesados / anomalías detectadas / tasa actual
- Feed de últimos 50 eventos con color por tipo (verde = normal, rojo = anomalía)
- Panel lateral: última anomalía detectada con resumen LLM
- Botón "Inject Anomaly" para demo manual — dispara anomalía sintética inmediatamente
- Velocidad de streaming configurable: slider 0.1s — 2.0s entre eventos

**Pestaña existente actualizada: 📊 Overview**
- Agregar sección de drift: semáforo por feature (verde/amarillo/rojo)
- Mostrar `model_version` y `last_trained` desde `/model/health`

**Modo demo robusto:**
- Si Ollama no está disponible → resúmenes mock pre-generados (no bloquea el streaming)
- Si dataset no está disponible → generar eventos completamente sintéticos

**Outputs:**
- `src/streaming/event_generator.py`
- `dashboard/app.py` actualizado
- Re-deploy en Streamlit Cloud

**Criterio de aceptación:** Streaming visible en vivo, "Inject Anomaly" funciona, resumen LLM aparece en ≤ 3s post-detección, modo demo funcional sin dependencias externas.

**Commit:** `feat: phase 11 - real-time streaming dashboard with live anomaly injection`

---

### Fase 12 — MLflow Completo + Tests V2 + README Final
**Objetivo:** Cerrar V2 con calidad de producción: tracking completo, tests nuevos en CI, README actualizado con arquitectura V2.

**MLflow — tracking completo:**
- Registrar experimentos de Fase 8 (active learner) y Fase 10 (drift)
- Model Registry: versionar modelos con stages (`Staging` → `Production`)
- `mlflow ui` instrucciones en README para explorar experimentos localmente
- Artefactos registrados: modelos, reportes Evidently HTML, curvas PR

**Tests V2 nuevos:**

| Archivo | Qué testea |
|---|---|
| `test_active_learner.py` | fit, predict, feature importance, `/retrain` endpoint |
| `test_agent.py` | tools individuales, memoria de sesión, mock Ollama |
| `test_drift_detector.py` | detección con datos artificialmente drifteados, `/model/health` |

**Regla de mocks en CI:**
- Ollama: siempre mockeado (`unittest.mock.patch("langchain_community.llms.Ollama"`)
- MLflow: usar `mlflow.set_tracking_uri("file:./mlflow_test")` en fixture
- Evidently: testear con DataFrames pequeños sintéticos

**Coverage objetivo V2:** mantener > 82% global (V1 tenía > 80%, agregar tests compensa código nuevo)

**README actualizado — nuevas secciones:**
- Diagrama de arquitectura V2 (ASCII o Mermaid)
- Tabla comparativa V1 vs V2 con métricas
- Sección "LLM Agent — ejemplos de uso" con 3 conversaciones reales
- Sección "Drift Detection" con captura del reporte Evidently
- GIF del dashboard streaming con anomalía inyectada en vivo
- Instrucciones `mlflow ui` para explorar experimentos

**Commit:** `feat: phase 12 - MLflow registry, V2 tests complete, README V2`

---

## 🗓️ Estimación de Tiempo V2

| Fase | Descripción | Días estimados |
|---|---|---|
| 8 | Active Learning con LightGBM | 3.0 |
| 9 | LLM Agent con memoria y tools | 4.0 |
| 10 | Drift Detection con Evidently | 2.5 |
| 11 | Dashboard Streaming | 3.0 |
| 12 | MLflow + Tests + README | 2.5 |
| **Total V2** | | **15 días** |

> Con ~3-4h/día: 3-4 semanas adicionales post-V1.

---

## 🗓️ Orden recomendado si el tiempo se acorta

Si por alguna razón el tiempo se reduce, ejecutar en este orden de prioridad:

1. **Fase 10 — Drift Detection** (mayor impacto en CV, cargo lo pide explícito como MLOps)
2. **Fase 9 — LLM Agent** (mayor diferenciador técnico)
3. **Fase 8 — Active Learning** (refuerza narrativa de mejora continua)
4. **Fase 11 — Streaming** (mayor impacto visual en demo)
5. **Fase 12 — Cierre** (siempre al final, no saltarla)

---

## 💼 Narrativa V2 para Entrevista SICPA

**Pregunta:** *"¿Cómo manejarías la degradación del modelo en producción?"*
> "Implementé drift detection con Evidently que monitorea la distribución de features en tiempo real. El endpoint `/model/health` reporta un drift score continuo y lista qué features están cambiando. Cuando el drift supera el umbral, el sistema puede disparar automáticamente un re-entrenamiento con `/retrain`, que usa los labels generados por el LLM como supervisión débil — lo que llamo active learning lite."

**Pregunta:** *"¿Cómo escalarías el análisis cuando hay muchas alertas simultáneas?"*
> "El agente LLM tiene memoria de sesión y herramientas para consultar historial. En lugar de analizar cada alerta en aislado, puede responder preguntas como '¿cuáles son los 3 nodos más problemáticos esta semana?' o 'compara este incidente con el del martes'. Para alertas CRITICAL, el agente crea automáticamente un ticket en ServiceNow — en el proyecto implementé un mock completo del flujo."

---

## 📌 Notas para Claude Code — V2

- **No modificar archivos V1 sin necesidad** — agregar funcionalidad nueva en archivos nuevos siempre que sea posible. Si hay que modificar `api/main.py` o `api/schemas.py`, hacerlo en un branch separado por fase.
- **Branch por fase:** `feature/phase-8-active-learning`, `feature/phase-9-llm-agent`, etc. Mismo workflow que V1: branch → PR → merge → CI verde antes de continuar.
- **Ollama debe estar corriendo localmente** antes de ejecutar cualquier código de Fase 9 (`ollama serve` en terminal separado). CI siempre usa mock.
- **MLflow UI** para verificar experimentos: `mlflow ui --backend-store-uri ./mlflow` → abrir `http://localhost:5000`
- **Evidently genera HTML** — no PDF. Guardar en `reports/drift/` con timestamp en nombre.
- **El generador de streaming** en Fase 11 debe ser determinístico con `random.seed()` para que los tests sean reproducibles.
- **`st.rerun()`** es el método correcto en Streamlit moderno — no `st.experimental_rerun()` (deprecated).
- **Memoria del agente** es in-process — se pierde al reiniciar la API. Documentar esta limitación explícitamente en el README como decisión arquitectónica (sin Redis por simplicidad de portafolio).
- Notebooks cerrar en VS Code antes de editar desde terminal.
- `make fix` antes de cada commit.
