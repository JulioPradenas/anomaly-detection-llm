"""Conversational LLM agent with memory and tools for anomaly investigation."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.llm.prompts import AGENT_SYSTEM_PROMPT
from src.llm.provider import get_chat_model


class LogAgent:
    """Stateful conversational agent for IT log anomaly investigation.

    Uses LangChain 1.x create_agent (LangGraph-based) with MemorySaver for
    per-session conversation history. Sessions are keyed by session_id strings.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        features_path: Path = Path("data/processed/features_train.parquet"),
        labels_path: Path = Path("data/labels/llm_confirmed.parquet"),
    ) -> None:
        self.model = model
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        self._checkpointer = MemorySaver()
        self._agent: Any = None
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                get_chat_model(ollama_model=self.model).invoke("ping")
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def chat(
        self,
        message: str,
        session_id: str,
        anomaly_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a message and return agent response with tool usage metadata."""
        if not self.is_available:
            return {
                "response": "LLM no disponible — verifica que Ollama esté corriendo con llama3.2.",
                "tools_used": [],
                "session_id": session_id,
            }

        agent = self._get_agent()
        full_message = _build_message(message, anomaly_context)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=full_message)]},
                config={"configurable": {"thread_id": session_id}},
            )
            response_msg = result["messages"][-1].content if result.get("messages") else ""
            tools_used = _extract_tools_used(result.get("messages", []))
            return {
                "response": response_msg,
                "tools_used": tools_used,
                "session_id": session_id,
            }
        except Exception as e:
            return {
                "response": f"Error al procesar: {e!s}",
                "tools_used": [],
                "session_id": session_id,
            }

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session. Returns True if it existed."""
        existed = bool(self._checkpointer.storage.get(session_id))
        self._checkpointer.delete_thread(session_id)
        return existed

    def _get_agent(self) -> Any:
        if self._agent is None:
            llm = get_chat_model(temperature=0.1, ollama_model=self.model)
            tools = self._build_tools()
            self._agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
                checkpointer=self._checkpointer,
            )
        return self._agent

    def _build_tools(self) -> list:
        labels_path = self.labels_path

        @tool
        def query_anomaly_history(node: str, days_back: int = 7) -> str:
            """Consulta anomalías históricas de un nodo en los últimos N días.
            Retorna conteo por severidad y timestamps recientes del historial."""
            if not labels_path.exists():
                return f"Sin historial disponible para nodo {node} (ejecuta notebook 06 primero)."
            df = pd.read_parquet(labels_path)
            if "node" not in df.columns:
                return "Historial no contiene columna 'node'."
            node_df = df[df["node"].str.contains(node, case=False, na=False)]
            if node_df.empty:
                return f"Sin anomalías registradas para nodo {node}."
            total = len(node_df)
            sev_col = "severidad" if "severidad" in df.columns else None
            if sev_col:
                critical = int((node_df[sev_col] == "CRITICAL").sum())
                high = int((node_df[sev_col] == "HIGH").sum())
                medium = int((node_df[sev_col] == "MEDIUM").sum())
                return (
                    f"Nodo {node}: {total} anomalías en historial — "
                    f"CRITICAL: {critical}, HIGH: {high}, MEDIUM: {medium}."
                )
            return f"Nodo {node}: {total} anomalías en historial."

        @tool
        def top_anomalous_nodes(limit: int = 5) -> str:
            """Lista los nodos con más anomalías en el historial, ordenados de mayor a menor.
            Úsalo para preguntas abiertas como '¿qué nodo tiene más anomalías?'."""
            if not labels_path.exists():
                return "Sin historial disponible (ejecuta notebook 06 o genera los labels primero)."
            df = pd.read_parquet(labels_path)
            if "node" not in df.columns:
                return "Historial no contiene columna 'node'."
            counts = df["node"].value_counts().head(limit)
            lines = [
                f"{i + 1}. {node} — {n} anomalías" for i, (node, n) in enumerate(counts.items())
            ]
            return "Nodos con más anomalías:\n" + "\n".join(lines)

        @tool
        def list_recent_incidents(limit: int = 5) -> str:
            """Lista los incidentes más recientes con su anomaly_id, nodo y severidad.
            Úsalo cuando pidan 'los últimos N incidentes' o antes de comparar
            incidentes (compare_incidents necesita los anomaly_id de aquí)."""
            if not labels_path.exists():
                return "Sin historial disponible (ejecuta notebook 06 o genera los labels primero)."
            df = pd.read_parquet(labels_path)
            if "anomaly_id" not in df.columns or "timestamp" not in df.columns:
                return "El historial no contiene anomaly_id/timestamp para listar incidentes."
            recent = df.sort_values("timestamp", ascending=False).head(limit)
            lines = [
                f"{i + 1}. id={r['anomaly_id']} | nodo={r.get('node', 'N/A')} | "
                f"severidad={r.get('severidad', 'N/A')} | {r['timestamp']}"
                for i, (_, r) in enumerate(recent.iterrows())
            ]
            return "Incidentes más recientes:\n" + "\n".join(lines)

        @tool
        def get_anomaly_details(anomaly_id: str) -> str:
            """Retorna detalles completos de una anomalía específica por su ID."""
            if not labels_path.exists():
                return f"Sin datos para anomalía {anomaly_id}."
            df = pd.read_parquet(labels_path)
            if "anomaly_id" not in df.columns:
                return f"El historial no está indexado por anomaly_id. ID solicitado: {anomaly_id}."
            row = df[df["anomaly_id"] == anomaly_id]
            if row.empty:
                return f"Anomalía {anomaly_id} no encontrada en el historial."
            r = row.iloc[0]
            return (
                f"Anomalía {anomaly_id}: "
                f"nodo={r.get('node', 'N/A')}, "
                f"severidad={r.get('severidad', 'N/A')}, "
                f"timestamp={r.get('timestamp', 'N/A')}."
            )

        @tool
        def compare_incidents(anomaly_id_1: str, anomaly_id_2: str) -> str:
            """Compara dos incidentes: similitudes en nodo, severidad y patrón temporal."""
            if not labels_path.exists():
                return "Historial no disponible para comparar incidentes."
            df = pd.read_parquet(labels_path)
            if "anomaly_id" not in df.columns:
                return "El historial actual no soporta comparación por ID."
            r1 = df[df["anomaly_id"] == anomaly_id_1]
            r2 = df[df["anomaly_id"] == anomaly_id_2]
            if r1.empty or r2.empty:
                missing = anomaly_id_1 if r1.empty else anomaly_id_2
                return f"Incidente {missing} no encontrado en historial."
            a, b = r1.iloc[0], r2.iloc[0]
            same_node = a.get("node") == b.get("node")
            same_sev = a.get("severidad") == b.get("severidad")
            return (
                f"Comparación {anomaly_id_1} vs {anomaly_id_2}: "
                f"nodo {'IGUAL' if same_node else 'DIFERENTE'} "
                f"({a.get('node')} vs {b.get('node')}), "
                f"severidad {'IGUAL' if same_sev else 'DIFERENTE'} "
                f"({a.get('severidad')} vs {b.get('severidad')})."
            )

        @tool
        def create_mock_ticket(severity: str, summary: str, node: str) -> str:
            """Simula la creación de un ticket en ServiceNow para incidentes críticos.
            Úsalo cuando la severidad sea HIGH o CRITICAL."""
            ticket_id = f"INC{uuid.uuid4().hex[:8].upper()}"
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            assignee = "NOC-Team-1" if severity in ("HIGH", "CRITICAL") else "NOC-Team-2"
            return (
                f"Ticket creado exitosamente: {ticket_id} | "
                f"Severidad: {severity} | Nodo: {node} | "
                f"Resumen: {summary[:120]} | "
                f"Creado: {ts} | Estado: OPEN | "
                f"Asignado a: {assignee} (mock ServiceNow)"
            )

        return [
            query_anomaly_history,
            top_anomalous_nodes,
            list_recent_incidents,
            get_anomaly_details,
            compare_incidents,
            create_mock_ticket,
        ]


def _build_message(message: str, anomaly_context: dict[str, Any] | None) -> str:
    if not anomaly_context:
        return message
    ctx = (
        f"[Contexto — nodo: {anomaly_context.get('node', 'N/A')}, "
        f"score: {float(anomaly_context.get('anomaly_score', 0)):.3f}, "
        f"timestamp: {anomaly_context.get('timestamp', 'N/A')}]\n\n"
    )
    return ctx + message


def _extract_tools_used(messages: list) -> list[str]:
    """Extract tool names from LangGraph message list (ToolMessage type)."""
    tools: list[str] = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if name and msg.__class__.__name__ == "ToolMessage":
            if name not in tools:
                tools.append(name)
    return tools
