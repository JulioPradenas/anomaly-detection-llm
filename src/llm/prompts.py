"""Prompt templates for the LLM summarizer."""

from langchain_core.prompts import ChatPromptTemplate

ANOMALY_SUMMARY_TEMPLATE = """\
Eres un ingeniero experto en operaciones IT analizando logs de sistema.
Se detectó una anomalía en el nodo {node} a las {timestamp}.

Contexto de los últimos 5 minutos:
- Eventos totales: {total_events}
- Errores: {error_count} ({error_rate} del total)
- Warnings: {warning_count}
- Fatales: {fatal_count}
- Componentes afectados: {components}
- Severidad promedio (escala 0-4): {avg_severity}
- Mensajes representativos:
{sample_messages}

Anomaly score: {anomaly_score:.4f}

Proporciona un análisis estructurado con exactamente este formato JSON:
{{
  "resumen": "<2 oraciones máximo describiendo el incidente>",
  "severidad": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "causa_probable": "<causa técnica probable>",
  "accion_recomendada": "<acción inmediata concreta>"
}}
Responde únicamente con el JSON, sin texto adicional.\
"""

anomaly_summary_prompt = ChatPromptTemplate.from_template(ANOMALY_SUMMARY_TEMPLATE)
