"""LLM provider selection — Groq (cloud) or Ollama (local).

Uses Groq when GROQ_API_KEY is set (Streamlit Cloud deploy), otherwise falls
back to a local Ollama server. This lets the same code run locally with
llama3.2 and on a public demo with a hosted model — no code changes needed.
"""

import os
from typing import Any

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OLLAMA_MODEL = "llama3.2"


def using_groq() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def provider_name() -> str:
    return "groq" if using_groq() else "ollama"


def get_chat_model(
    temperature: float = 0.1,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = "http://localhost:11434",
) -> Any:
    """Return a LangChain chat model for the active provider."""
    if using_groq():
        from langchain_groq import ChatGroq

        # ChatGroq reads GROQ_API_KEY from the environment automatically.
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=temperature)
