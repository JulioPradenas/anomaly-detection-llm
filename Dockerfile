# ── Builder ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY api/ api/
COPY models/ models/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
