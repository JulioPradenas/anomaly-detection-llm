.PHONY: install fix test run api dashboard

install:
	uv sync --extra dev

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests
	uv run mypy src

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --no-cov

run: api

api:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	uv run streamlit run dashboard/app.py

download-data:
	@echo "Descarga BGL desde: https://github.com/logpai/loghub"
	@echo "Coloca BGL.log en data/raw/BGL.log"
