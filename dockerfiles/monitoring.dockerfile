FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY monitoring/ monitoring/

RUN uv sync --frozen --no-install-project


RUN uv sync --frozen


ENTRYPOINT ["uv", "run", "uvicorn", "monitoring.monitoring_fast_api:app", "--host", "0.0.0.0", "--port", "8000"]


