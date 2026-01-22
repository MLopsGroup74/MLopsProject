FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

RUN apk add --no-cache bash

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml


RUN uv sync --frozen --no-install-project

COPY src src/
COPY monitoring/ monitoring/
RUN uv sync --frozen


ENTRYPOINT ["uv", "run", "uvicorn", "monitoring.monitoring_api:app", "--host", "0.0.0.0", "--port", "8000"]


