FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY PokemonData/ PokemonData/
COPY models/ models/
COPY src src/

RUN uv sync --frozen --no-install-project



RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.assignment.api:app", "--host", "0.0.0.0", "--port", "8000"]
