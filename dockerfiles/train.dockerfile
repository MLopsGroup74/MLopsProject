FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY README.md README.md
COPY src/ src/
# COPY PokemonData/ PokemonData/

WORKDIR /
RUN uv sync --locked --no-cache

#ENTRYPOINT ["uv", "run", "src/assignment/train.py", "--data_dir", "sample", "--max_epochs", "1", "--batch_size", "2"]

ENTRYPOINT ["uv", "run", "src/assignment/train-kopi.py"]
