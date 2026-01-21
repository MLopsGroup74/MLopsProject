FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy pyproject + lock
COPY pyproject.toml uv.lock /app/
WORKDIR /app

# Install all dependencies (including PyTorch & PyTorch Lightning)
RUN uv sync --locked --no-cache

# Copy project code
COPY src/ src/
COPY README.md README.md

# Set entrypoint for training
ENTRYPOINT ["uv", "run", "src/assignment/train.py"]
