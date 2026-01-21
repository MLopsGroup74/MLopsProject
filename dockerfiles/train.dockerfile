FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install OS dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY README.md README.md

# Install all dependencies from lock file (including pytorch-lightning)
RUN uv sync --locked --no-cache

# Set working directory
WORKDIR /

# Entrypoint to run your training
ENTRYPOINT ["uv", "run", "src/assignment/train.py"]
