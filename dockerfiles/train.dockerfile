
# Base image with Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc python3-dev libffi-dev libssl-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install Python dependencies from lock file (including PyTorch, Lightning, Wandb)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ src/
COPY README.md README.md

# Set working directory
WORKDIR /

# Install the project itself (so your code is available to import)
RUN uv sync --locked --no-cache

# Set entrypoint
ENTRYPOINT ["uv", "run", "src/assignment/train.py"]
