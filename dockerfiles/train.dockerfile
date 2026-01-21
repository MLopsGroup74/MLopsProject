# Use the official uv image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 1. Set the working directory FIRST
WORKDIR /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Copy only dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# 4. Install dependencies
# --frozen ensures we use the lockfile exactly
# --no-install-project skips installing your local 'src' as a package for now
RUN uv sync --frozen --no-install-project --no-dev

# 5. Copy the rest of the application
COPY src/ src/
COPY README.md README.md

# 6. Final sync to install the project itself
RUN uv sync --frozen --no-dev


# Add PYTHONPATH so it finds logging_setup.py and src
ENV PYTHONPATH="/app:/app/src"

# Run with python directly to skip the 'uv' sync check at runtime
ENTRYPOINT ["python", "-m", "assignment.train"]

# 7. Use the virtual environment's path automatically
#ENV PATH="/app/.venv/bin:$PATH"

# Set entrypoint
#ENTRYPOINT ["uv", "run", "src/assignment/train.py"]
