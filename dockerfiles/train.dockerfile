# Use the official uv image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 1. Set the working directory
WORKDIR /app

# 2. Install system dependencies (build-essential helps with C-based packages)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Copy only dependency files first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# 4. Pre-install dependencies into the image's virtual environment
# We use --frozen to ensure it matches your lockfile exactly
RUN uv sync --frozen --no-install-project --no-dev

# 5. Copy your entire source code (including src/ and logging_setup.py)
COPY src/ src/
COPY README.md README.md

# 6. Final sync to include your project code in the environment
RUN uv sync --frozen --no-dev

# 7. CRITICAL: Add the .venv to the PATH
ENV PATH="/app/.venv/bin:$PATH"
# Ensure 'import assignment' works from the root
ENV PYTHONPATH="/app:/app/src"

# 8. Use uv run with --no-sync to ensure it starts instantly without internet
# We use the module syntax (-m) to avoid pathing issues with 'src'
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "assignment.train"]
