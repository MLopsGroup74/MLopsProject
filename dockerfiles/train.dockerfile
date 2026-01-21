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

# Add the current directory to PYTHONPATH as a backup
ENV PYTHONPATH="/app"

# Use -m to run the script as a module
# Note: Use dots (.) instead of slashes (/) and remove the .py extension
ENTRYPOINT ["python", "-m", "src.assignment.train"]
