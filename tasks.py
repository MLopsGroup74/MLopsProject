import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "assignment"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess raw data into training format.

    Args:
        ctx: Invoke context for running commands.
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train the model using Lightning trainer.

    Args:
        ctx: Invoke context for running commands.
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests with coverage reporting.

    Args:
        ctx: Invoke context for running commands.
    """
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build Docker images for training and API.

    Args:
        ctx: Invoke context for running commands.
        progress: Docker build progress output style (plain, auto, or tty).
    """
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation using MkDocs.

    Args:
        ctx: Invoke context for running commands.
    """
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation locally with live reload.

    Args:
        ctx: Invoke context for running commands.
    """
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
