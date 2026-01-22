"""Load environment variables from .env file.

Loads Weights & Biases API key and other configuration from environment.
"""

import os

from dotenv import load_dotenv

load_dotenv()

api_key: str | None = os.getenv("WANDB_API_KEY")

print("Environment (Wandb) variables loaded from .env")
