import json
import logging
import random
from pathlib import Path
from locust import HttpUser, task, between

# --- PATH LOGIC ---
# locustfile.py is in tests/performancetests/
# We go up two levels to reach the project root
_repo_root = Path(__file__).resolve().parents[2]
# Look for data.json or the specific test_pokemon.jpg payload in the root
_data_path = _repo_root / "data.json"

real_payload = {}

# Load payload at the module level (only once when Locust starts)
if _data_path.exists():
    try:
        with open(_data_path, "r") as f:
            real_payload = json.load(f)
        logging.info(f"Successfully loaded payload from {_data_path}")
    except Exception as e:
        logging.error(f"Failed to parse JSON at {_data_path}: {e}")
else:
    # Fallback: Generate a dummy payload if file is missing
    # This prevents the load test from crashing if the file is moved
    logging.warning(f"data.json not found at {_data_path}. Using dummy data.")
    real_payload = {"image_data": [0.0] * 150528}

class PokemonUser(HttpUser):
    # wait_time simulates real human behavior (resting 1-2s between requests)
    wait_time = between(1, 2)

    @task(10)  # Heavy weight: 10/11 requests will be predictions
    def predict_pokemon(self):
        """Simulates a user sending an image for classification."""
        # Handle cases where data.json might be a list of images or a single dict
        payload = random.choice(real_payload) if isinstance(real_payload, list) else real_payload

        # We use the root endpoint "/" as that's where your Cloud Function is listening
        with self.client.post("/", json=payload, catch_response=True, name="POST /predict") as response:
            if response.status_code == 200:
                try:
                    # Basic validation of the response schema
                    if "pokemon_id" in response.json():
                        response.success()
                    else:
                        response.failure("Response missing 'pokemon_id'")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(f"Predicted failed with status {response.status_code}")

    @task(1)  # Light weight: 1/11 requests will be a health check
    def health_check(self):
        """Lightweight check to measure latency without model inference."""
        # If your main.py handles GET / as a welcome message, this is useful
        with self.client.get("/", name="GET Health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
