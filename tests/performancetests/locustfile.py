import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from locust import HttpUser, task, between
# Locust file

# --- IMAGE PREPARATION HELPER ---
def prepare_image_payload(image_path):
    """Converts a local image to the list format expected by the API."""
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img).transpose(2, 0, 1)  # Change to (C, H, W)
        img_array = img_array / 255.0                 # Normalize
        flattened_data = img_array.flatten().tolist()
        return {"image_data": flattened_data}
    except Exception as e:
        logging.error(f"Could not prepare image payload: {e}")
        return {"image_data": [0.0] * 150528} # Fallback to dummy

# --- PATH LOGIC ---
# locustfile.py is in tests/performancetests/
_repo_root = Path(__file__).resolve().parents[2]
_image_path = _repo_root / "tests" / "test_data" / "test_pokemon.jpg"

# Prepare the payload ONCE at module level
logging.info(f"Loading test image from: {_image_path}")
REAL_PAYLOAD = prepare_image_payload(_image_path)

class PokemonUser(HttpUser):
    wait_time = between(1, 2)

    @task(10)
    def predict_pokemon(self):
        """Simulates a user sending the real image for classification."""
        # Use the pre-processed REAL_PAYLOAD
        with self.client.post("/", json=REAL_PAYLOAD, catch_response=True, name="POST /predict") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "pokemon_id" in data:
                        response.success()
                    else:
                        response.failure(f"Response missing 'pokemon_id': {data}")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
            else:
                response.failure(f"Predict failed with status {response.status_code}: {response.text}")

    @task(1)
    def health_check(self):
        """Standard GET health check."""
        with self.client.get("/", name="GET Health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
