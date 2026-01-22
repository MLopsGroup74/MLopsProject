import httpx
import pytest
import numpy as np
from PIL import Image
import json

URL = "https://pokemon-pred-function-952726112544.europe-west1.run.app"


def prepare_image(image_path):
    """Converts a local image to the list format expected by the API."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).transpose(2, 0, 1) # Change to (C, H, W)
    img_array = img_array / 255.0                # Normalize
    return img_array.flatten().tolist()

def test_predict_real_image():
    """Test the API with a real image file."""
    image_data = prepare_image("tests/test_data/test_pokemon.jpg")
    payload = {"image_data": image_data}

    with httpx.Client() as client:
        # Increase timeout because real images take longer to upload
        response = client.post(URL, json=payload, timeout=30.0)

    assert response.status_code == 200
    data = response.json()
    assert "pokemon_id" in data
