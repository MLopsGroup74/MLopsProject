from locust import HttpUser, task, between
import json

# Pre-load the data so we don't waste CPU during the test
with open("data.json", "r") as f:
    real_payload = json.load(f)

class PokemonUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_real_image(self):
        self.client.post("/", json=real_payload)
