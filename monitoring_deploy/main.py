import torch
import torch.nn.functional as F
from google.cloud import storage
import functions_framework
import os

# Important: Copy your ConvolutionalNetwork class definition here
# so the function knows the 'body' of your model.
import torch.nn as nn
class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_classes=150):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Global variables to load the model once
BUCKET_NAME = "mlopsproject-data"
MODEL_FILE = "models/model-epoch=20-val_acc=0.38.ckpt"
model = None


def load_model():
    global model
    if model is None:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE)
        blob.download_to_filename("/tmp/model.ckpt")

        model = ConvolutionalNetwork(num_classes=150)
        # Load weights - map to CPU since Functions don't have GPUs
        checkpoint = torch.load("/tmp/model.ckpt", map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

@functions_framework.http
def predict_pokemon(request):
    load_model()
    request_json = request.get_json()
    if request.method == 'GET':
        return {"status": "healthy"}, 200
    if request_json and "image_data" in request_json:
        # Expecting a list of floats representing the image pixels
        input_tensor = torch.tensor(request_json["image_data"]).float().view(1, 3, 224, 224)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return {"pokemon_id": prediction}
    return {"error": "No input_data provided."}
