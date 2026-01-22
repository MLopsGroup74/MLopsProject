import io
import os
import torch
import torch.nn.functional as F
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from PIL import Image
from torchvision import transforms as T
from contextlib import asynccontextmanager
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule
from transformers import CLIPModel, CLIPProcessor
from google.cloud import storage
from datetime import datetime
#from pyexpat import model

"""
run the fast api by running:
uv run uvicorn fast_api:app --host 0.0.0.0 --port 8000

afterwards go to http://localhost:8000/docs for interactive api
"""
# Configuration
CKPT_PATH = "models/model-epoch=17-val_acc=0.38.ckpt" 
DATA_DIR = "PokemonData/"

BUCKET_NAME="mlopsproject-data" 

inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and class names once at startup."""
    print("Startup: Loading DataModule for class names...")
    dm = ImageFolderDataModule(data_dir=DATA_DIR)
    app.state.class_names = dm.class_names
    
    print(f"Startup: Loading model from {CKPT_PATH}...")
    # load_from_checkpoint restores weights and hyperparameters
    app.state.model = ConvolutionalNetwork.load_from_checkpoint(CKPT_PATH)
    app.state.model.eval() # CRITICAL: Sets layers like Dropout to eval mode

    print("Startup: Loading CLIP model for monitoring...")
    app.state.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    app.state.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    print("Startup complete. Database will be created on first prediction.")
    yield
    print("Shutdown: Cleaning up...")
    del app.state.model
    del app.state.clip_model

#The background task function
def extract_image_features(img: Image.Image, clip_model, clip_processor):
    """Extracts the same features from the image"""
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
    return img_emb.squeeze().tolist()

def save_prediction_to_gcp(features: list, prediction: str):
    "Uploads prediction features as json file to a bucket"

    try:
        client = storage.Client()
        bucket= client.bucket(BUCKET_NAME)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        blob_name = f"predictions/prediction_{timestamp}.json"

        data = {
            "features": features,
            "prediction": prediction,
            "timestamp": timestamp
        }

        #Make JSON file and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            data=json.dumps(data),
            content_type="application/json"
        )
        print(f"DEBUG: Logged prediction to GCP: {blob_name}")
    except Exception as e:
        print(f"CLOUD ERROR:{e}")

app = FastAPI(title="Pokémon Inference API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Pokémon Classifier API is running!"}

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    # 1. Load image from upload
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    clip_features = extract_image_features(img, app.state.clip_model, app.state.clip_processor)
    

    # 2. Preprocess image
    input_tensor = inference_transform(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

    # 3. Inference
    with torch.no_grad(): # Disable gradient calculation for speed/memory
        logits = app.state.model(input_tensor)
        probs = F.softmax(logits, dim=1) # Convert to probabilities
        conf, pred_idx = torch.max(probs, dim=1)

    #Prepare data for background 
    pred_label = app.state.class_names[pred_idx.item()]

    #Add the task to run AFTER the response is sent
    background_tasks.add_task(save_prediction_to_gcp, clip_features, pred_label)
        
    return {
        "prediction": app.state.class_names[pred_idx.item()],
        "confidence": float(conf.item()),
        "class_index": int(pred_idx.item())
    }

