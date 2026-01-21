import io
import os
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from PIL import Image
from torchvision import transforms as T
from contextlib import asynccontextmanager
from datetime import datetime
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule

"""
run the fast api by running:
uv run uvicorn fast_api:app --host 0.0.0.0 --port 8000

afterwards go to http://localhost:8000/docs for interactive api
"""
# Configuration
CKPT_PATH = "models/model-epoch=17-val_acc=0.38.ckpt" # Ensure this filename is correct
DATA_DIR = "PokemonData/"

# Define the exact same transforms used during training/validation
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
    print("Startup complete. Database will be created on first prediction.")
    yield
    print("Shutdown: Cleaning up...")
    del app.state.model

#The background task function
def extract_image_features(img: Image.Image):
    """Extracts the same features from the image"""
    img_gray = img.convert("L")  # Convert to grayscale for feature extraction
    img_np = np.array(img_gray)
    
    avg_brightness = np.mean(img_np)
    contrast = np.std(img_np)
    sharpness = np.mean(np.abs(np.gradient(img_np)))
    
    return float(avg_brightness), float(contrast), float(sharpness)

def add_to_database(brightness: float, contrast: float, sharpness: float, prediction: str):
    """Saves the timestamp, features, and prediction to CSV."""
    try:
        os.makedirs("monitoring", exist_ok=True)
        file_path = "monitoring/prediction_database.csv"
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, "a") as f:
            if not file_exists:
                f.write("brightness,contrast,sharpness,prediction\n")
            f.write(f"{brightness},{contrast},{sharpness},{prediction}\n")
        print(f"DEBUG: Successfully saved to {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"DEBUG ERROR: Could not write to file: {e}")

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
    
    brightness, contrast, sharpness = extract_image_features(img)

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
    background_tasks.add_task(
        add_to_database, 
        brightness, 
        contrast, 
        sharpness, 
        pred_label
    )
        
    return {
        "prediction": app.state.class_names[pred_idx.item()],
        "confidence": float(conf.item()),
        "class_index": int(pred_idx.item())
    }


