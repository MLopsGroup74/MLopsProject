import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms as T
from contextlib import asynccontextmanager

# Import your custom model class
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule

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
    
    yield
    print("Shutdown: Cleaning up...")
    del app.state.model

app = FastAPI(title="Pokémon Inference API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Pokémon Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Load image from upload
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Preprocess image
    input_tensor = inference_transform(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

    # 3. Inference
    with torch.no_grad(): # Disable gradient calculation for speed/memory
        logits = app.state.model(input_tensor)
        probs = F.softmax(logits, dim=1) # Convert to probabilities
        
        conf, pred_idx = torch.max(probs, dim=1)
        
    return {
        "prediction": app.state.class_names[pred_idx.item()],
        "confidence": float(conf.item()),
        "class_index": int(pred_idx.item())
    }


"""
from fastapi import FastAPI
from src.assignment.data import ImageFolderDataModule

DATA_DIR = "PokemonData/"

app = FastAPI()

# --- LOAD THIS ONCE HERE (Global) ---
# This happens during startup, so the root function stays fast!
print("Scanning PokemonData... please wait.")
dm = ImageFolderDataModule(data_dir=DATA_DIR)
print(f"Done! Found {dm.num_classes} classes.")

@app.get("/")
def root():
    # Now this function just looks at the data already in RAM
    return {
        "num_classes": dm.num_classes, 
        "classes": list(dm.class_names)[:5]
    }
"""