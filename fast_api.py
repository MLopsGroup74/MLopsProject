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