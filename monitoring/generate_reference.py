import pandas as pd
import os
import sys
import random
from PIL import Image
import numpy as np

# Use your existing extraction logic from fast_api.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fast_api import extract_image_features

def create_reference(data_dir, output_file="monitoring/reference_database.csv", sample_size=1000):
    reference_rows = []
    all_image_paths = []

    full_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", data_dir))

    #go through PokemonData folder
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root,file))

    #shuffle folders to get a random images reference set
    random.seed(42)
    random.shuffle(all_image_paths)

    #Take the first N from the shuffled list
    selected_paths = all_image_paths[:sample_size]    

    print(f"Extracting features from {len(selected_paths)} random images...")

    for img_path in selected_paths:
                img = Image.open(img_path).convert("RGB")
                # Extract the 3 core features
                b, c, s = extract_image_features(img)
                # Get the folder name as the 'target' label
                label = os.path.basename(os.path.dirname(img_path))
                
                reference_rows.append({
                    "brightness": b,
                    "contrast": c,
                    "sharpness": s,
                    "target": label
                })
    
                
    df = pd.DataFrame(reference_rows)
    df.to_csv(output_file, index=False)
    print(f"Randomized reference data saved to {output_file}")
    

if __name__ == "__main__":
    create_reference("PokemonData/")

