import pandas as pd
import os
import sys
import random
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

## 1. Initialize CLIP 
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def create_reference(data_dir, output_file="monitoring/reference_database.csv", sample_size=500):
    reference_rows = []
    all_image_paths = []

    #Go through PokemonData folder
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root,file))

    #Randomized sampling
    random.seed(42)
    random.shuffle(all_image_paths)

    #Take the first N from the shuffled list
    selected_paths = all_image_paths[:sample_size]    

    print(f"Extracting features from {len(selected_paths)} random images...")

    for img_path in selected_paths:
        try:
            image= Image.open(img_path).convert("RGB")

            # 2. Extract 512-dimensional vector
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_emb = model.get_image_features(**inputs)

            # Convert to a list of 512 floats
            embedding_list = img_emb.squeeze().tolist()

            # Columns will be f_0, f_1, ... f_511
            row = {f"f_{i}": val for i, val in enumerate(embedding_list)}
            row["target"] = os.path.basename(os.path.dirname(img_path))
            
            reference_rows.append(row)
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
                
    df = pd.DataFrame(reference_rows)
    df.to_csv(output_file, index=False)
    print(f"Success! Saved CLIP features to {output_file}")
    

if __name__ == "__main__":
    create_reference("PokemonData/")
