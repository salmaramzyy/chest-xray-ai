# download_data.py
import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

print("Loading dataset (already cached, will be fast)...")
dataset = load_dataset("hf-vision/chest-xray-pneumonia", split="train[:200]")

os.makedirs("data/raw/images", exist_ok=True)

saved = []
print("Saving images...")
for i, sample in enumerate(tqdm(dataset)):
    img_path = f"data/raw/images/{i:04d}.png"
    sample["image"].convert("RGB").save(img_path)
    saved.append({
        "uid": f"{i:04d}",
        "image_path": img_path,
        "label": sample["label"],
    })

df = pd.DataFrame(saved)
df.to_csv("data/raw/metadata.csv", index=False)
print(f"Done! Saved {len(saved)} images.")