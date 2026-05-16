import json
import torch
import numpy as np
import open_clip
from PIL import Image

class CLIPReportGenerator:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        print("Loading CLIP...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.embeddings = None
        self.database = []
        print("CLIP ready.")

    def build_index(self, image_paths: list, reports: list):
        print(f"Building CLIP index for {len(image_paths)} images...")
        embs = []
        for i, path in enumerate(image_paths):
            try:
                img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    e = self.model.encode_image(img)
                    e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy())
                self.database.append({"image_path": path, "report": reports[i]})
            except Exception as ex:
                print(f"  Skipping {path}: {ex}")
        self.embeddings = np.vstack(embs)
        print("CLIP index ready.")

    def generate_report(self, image_path: str, top_k=3) -> dict:
        if self.embeddings is None:
            raise RuntimeError("Call build_index() first.")

        img = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model.encode_image(img)
            q = q / q.norm(dim=-1, keepdim=True)
        q = q.cpu().numpy()

        scores = (self.embeddings @ q.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]

        retrieved = [
            {"similarity": float(scores[i]),
             "image_path": self.database[i]["image_path"],
             "report": self.database[i]["report"]}
            for i in top_idx
        ]
        best = retrieved[0]
        return {
            "findings": best["report"],
            "impression": f"Retrieved from similar case (similarity: {best['similarity']:.3f})",
            "recommendations": "Compare with prior imaging if available.",
            "raw_output": best["report"],
            "retrieved_cases": retrieved,
        }

    def save_index(self, path: str):
        np.save(f"{path}_emb.npy", self.embeddings)
        with open(f"{path}_db.json", "w") as f:
            json.dump(self.database, f)

    def load_index(self, path: str):
        self.embeddings = np.load(f"{path}_emb.npy")
        with open(f"{path}_db.json") as f:
            self.database = json.load(f)