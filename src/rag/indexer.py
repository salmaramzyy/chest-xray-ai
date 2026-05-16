import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

class ColPaliIndexer:
    MODEL_ID = "vidore/colpali-v1.2"

    def __init__(self):
        token = os.getenv("HF_TOKEN")
        print("Loading ColPali on CPU (no GPU required)...")

        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = ColPali.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float32,   # float32 works on CPU
            device_map="cpu",            # force CPU
            token=token,
        ).eval()

        self.processor = ColPaliProcessor.from_pretrained(
            self.MODEL_ID, token=token
        )

        self.device = torch.device("cpu")
        self.doc_embeddings = []
        self.doc_metadata = []
        print("ColPali loaded on CPU.")

    def index_documents(self, docs: list, batch_size=1):  # batch_size=1 for CPU
        print(f"Indexing {len(docs)} documents with ColPali...")
        for i, doc in enumerate(docs):
            try:
                img = Image.open(doc["image"]).convert("RGB") \
                      if isinstance(doc["image"], str) else doc["image"]

                with torch.no_grad():
                    inp = self.processor.process_images([img])
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    emb = self.model(**inp)

                self.doc_embeddings.append(emb[0].cpu().float())
                self.doc_metadata.append({
                    "text": doc["text"],
                    "source": doc.get("source", f"doc_{i}")
                })
                if (i + 1) % 5 == 0:
                    print(f"  Indexed {i+1}/{len(docs)}")
            except Exception as e:
                print(f"  Skipping doc {i}: {e}")

        print(f"Done. {len(self.doc_metadata)} documents indexed.")

    def retrieve(self, query_text: str, top_k=3) -> list:
        if not self.doc_embeddings:
            raise RuntimeError("Index empty. Call index_documents() first.")

        with torch.no_grad():
            q_inp = self.processor.process_queries([query_text])
            q_inp = {k: v.to(self.device) for k, v in q_inp.items()}
            q_emb = self.model(**q_inp)[0].cpu().float()

        scores = []
        for doc_emb in self.doc_embeddings:
            sim = torch.matmul(q_emb, doc_emb.T)
            scores.append(sim.max(dim=1).values.sum().item())

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "text": self.doc_metadata[i]["text"],
                "source": self.doc_metadata[i]["source"],
                "score": scores[i],
            }
            for i in top_idx
        ]