import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

class LightweightRetriever:
    """
    CPU-friendly retriever using sentence-transformers.
    Same interface as ColPaliIndexer.
    Used in the demo app since ColPali requires GPU.
    ColPali implementation is preserved in indexer.py for reference.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading lightweight retriever...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # tiny, fast, CPU-friendly
        self.doc_embeddings = None
        self.doc_metadata = []
        print("Retriever ready.")

    def index_documents(self, docs: list, batch_size=16):
        print(f"Indexing {len(docs)} documents...")
        texts = []
        for doc in docs:
            self.doc_metadata.append({
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
            })
            texts.append(doc["text"])

        self.doc_embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        print(f"Indexed {len(docs)} documents.")

    def retrieve(self, query_text: str, top_k=3) -> list:
        if self.doc_embeddings is None:
            raise RuntimeError("Call index_documents() first.")

        query_emb = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        scores = (self.doc_embeddings @ query_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self.doc_metadata[i]["text"],
                "source": self.doc_metadata[i]["source"],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]