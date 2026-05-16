import os
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

class RAGQAPipeline:
    """
    RAG QA Pipeline:
    1. ColPali retrieves relevant documents
    2. Groq (llama-4-scout) generates grounded answer
    """

    def __init__(self, indexer, medgemma=None):
        from groq import Groq
        self.indexer = indexer
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print("RAG QA pipeline ready.")

    def answer(self, image_path: str, question: str, top_k=3) -> dict:
        # Step 1 — ColPali retrieval
        print("Retrieving relevant documents...")
        retrieved = self.indexer.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"[{d['source']}]\n{d['text']}" for d in retrieved
        )

        # Step 2 — Encode image
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Step 3 — Build RAG prompt
        prompt = f"""You are an expert radiologist with access to a medical knowledge base.

RETRIEVED SIMILAR CASES FROM KNOWLEDGE BASE:
{context}

CLINICAL QUESTION: {question}

Using both the chest X-ray image and the retrieved cases, provide a clear and grounded answer."""

        # Step 4 — Generate answer
        print("Generating answer...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=400,
        )

        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "retrieved_context": retrieved,
            "sources": [d["source"] for d in retrieved],
        }