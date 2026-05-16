import os
import re
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

class MedGemmaReportGenerator:

    REPORT_PROMPT = """You are an expert radiologist. Analyze this chest X-ray and write a structured radiology report with these exact sections:

FINDINGS: Describe the lungs, heart, mediastinum, pleura, and bones in detail.
IMPRESSION: Summarize the most important findings in 1-3 sentences.
RECOMMENDATIONS: Suggest any follow-up if needed.

Be precise and use proper medical terminology."""

    def __init__(self, use_4bit=True):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print("MedGemma-compatible model ready (Groq API).")

    def generate_report(self, image_path: str, max_new_tokens=512) -> dict:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("Analyzing X-ray...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": self.REPORT_PROMPT}
                ]
            }],
            max_tokens=max_new_tokens,
        )

        raw = response.choices[0].message.content
        return self._parse(raw)

    def _parse(self, raw: str) -> dict:
        result = {"findings": "", "impression": "",
                  "recommendations": "", "raw_output": raw}
        for key, pattern in [
            ("findings",        r"FINDINGS:?(.*?)(?=IMPRESSION|RECOMMENDATIONS|$)"),
            ("impression",      r"IMPRESSION:?(.*?)(?=RECOMMENDATIONS|FINDINGS|$)"),
            ("recommendations", r"RECOMMENDATIONS:?(.*?)$"),
        ]:
            m = re.search(pattern, raw, re.I | re.S)
            if m:
                result[key] = m.group(1).strip()
        if not result["findings"]:
            result["findings"] = raw
        return result