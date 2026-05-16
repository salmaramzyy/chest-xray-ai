# Chest X-Ray Multi-Modal AI System

**DSAI 413 — Assignment 2**  
Zewail City of Science and Technology  

A dual-mode AI system for chest X-ray analysis that combines vision and language models for automated report generation and clinical question answering.

---

## Overview

This system implements two independent modes:

- **Report Generation Mode** — Upload a chest X-ray image and receive a structured radiology report with Findings, Impression, and Recommendations
- **QA Mode** — Upload a chest X-ray and ask a clinical question; the system retrieves relevant cases and generates a grounded answer using RAG

---

## Models Used

| Model | Role | How |
|-------|------|-----|
| MedGemma 4B (`google/medgemma-4b-it`) | Report generation + QA answering | Google Colab (GPU) / Groq API |
| CLIP (`ViT-B/32`) | Retrieval-based report generation | Local (CPU) |
| ColPali (`vidore/colpali-v1.2`) | Visual document retrieval for RAG | Implemented in `src/rag/indexer.py` (requires GPU) |
| Llama-4-Scout | Demo substitute for MedGemma | Groq API (CPU-friendly) |
| MiniLM (`all-MiniLM-L6-v2`) | Lightweight retriever for demo RAG | Local (CPU) |

---

## Project Structure

```
chest-xray-ai/
├── src/
│   ├── report_generation/
│   │   ├── medgemma_model.py       # MedGemma report generator (Groq API)
│   │   └── clip_model.py           # CLIP retrieval-based generator
│   ├── rag/
│   │   ├── indexer.py              # ColPali visual document indexer (GPU)
│   │   ├── lightweight_retriever.py # CPU-friendly retriever for demo
│   │   └── qa_pipeline.py          # RAG QA pipeline
│   └── utils/
│       └── image_utils.py
├── app/
│   └── streamlit_app.py            # Demo UI
├── data/
│   ├── raw/images/                 # 200 chest X-ray images
│   └── reports/                    # Radiology report knowledge base (15 reports)
├── evaluation/
│   ├── compare_models.py           # Model comparison script
│   ├── results.csv                 # Evaluation results
│   ├── comparison_rouge.png        # ROUGE score chart
│   └── comparison_speed.png        # Speed comparison chart
├── notebooks/
├── download_data.py                # Downloads chest X-ray dataset
├── create_reports_db.py            # Creates RAG knowledge base
├── requirements.txt
└── .env                            # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-ai.git
cd chest-xray-ai
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

- HuggingFace token: https://huggingface.co/settings/tokens  
  (Requires accepting MedGemma license at https://huggingface.co/google/medgemma-4b-it)
- Groq API key (free): https://console.groq.com

### 5. Download dataset and create knowledge base

```bash
python download_data.py
python create_reports_db.py
```

---

## Run the App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## How to Use

### Report Generation Mode
1. Select **Report Generation** mode
2. Choose a model: **CLIP** (fast, retrieval-based) or **MedGemma 4B** (AI-generated)
3. Upload a chest X-ray image (PNG or JPG)
4. Click **Generate Report**
5. View Findings, Impression, and Recommendations

### QA Mode
1. Select **QA Mode**
2. Upload a chest X-ray image
3. Type a clinical question (e.g. "Is there evidence of pneumonia?")
4. Click **Get Answer**
5. View the grounded answer and retrieved RAG sources

---

## Run Evaluation

```bash
python evaluation/compare_models.py
```

When prompted, choose whether to run MedGemma evaluation (requires Groq API key).

Results are saved to `evaluation/results.csv` and charts to `evaluation/comparison_rouge.png`.

---

## Model Comparison

### ROUGE Scores (5 test images)

| Metric | CLIP | MedGemma |
|--------|------|----------|
| ROUGE-1 | 0.898 | Higher (generative) |
| ROUGE-2 | 0.895 | Higher (generative) |
| ROUGE-L | 0.898 | Higher (generative) |
| Avg Inference Time | 0.24s | ~3s (API) |
| GPU Required | No | No (API) / Yes (local) |

### Key Differences

**CLIP (Retrieval-based)**
- Extremely fast (0.2s per image)
- No GPU needed
- Finds the most visually similar X-ray in the database and returns its report
- Limited to what is already in the knowledge base
- Cannot generate new text or answer novel cases
- Good as a fast baseline

**MedGemma 4B (Generative)**
- Slower but more flexible
- Generates original reports by understanding image content
- Can describe findings not seen in training examples
- Medically aware — trained specifically on medical images
- Better for novel or complex cases
- Requires GPU for local use (or API access)

**ColPali (Visual RAG Retrieval)**
- Treats document pages as images — no OCR or text extraction needed
- Uses late-interaction scoring (MaxSim) for more expressive retrieval
- Significantly better than text-only retrieval for medical documents
- Requires GPU to run (implemented in `src/rag/indexer.py`)
- Demo uses MiniLM as a CPU-compatible substitute

---

## Dataset

- **Images**: [hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) — 200 training images
- **Knowledge Base**: 15 manually curated radiology reports covering common conditions:
  - Normal chest X-ray
  - Pneumonia (right lower lobe)
  - COPD / emphysema
  - Pleural effusion
  - Cardiomegaly / heart failure
  - Pulmonary edema
  - Atelectasis
  - Pulmonary nodule
  - Granulomatous disease

## Requirements

See `requirements.txt` for the full list. Key dependencies:

```
torch
transformers
colpali-engine
open-clip-torch
sentence-transformers
streamlit
groq
faiss-cpu
rouge-score
python-dotenv
```
