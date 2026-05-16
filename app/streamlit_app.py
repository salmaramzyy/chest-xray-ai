import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import json
import glob
from dotenv import load_dotenv

load_dotenv()

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Chest X-Ray AI System")
st.write("DSAI 413 — Assignment 2 | Multi-Modal Report Generation & QA")
st.divider()

mode = st.radio("Select Mode:", ["Report Generation", "QA Mode"], horizontal=True)
st.divider()

if mode == "Report Generation":
    model_choice = st.selectbox("Select Model:", ["CLIP (Retrieval-based)", "MedGemma 4B"])

uploaded = st.file_uploader("Upload a chest X-ray image (PNG or JPG):", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_medgemma():
    from src.report_generation.medgemma_model import MedGemmaReportGenerator
    return MedGemmaReportGenerator(use_4bit=True)

@st.cache_resource
def load_clip():
    from src.report_generation.clip_model import CLIPReportGenerator
    with open("data/reports/knowledge_base.json") as f:
        reports = json.load(f)
    image_paths = sorted(glob.glob("data/raw/images/*.png"))
    report_texts = [r["findings"] + "\n" + r["impression"] for r in reports]
    all_texts = [report_texts[i % len(report_texts)] for i in range(len(image_paths))]
    gen = CLIPReportGenerator()
    gen.build_index(image_paths, all_texts)
    return gen

@st.cache_resource
def load_rag():
    from src.rag.lightweight_retriever import LightweightRetriever
    from src.rag.qa_pipeline import RAGQAPipeline
    with open("data/reports/knowledge_base.json") as f:
        reports = json.load(f)
    image_paths = sorted(glob.glob("data/raw/images/*.png"))
    docs = [{
        "image": image_paths[i % len(image_paths)],
        "text": f"FINDINGS: {r['findings']}\nIMPRESSION: {r['impression']}",
        "source": r["uid"],
    } for i, r in enumerate(reports)]
    retriever = LightweightRetriever()
    retriever.index_documents(docs)
    return RAGQAPipeline(retriever)

if uploaded:
    os.makedirs("data/processed", exist_ok=True)
    tmp_path = "data/processed/query_image.png"
    image = Image.open(uploaded).convert("RGB")
    image.save(tmp_path)

    st.image(image, caption="Uploaded X-Ray", width=300)
    st.divider()

    if mode == "Report Generation":
        if st.button("Generate Report"):
            if model_choice == "CLIP (Retrieval-based)":
                with st.spinner("Finding similar cases with CLIP..."):
                    result = load_clip().generate_report(tmp_path)
            else:
                with st.spinner("MedGemma is analyzing the image... (may take a minute)"):
                    result = load_medgemma().generate_report(tmp_path)

            st.subheader("Findings")
            st.write(result.get("findings") or "—")

            st.subheader("Impression")
            st.write(result.get("impression") or "—")

            st.subheader("Recommendations")
            st.write(result.get("recommendations") or "—")

            if "retrieved_cases" in result:
                with st.expander("Show retrieved similar cases (CLIP)"):
                    for c in result["retrieved_cases"]:
                        st.write(f"Similarity: {c['similarity']:.3f}")
                        st.write(c["report"][:300])
                        st.divider()

    else:
        question = st.text_input("Ask a question about this X-ray:",
                                 placeholder="e.g. Is there pneumonia? Are the lungs clear?")
        if question and st.button("Get Answer"):
            with st.spinner("Retrieving context and generating answer..."):
                result = load_rag().answer(tmp_path, question)

            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("Show retrieved context (RAG sources)"):
                for doc in result["retrieved_context"]:
                    st.write(f"Source: {doc['source']} | Score: {doc['score']:.2f}")
                    st.write(doc["text"][:400])
                    st.divider()