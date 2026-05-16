import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()

with open("data/reports/knowledge_base.json") as f:
    REPORTS = json.load(f)

REFERENCE_TEXTS = [r["findings"] + " " + r["impression"] for r in REPORTS]

def rouge_scores(hypothesis: str, references: list[str]) -> dict:
    """Score a generated text against multiple references, return best score."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    best = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for ref in references:
        s = scorer.score(ref, hypothesis)
        for k in best:
            if s[k].fmeasure > best[k]:
                best[k] = s[k].fmeasure
    return best


def evaluate_clip(image_paths: list) -> list:
    print("\nEvaluating CLIP")
    from src.report_generation.clip_model import CLIPReportGenerator

    gen = CLIPReportGenerator()
    report_texts = [r["findings"] + "\n" + r["impression"] for r in REPORTS]
    index_texts  = [report_texts[i % len(report_texts)] for i in range(len(image_paths))]
    gen.build_index(image_paths, index_texts)

    results = []
    for i, path in enumerate(image_paths[:5]):   # test on 5 images
        print(f"  Image {i+1}/5: {os.path.basename(path)}")
        t0 = time.time()
        out = gen.generate_report(path)
        elapsed = time.time() - t0

        text = (out["findings"] + " " + out["impression"]).strip()
        scores = rouge_scores(text, REFERENCE_TEXTS)
        results.append({
            "model":    "CLIP",
            "image":    os.path.basename(path),
            "time_s":   round(elapsed, 2),
            "rouge1":   round(scores["rouge1"], 4),
            "rouge2":   round(scores["rouge2"], 4),
            "rougeL":   round(scores["rougeL"], 4),
            "output":   text[:300],
        })
        print(f"    ROUGE-1={scores['rouge1']:.3f}  time={elapsed:.1f}s")
    return results

def evaluate_medgemma(image_paths: list) -> list:
    print("\nEvaluating MedGemma ")
    from src.report_generation.medgemma_model import MedGemmaReportGenerator

    gen = MedGemmaReportGenerator(use_4bit=True)
    results = []
    for i, path in enumerate(image_paths[:5]):
        print(f"  Image {i+1}/5: {os.path.basename(path)}")
        t0 = time.time()
        out = gen.generate_report(path)
        elapsed = time.time() - t0

        text = (out["findings"] + " " + out["impression"]).strip()
        scores = rouge_scores(text, REFERENCE_TEXTS)
        results.append({
            "model":    "MedGemma",
            "image":    os.path.basename(path),
            "time_s":   round(elapsed, 2),
            "rouge1":   round(scores["rouge1"], 4),
            "rouge2":   round(scores["rouge2"], 4),
            "rougeL":   round(scores["rougeL"], 4),
            "output":   text[:300],
        })
        print(f"    ROUGE-1={scores['rouge1']:.3f}  time={elapsed:.1f}s")
    return results

def plot_comparison(df: pd.DataFrame):
    os.makedirs("evaluation", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("MedGemma vs CLIP — Model Comparison", fontsize=14, fontweight="bold")

    metrics = ["rouge1", "rouge2", "rougeL"]
    titles  = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    colors  = {"CLIP": "#4C9BE8", "MedGemma": "#E87B4C"}

    for ax, metric, title in zip(axes, metrics, titles):
        for model, grp in df.groupby("model"):
            ax.bar(model, grp[metric].mean(), color=colors[model], width=0.4)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/comparison_rouge.png", dpi=150)
    print("\nPlot saved → evaluation/comparison_rouge.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for model, grp in df.groupby("model"):
        ax2.bar(model, grp["time_s"].mean(), color=colors[model], width=0.4)
    ax2.set_title("Average Inference Time (seconds)")
    ax2.set_ylabel("Seconds")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation/comparison_speed.png", dpi=150)
    print("Plot saved → evaluation/comparison_speed.png")

if __name__ == "__main__":
    image_paths = sorted(glob.glob("data/raw/images/*.png"))
    if not image_paths:
        print("No images found. Run download_data.py first.")
        sys.exit(1)

    all_results = []

    all_results += evaluate_clip(image_paths)

    run_medgemma = input("\nRun MedGemma evaluation? (y/n): ").strip().lower() == "y"
    if run_medgemma:
        all_results += evaluate_medgemma(image_paths)

    df = pd.DataFrame(all_results)
    df.to_csv("evaluation/results.csv", index=False)
    print("\nSummary")
    print(df.groupby("model")[["rouge1", "rouge2", "rougeL", "time_s"]].mean().round(4))

    plot_comparison(df)
    print("\nDone! Check evaluation/ folder for results.")