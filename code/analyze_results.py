"""
analyze_results.py — Deep analysis visualizations for v1.5 (paper-quality figures)

Generates analysis plots BEYOND basic metrics:
  1. t-SNE of RoBERTa CLS embeddings colored by scam type (cluster structure)
  2. Per-class ROC curves (one-vs-rest) + macro-average AUC
  3. Error analysis: misclassified pairs + confidence of errors
  4. Tactic-probability heatmap per scam type (what tactics fire for each scam)
  5. Confidence distribution: correct vs incorrect predictions
  6. Layer 2: true vs predicted victim-profile distributions

Usage (after evaluate_test.py):
    python analyze_results.py \
        --dataset_dir ".../COVA-X_Split" \
        --layer1_dir models/layer1_multitask \
        --output_dir results_v1rerun
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import ID2LABEL, TACTICS, MODEL_NAME, MAX_LENGTH, SEED
from dataset_utils import load_split

# ---------------------------------------------------------------------------
def extract_embeddings(test_df, layer1_dir, device, model_name, max_samples=1500, batch_size=32):
    """Run the trained encoder and collect CLS embeddings + predictions."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(layer1_dir + "/tokenizer")
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    idx = np.random.RandomState(SEED).choice(len(test_df), min(max_samples, len(test_df)), replace=False)
    sub = test_df.iloc[idx].reset_index(drop=True)
    texts = sub["text"].tolist()

    embs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                            padding="max_length", return_tensors="pt").to(device)
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cls = out.last_hidden_state[:, 0]                      # [B, H]
            embs.append(cls.cpu().numpy())
    embs = np.vstack(embs)
    return sub, embs

def fig_tsne(sub, embs, outdir):
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca", learning_rate="auto")
    emb2d = tsne.fit_transform(embs)

    labels = sorted(set(sub["label"].tolist()))
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(11, 8))
    for li, lab in enumerate(labels):
        mask = (sub["label"].values == lab)
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], s=14, alpha=0.75,
                   color=cmap(li % 10), label=ID2LABEL[lab], edgecolors="none")
    ax.set_title("t-SNE of RoBERTa CLS Embeddings (test conversations, colored by TRUE scam type)")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, fontsize=9, loc="best", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(outdir, "analysis_tsne_clusters.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved -> {path}")

def fig_roc(records, outdir):
    y_true = np.array([r["true_scam"] for r in records])
    classes = [ID2LABEL[i] for i in range(8)]
    Y = label_binarize(y_true, classes=classes)
    probs = np.array([[r["scam_probs"][c] for c in classes] for r in records])

    fig, ax = plt.subplots(figsize=(8, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, lw=1.4, label=f"{c} (AUC={roc_auc:.3f})")
    mean_tpr = np.mean(tprs, axis=0)
    ax.plot(mean_fpr, mean_tpr, "k--", lw=2.2,
            label=f"macro-average (AUC={auc(mean_fpr, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Layer 1 ROC Curves (one-vs-rest, test set)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(outdir, "analysis_roc_curves.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved -> {path}")

def fig_error_analysis(records, outdir):
    errors = [r for r in records if not r["scam_correct"]]
    conf_err = [r["scam_probs"][r["pred_scam"]] for r in errors]
    conf_ok = [r["scam_probs"][r["true_scam"]] for r in records if r["scam_correct"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: confidence of correct vs incorrect
    ax1.hist(conf_ok, bins=30, alpha=0.6, label=f"correct (n={len(conf_ok)})", color="seagreen")
    ax1.hist(conf_err, bins=30, alpha=0.7, label=f"incorrect (n={len(conf_err)})", color="tomato")
    ax1.set_xlabel("Model confidence (probability assigned to predicted class)")
    ax1.set_ylabel("# conversations")
    ax1.set_title("Confidence: correct vs misclassified")
    ax1.legend()

    # right: top misclassification pairs
    from collections import Counter
    pairs = Counter((r["true_scam"], r["pred_scam"]) for r in errors)
    top = pairs.most_common(10)
    if top:
        names = [f"{t[:14]}→{p[:14]}" for (t, p), _ in top]
        counts = [n for _, n in top]
        ax2.barh(range(len(top))[::-1], counts, color="tomato")
        ax2.set_yticks(range(len(top))[::-1]); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("# misclassifications")
    ax2.set_title("Top true→predicted error pairs")

    fig.tight_layout()
    path = os.path.join(outdir, "analysis_error_analysis.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved -> {path}")

def fig_tactic_heatmap(records, outdir):
    classes = [ID2LABEL[i] for i in range(8)]
    mat = np.zeros((8, len(TACTICS)))
    for r in records:
        i = classes.index(r["true_scam"])
        for j, t in enumerate(TACTICS):
            mat[i, j] += r["tactic_probs"][t]
    mat /= np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(mat.max(), 0.05))
    ax.set_xticks(range(len(TACTICS))); ax.set_xticklabels(TACTICS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(8)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted manipulation tactic"); ax.set_ylabel("True scam type")
    ax.set_title("Tactic-probability distribution per scam type (test set)")
    for i in range(8):
        for j in range(len(TACTICS)):
            if mat[i, j] > 0.02:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 0.5 else "black")
    fig.colorbar(im, label="avg probability")
    fig.tight_layout()
    path = os.path.join(outdir, "analysis_tactic_heatmap.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved -> {path}")

def fig_layer2_profiles(test_df, records, outdir):
    dims = [("trust_label", "Trust susceptibility"), ("tech_label", "Tech-savviness"),
            ("awareness_label", "Scam awareness")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title) in zip(axes, dims):
        true_counts = np.bincount(test_df[col].values, minlength=3)
        pred_counts = np.bincount(np.array([r["profile"][col] for r in records]), minlength=3)
        x = np.arange(3)
        ax.bar(x - 0.2, true_counts, 0.4, label="ground truth", color="steelblue")
        ax.bar(x + 0.2, pred_counts, 0.4, label="predicted", color="darkorange")
        ax.set_xticks(x); ax.set_xticklabels(["low", "moderate", "high"])
        ax.set_title(title); ax.legend(fontsize=8)
        ax.set_ylabel("# conversations")
    fig.suptitle("Layer 2: true vs predicted victim profiles (test set)")
    fig.tight_layout()
    path = os.path.join(outdir, "analysis_layer2_profiles.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved -> {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--layer1_dir", default="models/layer1_multitask")
    parser.add_argument("--predictions", default="results_v1rerun/test_predictions.jsonl")
    parser.add_argument("--output_dir", default="results_v1rerun")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_samples", type=int, default=1500)
    args = parser.parse_args()

    model_name = args.model_name or MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    records = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    test_df = load_split(args.dataset_dir, "test")
    print(f"Loaded {len(records)} predictions, {len(test_df)} test conversations")

    # 1) t-SNE clusters (needs the encoder)
    print("\n[1/6] Extracting embeddings + t-SNE...")
    sub, embs = extract_embeddings(test_df, args.layer1_dir, device, model_name, args.max_samples)
    fig_tsne(sub, embs, args.output_dir)

    # 2) ROC curves
    print("[2/6] ROC curves...")
    fig_roc(records, args.output_dir)

    # 3) Error analysis
    print("[3/6] Error analysis...")
    fig_error_analysis(records, args.output_dir)

    # 4) Tactic heatmap
    print("[4/6] Tactic heatmap...")
    fig_tactic_heatmap(records, args.output_dir)

    # 5) Layer 2 profiles
    print("[5/6] Layer 2 profile comparison...")
    fig_layer2_profiles(test_df, records, args.output_dir)

    print(f"\nAll analysis figures saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
