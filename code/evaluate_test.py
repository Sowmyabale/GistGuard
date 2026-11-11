"""
evaluate_test.py — Full test-split evaluation for the v1 RERUN (v1.5)

Runs the trained Layer 1 (multi-task) and Layer 2 (BiLSTM profiler) over the
TEST split, prints metrics, saves figures, and writes a predictions JSONL that
feeds Layer 3 gist generation.

"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import (
    ID2LABEL, TACTICS, ID2TACTIC, MODEL_NAME,
    MAX_LENGTH, RESULTS_DIR, TRUST_NAMES, TECH_NAMES, AWARENESS_NAMES,
)
from dataset_utils import load_split
from train_layer1_multitask import MultiTaskScamClassifier
from train_layer2_profiler import BiLSTMProfiler, encode

def evaluate_layer1(test_df, layer1_dir, device, model_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(layer1_dir + "/tokenizer")
    model = MultiTaskScamClassifier(model_name, num_scam=8, num_tactic=6).to(device)
    model.load_state_dict(torch.load(os.path.join(layer1_dir, "layer1_multitask.pt"), map_location=device))
    model.eval()

    records = []
    scam_golds, scam_preds = [], []
    tac_golds, tac_preds = [], []

    texts = test_df["text"].tolist()
    bs = 16
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i + bs]
            enc = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH,
                            padding="max_length", return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            scam_probs = torch.softmax(out["scam_logits"], -1).cpu().numpy()
            tac_probs = torch.softmax(out["tactic_logits"], -1).cpu().numpy()
            for j in range(len(batch_texts)):
                idx = i + j
                row = test_df.iloc[idx]
                sp, tp = scam_probs[j], tac_probs[j]
                scam_pred, tac_pred = int(sp.argmax()), int(tp.argmax())
                scam_golds.append(int(row["label"])); scam_preds.append(scam_pred)
                tac_golds.append(int(row["tactic_label"])); tac_preds.append(tac_pred)
                records.append({
                    "conversation_id": row["conversation_id"],
                    "scam_type": row["scam_type"],
                    "true_scam": row["scam_type"],
                    "pred_scam": ID2LABEL[scam_pred],
                    "scam_correct": bool(scam_pred == int(row["label"])),
                    "scam_probs": {ID2LABEL[k]: float(sp[k]) for k in range(len(sp))},
                    "true_tactic": row["tactic"],
                    "pred_tactic": ID2TACTIC[tac_pred],
                    "tactic_probs": {ID2TACTIC[k]: float(tp[k]) for k in range(len(tp))},
                })

    print("\n=== LAYER 1 (multi-task) TEST RESULTS ===")
    acc = accuracy_score(scam_golds, scam_preds)
    f1 = f1_score(scam_golds, scam_preds, average="macro")
    print(f"Scam-type accuracy: {acc:.4f} | macro-F1: {f1:.4f}")
    scam_report = classification_report(scam_golds, scam_preds,
                                        target_names=[ID2LABEL[i] for i in range(8)], digits=4, output_dict=True)
    print(classification_report(scam_golds, scam_preds,
                                target_names=[ID2LABEL[i] for i in range(8)], digits=4))
    tac_acc = accuracy_score(tac_golds, tac_preds)
    print(f"Tactic accuracy: {tac_acc:.4f} (labels derived from scam-type mapping)")
    print(classification_report(tac_golds, tac_preds,
                                labels=list(range(len(TACTICS))),
                                target_names=TACTICS, digits=4))

    # ---------- figures ----------
    figures = {}

    # 1) scam-type confusion matrix
    cm = confusion_matrix(scam_golds, scam_preds, labels=list(range(8)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([ID2LABEL[i] for i in range(8)], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([ID2LABEL[i] for i in range(8)], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Layer 1 Test Confusion Matrix (v1.5 multi-task)")
    fig.colorbar(im)
    fig.tight_layout()
    figures["confusion_matrix_test"] = fig

    # 2) per-class accuracy / F1 bars
    labels8 = [ID2LABEL[i] for i in range(8)]
    p_acc = [scam_report[l]["recall"] for l in labels8]       # per-class recall == per-class accuracy
    p_f1 = [scam_report[l]["f1-score"] for l in labels8]
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, p_acc, 0.4, label="Per-class accuracy (recall)")
    ax.bar(x + 0.2, p_f1, 0.4, label="Per-class F1")
    ax.set_xticks(x); ax.set_xticklabels(labels8, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Layer 1 Per-Scam-Type Performance (test)")
    fig.tight_layout()
    figures["per_class_performance"] = fig

    # 3) tactic confusion matrix
    tcm = confusion_matrix(tac_golds, tac_preds, labels=list(range(len(TACTICS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(tcm, cmap="Greens")
    ax.set_xticks(range(len(TACTICS))); ax.set_yticks(range(len(TACTICS)))
    ax.set_xticklabels(TACTICS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(TACTICS, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Manipulation-Tactic Confusion Matrix (test)")
    fig.colorbar(im)
    fig.tight_layout()
    figures["tactic_confusion_matrix"] = fig

    # 4) confidence distribution (true-class predicted probability)
    true_conf = [records[i]["scam_probs"][records[i]["true_scam"]] for i in range(len(records))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(true_conf, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Predicted probability of TRUE scam type")
    ax.set_ylabel("# conversations")
    ax.set_title(f"Layer 1 Confidence Distribution (mean={np.mean(true_conf):.3f})")
    fig.tight_layout()
    figures["confidence_distribution"] = fig

    metrics = {"scam_accuracy": acc, "scam_f1_macro": f1, "tactic_accuracy": tac_acc}
    metrics["per_class"] = {l: {"recall": round(scam_report[l]["recall"], 4),
                                "f1": round(scam_report[l]["f1-score"], 4)} for l in labels8}
    metrics["mean_true_class_confidence"] = float(np.mean(true_conf))
    return records, metrics, figures

def evaluate_layer2(test_df, layer2_dir, device):
    ckpt = torch.load(os.path.join(layer2_dir, "layer2_profiler.pt"), map_location=device)
    vocab = ckpt["vocab"]
    model = BiLSTMProfiler(len(vocab)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = {"trust": [], "tech": [], "aware": []}
    golds = {"trust": [], "tech": [], "aware": []}
    encs = [encode(t, vocab) for t in test_df["victim_text"]]

    with torch.no_grad():
        for i in range(len(encs)):
            x = torch.tensor([encs[i]], dtype=torch.long).to(device)
            lens = torch.tensor([len(encs[i])])
            pt, pte, pa = model(x, lens)   # x on GPU, lens stays CPU (pack_padded_sequence requirement)
            preds["trust"].append(int(pt.argmax(-1).cpu()))
            preds["tech"].append(int(pte.argmax(-1).cpu()))
            preds["aware"].append(int(pa.argmax(-1).cpu()))
            golds["trust"].append(int(test_df.iloc[i]["trust_label"]))
            golds["tech"].append(int(test_df.iloc[i]["tech_label"]))
            golds["aware"].append(int(test_df.iloc[i]["awareness_label"]))

    print("\n=== LAYER 2 (ToM / susceptibility) TEST RESULTS ===")
    metrics = {}
    for key, names in [("trust", TRUST_NAMES), ("tech", TECH_NAMES), ("aware", AWARENESS_NAMES)]:
        acc = accuracy_score(golds[key], preds[key])
        f1 = f1_score(golds[key], preds[key], average="macro")
        metrics[f"{key}_acc"] = acc
        metrics[f"{key}_f1"] = f1
        print(f"{key:6s}: acc={acc:.4f} macro-F1={f1:.4f}")
        print(classification_report(golds[key], preds[key], target_names=names, digits=4))

    # Layer 2 figure: per-dimension acc/F1 bars + predicted profile distribution
    dims = ["Trust susceptibility", "Tech-savviness", "Scam awareness"]
    accs = [metrics["trust_acc"], metrics["tech_acc"], metrics["aware_acc"]]
    f1s = [metrics["trust_f1"], metrics["tech_f1"], metrics["aware_f1"]]
    x = np.arange(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x - 0.2, accs, 0.4, label="Accuracy")
    ax1.bar(x + 0.2, f1s, 0.4, label="Macro-F1")
    ax1.set_xticks(x); ax1.set_xticklabels(dims, fontsize=9)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Score"); ax1.legend()
    ax1.set_title("Layer 2 Per-Dimension Performance (test)")
    for key, title in [("trust", "Predicted trust distribution"),
                       ("tech", "Predicted tech-savviness"),
                       ("aware", "Predicted scam awareness")]:
        counts = np.bincount(preds[key], minlength=3)
        ax2.plot(["low", "moderate", "high"], counts, marker="o", label=title.split()[1])
    ax2.set_title("Layer 2 Predicted Profile Distribution")
    ax2.set_ylabel("# conversations"); ax2.legend(fontsize=8)
    fig.tight_layout()
    figures2 = {"layer2_performance": fig}

    # attach predictions to records by conversation order
    profiles = []
    for i in range(len(encs)):
        profiles.append({
            "trust_label": preds["trust"][i],
            "tech_label": preds["tech"][i],
            "awareness_label": preds["aware"][i],
        })
    return profiles, metrics, figures2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--layer1_dir", default="models/layer1_multitask")
    parser.add_argument("--layer2_dir", default="models/layer2_profiler")
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    test_df = load_split(args.dataset_dir, "test")
    print(f"Test conversations: {len(test_df)}")

    records, l1_metrics, l1_figs = evaluate_layer1(test_df, args.layer1_dir, device, args.model_name)
    profiles, l2_metrics, l2_figs = evaluate_layer2(test_df, args.layer2_dir, device)

    for rec, prof in zip(records, profiles):
        rec["profile"] = prof

    for name, fig in {**l1_figs, **l2_figs}.items():
        path = os.path.join(args.output_dir, f"{name}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved -> {path}")
    with open(os.path.join(args.output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump({"layer1": l1_metrics, "layer2": l2_metrics, "n_test": len(test_df)}, f, indent=2)

    print(f"\nSaved -> {args.output_dir}/test_predictions.jsonl")
    print(f"Saved -> {args.output_dir}/test_metrics.json")

if __name__ == "__main__":
    main()
