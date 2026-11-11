"""
eval_v3_tactics.py — Tactic-level evaluation of the V3 hybrid features.

Reuses the SAME hybrid feature pipeline as train_v3_hybrid.py (MiniLM embeddings +
tactic-similarity + linguistic markers) but predicts the 6-class MANIPULATION
TACTIC label instead of the 8-class scam type. Completes the v3 ablation:
both task rows (scam type + tactic) become measurable for the paper.

Note on gold labels: COVA-X ships no per-turn tactic annotations, so tactic
gold = scam-type -> dominant-tactic mapping (same documented heuristic used
everywhere in this project).
"""

import os
import json
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import TACTICS, SEED
from dataset_utils import load_all_splits
from train_v3_hybrid import encode_features, TACTIC_LEXICONS  # reuse

def zero_shot_tactic_by_similarity(test_feats, n_emb: int, test_labels):
    """Pure zero-shot: argmax over the 6 tactic-similarity features."""
    sims = test_feats[:, n_emb:n_emb + len(TACTICS)]
    pred = np.argmax(sims, axis=1)
    return (accuracy_score(test_labels, pred),
            f1_score(test_labels, pred, average="macro"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", default="results_v3hybrid_tactics")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, val_df, test_df = load_all_splits(args.dataset_dir)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    feats = encode_features([train_df, val_df, test_df], args.encoder, args.batch_size)
    Xtr, Xva, Xte = feats
    # tactic labels (heuristic gold via scam-type mapping)
    ytr = train_df["tactic_label"].values
    yva = val_df["tactic_label"].values
    yte = test_df["tactic_label"].values
    n_lex = sum(len(v) for v in TACTIC_LEXICONS.values())
    n_emb = Xtr.shape[1] - len(TACTICS) - n_lex

    summary = {}

    # ---- pure zero-shot reference ----
    zacc, zf1 = zero_shot_tactic_by_similarity(Xte, n_emb, yte)
    print(f"\n[zero-shot tactic similarity] acc={zacc:.4f}  macro-F1={zf1:.4f}")
    summary["zero_shot_tactic_similarity"] = {"test_accuracy": round(zacc, 4),
                                              "test_f1_macro": round(zf1, 4)}

    models = {
        "hybrid_logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
        ]),
        "hybrid_rf": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced"),
        "hybrid_mlp": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                                  random_state=SEED, early_stopping=True)),
        ]),
    }

    for mname, model in models.items():
        print(f"\n=== {mname} (TACTIC) ===")
        model.fit(Xtr, ytr)
        va_pred = model.predict(Xva)
        te_pred = model.predict(Xte)
        va_acc = accuracy_score(yva, va_pred)
        te_acc = accuracy_score(yte, te_pred)
        te_f1 = f1_score(yte, te_pred, average="macro")
        print(f"val  acc={va_acc:.4f}")
        print(f"test acc={te_acc:.4f}  macro-F1={te_f1:.4f}")
        print(classification_report(yte, te_pred, labels=list(range(len(TACTICS))),
                                    target_names=TACTICS, digits=4))
        cm = confusion_matrix(yte, te_pred)
        np.save(os.path.join(args.output_dir, f"cm_tactic_{mname}.npy"), cm)
        summary[mname] = {"val_accuracy": round(va_acc, 4),
                          "test_accuracy": round(te_acc, 4),
                          "test_f1_macro": round(te_f1, 4)}

    with open(os.path.join(args.output_dir, "v3_tactics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.output_dir}/v3_tactics_summary.json")

if __name__ == "__main__":
    main()
