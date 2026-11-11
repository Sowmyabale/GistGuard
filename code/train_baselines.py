"""
train_baselines.py — Classical ML baselines (B1–B3) for the Paper-1 comparison table.

Models (all 8-way scam-type classification, identical 60/20/20 split as v1.5):
    B1  TF-IDF + Logistic Regression
    B2  TF-IDF + Linear SVM
    B3  TF-IDF + Random Forest

CPU-only, fast (~minutes). Reproduces the classical-ML baseline family used in
Lochstampfor & Roy (2026, arXiv:2604.11752) so our numbers are comparable in kind.

"""

import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import ID2LABEL, SEED
from dataset_utils import load_all_splits

def make_pipelines():
    tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
    )
    return {
        "tfidf_logreg": Pipeline([
            ("tfidf", tfidf),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
        ]),
        "tfidf_svm": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2)),
            ("clf", LinearSVC(C=1.0)),
        ]),
        "tfidf_rf": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2)),
            ("clf", RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced")),
        ]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="results_baselines")
    args = parser.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, val_df, test_df = load_all_splits(args.dataset_dir)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    summary = {}

    for model_name, pipe in make_pipelines().items():
        print(f"\n=== {model_name} ===")
        pipe.fit(train_df["text"], train_df["label"])

        val_pred = pipe.predict(val_df["text"])
        test_pred = pipe.predict(test_df["text"])

        val_acc = accuracy_score(val_df["label"], val_pred)
        val_f1 = f1_score(val_df["label"], val_pred, average="macro")
        test_acc = accuracy_score(test_df["label"], test_pred)
        test_f1 = f1_score(test_df["label"], test_pred, average="macro")

        print(f"val  acc={val_acc:.4f}  macro-F1={val_f1:.4f}")
        print(f"test acc={test_acc:.4f}  macro-F1={test_f1:.4f}")
        print(classification_report(test_df["label"], test_pred,
                                    labels=list(range(len(names))),
                                    target_names=names, digits=4))

        cm = confusion_matrix(test_df["label"], test_pred)
        report = classification_report(test_df["label"], test_pred,
                                       labels=list(range(len(names))),
                                       target_names=names, output_dict=True)
        summary[model_name] = {
            "val_accuracy": round(val_acc, 4), "val_f1_macro": round(val_f1, 4),
            "test_accuracy": round(test_acc, 4), "test_f1_macro": round(test_f1, 4),
            "per_class": {k: {"precision": round(v["precision"], 4),
                              "recall": round(v["recall"], 4),
                              "f1": round(v["f1-score"], 4)}
                          for k, v in report.items() if k in names},
        }
        np.save(os.path.join(args.output_dir, f"cm_{model_name}.npy"), cm)

    with open(os.path.join(args.output_dir, "tfidf_baselines_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.output_dir}/tfidf_baselines_summary.json")

if __name__ == "__main__":
    main()
