"""
train_layer1_tactic_v2.py — Layer 1 tactic head RETRAINED on independently
LLM-annotated per-conversation tactics (reviewer point 1).

The original multitask model derives tactic labels from the scam category
(SCAM_TYPE_TO_TACTIC). This run replaces them with the dominant_tactic field
from tactic_annotations.jsonl (annotate_tactics_llm.py, 7 categories incl.
"none"), trains the same multi-task architecture (8-class scam head unchanged
in structure; 7-class tactic head), and evaluates top-1/2/3 agreement with the
LLM annotations on the held-out test split.

Usage:
    python train_layer1_tactic_v2.py \
        --dataset_dir "/path/to/COVA-X_Split" \
        --annotations results_v1rerun/tactic_annotations.jsonl \
        --output_dir models/layer1_multitask_tacticv2
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from datasets import Dataset

from config import (
    LABEL_MAP, ID2LABEL, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, TACTIC_LOSS_WEIGHT, SEED, MODEL_NAME,
)
from dataset_utils import load_all_splits

# 7-class annotation scheme (first 6 keep the original index order; none=6)
TACTICS7 = ["authority_impersonation", "fear_threat", "isolation_secrecy",
            "over_familiarity", "reciprocity_gift", "social_proof", "none"]
TACTIC7_MAP = {t: i for i, t in enumerate(TACTICS7)}


class MultiTaskScamClassifier(nn.Module):
    def __init__(self, model_name: str, num_scam: int, num_tactic: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.scam_head = nn.Linear(hidden, num_scam)
        self.tactic_head = nn.Linear(hidden, num_tactic)

    def forward(self, input_ids, attention_mask, labels=None, tactic_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        scam_logits = self.scam_head(pooled)
        tactic_logits = self.tactic_head(pooled)
        loss = None
        if labels is not None and tactic_labels is not None:
            loss_scam = nn.functional.cross_entropy(scam_logits, labels)
            loss_tactic = nn.functional.cross_entropy(tactic_logits, tactic_labels)
            loss = loss_scam + TACTIC_LOSS_WEIGHT * loss_tactic
        return {"loss": loss, "scam_logits": scam_logits,
                "tactic_logits": tactic_logits}


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"],
                        tactic_labels=inputs["tactic_labels"])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        device = next(model.parameters()).device
        inputs_cuda = {k: v.to(device) for k, v in inputs.items()
                       if torch.is_tensor(v)}
        with torch.no_grad():
            outputs = model(**inputs_cuda)
        loss = outputs["loss"]
        return (loss, (outputs["scam_logits"], outputs["tactic_logits"]),
                (inputs_cuda["labels"], inputs_cuda["tactic_labels"]))


def compute_metrics(eval_pred):
    (scam_logits, tactic_logits), (labels, tactic_labels) = eval_pred
    scam_pred = np.argmax(scam_logits, axis=-1)
    tactic_pred = np.argmax(tactic_logits, axis=-1)
    return {
        "scam_accuracy": accuracy_score(labels, scam_pred),
        "scam_f1_macro": f1_score(labels, scam_pred, average="macro"),
        "tactic_accuracy": accuracy_score(tactic_labels, tactic_pred),
        "tactic_f1_macro": f1_score(tactic_labels, tactic_pred, average="macro"),
    }


def topk_acc(probs: np.ndarray, gold: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([g in topk[i] for i, g in enumerate(gold)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--annotations", required=True,
                    help="tactic_annotations.jsonl (dominant_tactic per conversation)")
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--output_dir", default="models/layer1_multitask_tacticv2")
    ap.add_argument("--results_dir", default="results_v1rerun")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    anno = {}
    for line in open(args.annotations, encoding="utf-8"):
        r = json.loads(line)
        anno[r["conversation_id"]] = r.get("dominant_tactic", "none")
    print(f"Annotations loaded: {len(anno)} conversations")

    train_df, val_df, test_df = load_all_splits(args.dataset_dir)

    def apply_annotations(df, name):
        df = df.copy()
        df["tactic"] = df["conversation_id"].map(lambda c: anno.get(c, "none"))
        df["tactic_label"] = df["tactic"].map(lambda t: TACTIC7_MAP.get(t, TACTIC7_MAP["none"]))
        missing = int((~df["conversation_id"].isin(anno)).sum())
        print(f"[{name}] n={len(df)} | without annotation: {missing} -> 'none'")
        return df

    train_df = apply_annotations(train_df, "train")
    val_df = apply_annotations(val_df, "val")
    test_df = apply_annotations(test_df, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def encode(df):
        enc = tokenizer(df["text"].tolist(), truncation=True,
                        max_length=MAX_LENGTH, padding="max_length")
        enc["labels"] = df["label"].tolist()
        enc["tactic_labels"] = df["tactic_label"].tolist()
        return Dataset.from_dict(enc)

    train_ds, val_ds = encode(train_df), encode(val_df)

    model = MultiTaskScamClassifier(args.model_name,
                                    num_scam=len(ID2LABEL), num_tactic=len(TACTICS7))

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="tactic_accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=SEED,
    )
    trainer = MultiTaskTrainer(model=model, args=targs,
                               train_dataset=train_ds, eval_dataset=val_ds,
                               compute_metrics=compute_metrics,
                               callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved -> {args.output_dir}")

    # ---- test evaluation: top-1/2/3 vs independently annotated dominant_tactic ----
    model.eval()
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True,
                         max_length=MAX_LENGTH, padding="max_length",
                         return_tensors="pt")
    scam_logits_all, tactic_probs_all = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    B = args.batch_size
    with torch.no_grad():
        for i in range(0, len(test_df), B):
            ids = test_enc["input_ids"][i:i + B].to(device)
            mask = test_enc["attention_mask"][i:i + B].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            scam_logits_all.append(out["scam_logits"].cpu().numpy())
            tactic_probs_all.append(torch.softmax(out["tactic_logits"], dim=-1).cpu().numpy())
    scam_logits_all = np.concatenate(scam_logits_all)
    tactic_probs_all = np.concatenate(tactic_probs_all)

    gold_scam = test_df["label"].to_numpy()
    gold_tac = test_df["tactic_label"].to_numpy()
    scam_pred = scam_logits_all.argmax(-1)

    summary = {
        "scam_accuracy": float(accuracy_score(gold_scam, scam_pred)),
        "scam_f1_macro": float(f1_score(gold_scam, scam_pred, average="macro")),
        "tactic_vs_llm_annotations": {
            "top1": topk_acc(tactic_probs_all, gold_tac, 1),
            "top2": topk_acc(tactic_probs_all, gold_tac, 2),
            "top3": topk_acc(tactic_probs_all, gold_tac, 3),
        },
        "n_test": int(len(test_df)),
        "annotation_distribution": test_df["tactic"].value_counts().to_dict(),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "tactic_v2_summary.json")
    json.dump(summary, open(out_path, "w"), indent=2)

    print("\n=== TACTIC-V2 TEST RESULTS (vs LLM-annotated dominant tactic) ===")
    print(f"scam acc={summary['scam_accuracy']:.4f} macro-F1={summary['scam_f1_macro']:.4f}")
    for k in (1, 2, 3):
        print(f"tactic top-{k}: {summary['tactic_vs_llm_annotations']['top'+str(k)]*100:.2f}%")
    print(classification_report(gold_tac, tactic_probs_all.argmax(-1),
                                target_names=TACTICS7, digits=4))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
