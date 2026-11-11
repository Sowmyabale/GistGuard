"""
train_layer1_multitask.py — Layer 1 RERUN (v1.5)

Multi-task DistilBERT:
    shared encoder -> scam-type head (8 classes)
                     -> manipulation-tactic head (6 classes)

Trains on the SAME COVA-X_Split folders used for the v2 zero-shot evaluation,
so v1.5 (supervised) vs v2 (zero-shot) results are directly comparable.

"""

import os
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
    LABEL_MAP, ID2LABEL, TACTICS, TACTIC_MAP, ID2TACTIC,
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    TACTIC_LOSS_WEIGHT, SEED, MODEL_NAME,
)
from dataset_utils import load_all_splits

# ---------------------------------------------------------------------------
# Model: shared encoder + two classification heads
# ---------------------------------------------------------------------------
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
        pooled = out.last_hidden_state[:, 0]          # CLS token
        pooled = self.dropout(pooled)
        scam_logits = self.scam_head(pooled)
        tactic_logits = self.tactic_head(pooled)

        loss = None
        if labels is not None and tactic_labels is not None:
            loss_scam = nn.functional.cross_entropy(scam_logits, labels)
            loss_tactic = nn.functional.cross_entropy(tactic_logits, tactic_labels)
            loss = loss_scam + TACTIC_LOSS_WEIGHT * loss_tactic

        return {
            "loss": loss,
            "scam_logits": scam_logits,
            "tactic_logits": tactic_logits,
        }

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            tactic_labels=inputs["tactic_labels"],
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        device = next(model.parameters()).device   # plain nn.Module has no .device attr
        inputs_cuda = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=inputs_cuda["input_ids"],
                attention_mask=inputs_cuda["attention_mask"],
            )
        loss = None
        scam_logits = outputs["scam_logits"]
        tactic_logits = outputs["tactic_logits"]
        labels = inputs_cuda.get("labels")
        tactic_labels = inputs_cuda.get("tactic_labels")
        if labels is not None and tactic_labels is not None:
            loss = nn.functional.cross_entropy(scam_logits, labels) \
                 + TACTIC_LOSS_WEIGHT * nn.functional.cross_entropy(tactic_logits, tactic_labels)
        return (loss, (scam_logits, tactic_logits), (labels, tactic_labels))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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

def tokenize_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["text", "label", "tactic_label"]].reset_index(drop=True))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    ds = ds.map(tok, batched=True, num_proc=4,
                remove_columns=[c for c in ds.column_names if c not in ("label", "tactic_label")])
    ds = ds.rename_column("label", "labels")           # Trainer convention
    ds = ds.rename_column("tactic_label", "tactic_labels")  # must match model.forward kwarg
    return ds

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name", default=None,
                        help="Encoder for Layer 1 (default: config.MODEL_NAME = roberta-base)")
    parser.add_argument("--output_dir", default="models/layer1_multitask")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- data ----
    train_df, val_df, test_df = load_all_splits(args.dataset_dir)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    model_name = args.model_name or MODEL_NAME
    print(f"Layer 1 encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)

    # ---- model ----
    model = MultiTaskScamClassifier(
        model_name, num_scam=len(LABEL_MAP), num_tactic=len(TACTIC_MAP)
    )
    if torch.cuda.is_available():
        model = model.cuda()

    targs = TrainingArguments(
        output_dir=args.output_dir + "/trainer",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=LEARNING_RATE,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="scam_f1_macro",
        greater_is_better=True,
        lr_scheduler_type="reduce_lr_on_plateau",   # lowers LR when val F1 plateaus
        lr_scheduler_kwargs={"factor": 0.5, "patience": 1},
        weight_decay=0.01,                          # L2 regularization against overfitting
        logging_steps=50,
        report_to=[],
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ---- save final artifacts ----
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "layer1_multitask.pt"))
    tokenizer.save_pretrained(args.output_dir + "/tokenizer")
    import json as _json
    with open(os.path.join(args.output_dir, "label_maps.json"), "w") as f:
        _json.dump({"label_map": LABEL_MAP, "tactic_map": TACTIC_MAP,
                    "scam_type_to_tactic": {
                        k: v for k, v in __import__("config").SCAM_TYPE_TO_TACTIC.items()}}, f, indent=2)
    print(f"\nSaved model -> {args.output_dir}")

    # ---- final validation report ----
    val_pred = trainer.predict(val_ds)
    (scam_logits, tactic_logits) = val_pred.predictions
    val_scam = np.argmax(scam_logits, axis=-1)
    val_tac = np.argmax(tactic_logits, axis=-1)
    print("\n=== VAL: SCAM TYPE ===")
    print(classification_report(val_df["label"], val_scam,
                                labels=list(range(len(ID2LABEL))),
                                target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4))
    print("=== VAL: TACTIC ===")
    print(classification_report(val_df["tactic_label"], val_tac,
                                labels=list(range(len(TACTICS))),
                                target_names=TACTICS, digits=4))

if __name__ == "__main__":
    main()
