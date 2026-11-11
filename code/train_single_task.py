"""
train_single_task.py — Single-task encoder baselines (B4, B5).

    B4  DistilBERT single-task  (scam-type head only)  — isolates encoder effect
    B5  RoBERTa single-task     (scam-type head only)  — isolates the multi-task
                                                        objective of v1.5

Same data, split, seed, epochs and schedule as train_layer1_multitask.py so the

"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback,
)
from datasets import Dataset

from config import LABEL_MAP, ID2LABEL, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED
from dataset_utils import load_all_splits

class SingleTaskClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

class SingleTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        device = next(model.parameters()).device
        inputs_cuda = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(input_ids=inputs_cuda["input_ids"],
                            attention_mask=inputs_cuda["attention_mask"])
        loss = None
        logits = outputs["logits"]
        labels = inputs_cuda.get("labels")
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return (loss, logits, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1_macro": f1_score(labels, pred, average="macro"),
    }

def tokenize_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    ds = ds.map(tok, batched=True, num_proc=4,
                remove_columns=[c for c in ds.column_names if c != "label"])
    return ds.rename_column("label", "labels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name", required=True,
                        help="e.g. distilbert-base-uncased or roberta-base")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_all_splits(args.dataset_dir)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    tag = os.path.basename(args.output_dir.rstrip("/"))
    print(f"Single-task encoder: {args.model_name}  [{tag}]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)

    model = SingleTaskClassifier(args.model_name, num_classes=len(LABEL_MAP))
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={"factor": 0.5, "patience": 1},
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = SingleTaskTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "single_task.pt"))
    tokenizer.save_pretrained(args.output_dir + "/tokenizer")
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # ---- test-set evaluation (the number that goes in the paper table) ----
    test_ds = tokenize_dataset(test_df, tokenizer)
    preds = trainer.predict(test_ds)
    test_pred = np.argmax(preds.predictions, axis=-1)
    test_acc = accuracy_score(test_df["label"], test_pred)
    test_f1 = f1_score(test_df["label"], test_pred, average="macro")

    print(f"\n=== TEST [{tag}] ===")
    print(f"accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print(classification_report(test_df["label"], test_pred,
                                labels=list(range(len(ID2LABEL))),
                                target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
                                digits=4))

    summary = {
        "model": args.model_name, "tag": tag,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
    }
    with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {args.output_dir}/test_summary.json")

if __name__ == "__main__":
    main()
