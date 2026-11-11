"""
train_layer1_bert.py
Layer 1: BERT/RoBERTa Fine-Tuning for Scam Type Classification
Run on: Google Colab (free T4 GPU) or university GPU cluster
"""

# ============================================================================
# STEP 0: Install dependencies (run once in Colab)
# ============================================================================
# !pip install transformers datasets accelerate scikit-learn pandas numpy matplotlib

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURATION
# ============================================================================
# Choose model: 'distilbert-base-uncased' for Colab, 'roberta-base' if you have more GPU
MODEL_NAME = 'distilbert-base-uncased'  # ~66M params, fits on Colab T4
# MODEL_NAME = 'roberta-base'  # ~125M params, better accuracy

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
OUTPUT_DIR = './layer1_bert_results'

# Scam type labels
LABEL_MAP = {
    'bank': 0,
    'government_impersonation': 1,
    'grandparent': 2,
    'investment': 3,
    'lottery': 4,
    'medicare': 5,
    'romance': 6,
    'virtual_kidnapping': 7
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ============================================================================
# STEP 1: Load Full COVA-X Dataset
# ============================================================================
def load_cova_x_full(base_dir="dataset/COVA-X Dataset/COVA-X_Dataset"):
    """Load ALL COVA-X conversations (10,985 files)."""
    rows = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                
                # Concatenate all turns into single text
                turns = conv.get('turns', [])
                texts = []
                for turn in turns:
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    texts.append(f"{role}: {content}")
                full_text = '\n'.join(texts)
                
                rows.append({
                    'text': full_text,
                    'label': LABEL_MAP.get(conv.get('scam_type'), 0),
                    'scam_type': conv.get('scam_type'),
                    'conversation_id': conv.get('conversation_id'),
                    'num_turns': len(turns)
                })
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    
    return pd.DataFrame(rows)

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================
def prepare_data():
    print("Loading full COVA-X dataset (this may take 1-2 minutes)...")
    df = load_cova_x_full()
    print(f"Loaded {len(df)} conversations")
    print(f"Label distribution:\n{df['scam_type'].value_counts()}")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])
    test_ds = Dataset.from_pandas(test_df[['text', 'label']])
    
    return train_ds, val_ds, test_ds, test_df

# ============================================================================
# STEP 3: Tokenize
# ============================================================================
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    return dataset.map(tokenize_function, batched=True)

# ============================================================================
# STEP 4: Metrics
# ============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# ============================================================================
# STEP 5: Training
# ============================================================================
def train():
    # Load data
    train_ds, val_ds, test_ds, test_df = prepare_data()
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=8,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    # Tokenize
    print("Tokenizing...")
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)
    
    # Set format for PyTorch
    train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU
        report_to='none',  # Disable wandb/tensorboard for simplicity
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_ds)
    print(f"Test Results: {results}")
    
    # Save model
    trainer.save_model(f'{OUTPUT_DIR}/best_model')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/best_model')
    print(f"\nModel saved to {OUTPUT_DIR}/best_model")
    
    # Detailed per-class report
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    print("\nPer-class classification report:")
    print(classification_report(test_ds['label'], preds, target_names=list(LABEL_MAP.keys()), digits=3))
    
    return model, tokenizer

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, tokenizer = train()
    print("\nLayer 1 training complete!")
