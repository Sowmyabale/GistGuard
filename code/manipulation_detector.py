"""
manipulation_detector.py
Multi-task Manipulation Tactic Detection Head for Layer 1 BERT
Detects 6 persuasion tactics commonly used in scams targeting elderly:
    - urgency: Time pressure ("act now", "limited time")
    - authority_claim: Pretending to be official ("IRS", "bank security")
    - fear_appeal: Threats or scare tactics ("arrest", "account frozen")
    - social_proof: Fake testimonials or social validation
    - false_urgency: Fake deadlines or scarcity
    - personal_info_request: Direct requests for SSN, card, password

Usage:
    python manipulation_detector.py --train  # Train multi-task BERT
    python manipulation_detector.py --predict "conversation_text.txt"
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================
BERT_PATH = './layer1_bert_results/best_model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
OUTPUT_DIR = './manipulation_detector_results'

# Manipulation tactic labels (multi-label: each can be 0 or 1)
TACTIC_LABELS = [
    'urgency',
    'authority_claim',
    'fear_appeal',
    'social_proof',
    'false_urgency',
    'personal_info_request'
]

# Keyword bootstrapping patterns for initial labeling
TACTIC_PATTERNS = {
    'urgency': [
        r'\b(act now|right now|immediately|today only|before it\'s too late|urgent|hurry|quickly|asap)\b',
        r'\b(limited time|expires?|deadline|within \d+ (hour|minute|day)s?)\b',
    ],
    'authority_claim': [
        r'\b(irs|fbi|social security administration|medicare|bank security|federal|government agency|official)\b',
        r'\b(this is .* (officer|agent|supervisor|manager|director))\b',
        r'\b(we are calling from|this is .* calling from)\b',
    ],
    'fear_appeal': [
        r'\b(arrest|warrant|jail|prison|legal action|lawsuit|prosecution)\b',
        r'\b(account (frozen|suspended|locked|closed|terminated))\b',
        r'\b(you will lose|you are at risk|your .* will be|penalty|fine)\b',
        r'\b(don\'t tell|don\'t mention|keep this confidential)\b',
    ],
    'social_proof': [
        r'\b(other people|many customers|everyone|thousands|millions|people like you)\b',
        r'\b(already (won|received|claimed|approved))\b',
        r'\b(your neighbor|your friend|someone in your area)\b',
    ],
    'false_urgency': [
        r'\b(only \d+ left|last chance|final notice|final warning)\b',
        r'\b(your (package|refund|prize) expires)\b',
    ],
    'personal_info_request': [
        r'\b(social security number|ssn|credit card|debit card|bank account|routing number)\b',
        r'\b(password|pin|cvv|security code|date of birth|mother\'s maiden name)\b',
        r'\b(verify your identity|confirm your|need your)\b',
    ]
}

# ============================================================================
# STEP 1: Bootstrap Labels from COVA-X using Regex Patterns
# ============================================================================
def extract_tactics_from_text(text):
    """Label a conversation text with manipulation tactics using regex patterns."""
    text_lower = text.lower()
    labels = {}
    for tactic, patterns in TACTIC_PATTERNS.items():
        matched = False
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched = True
                break
        labels[tactic] = 1 if matched else 0
    return labels

def load_cova_x_with_tactics(base_dir="dataset/COVA-X Dataset/COVA-X_Dataset"):
    """Load COVA-X and bootstrap manipulation tactic labels."""
    rows = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                
                turns = conv.get('turns', [])
                texts = []
                for turn in turns:
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    texts.append(f"{role}: {content}")
                full_text = '\n'.join(texts)
                
                # Extract attacker turns only (scammers use manipulation tactics)
                attacker_text = '\n'.join([
                    t.get('content', '') for t in turns if t.get('role') == 'attacker'
                ])
                
                # Bootstrap labels from attacker text
                tactics = extract_tactics_from_text(attacker_text)
                
                rows.append({
                    'conversation_id': conv.get('conversation_id'),
                    'scam_type': conv.get('scam_type'),
                    'full_text': full_text,
                    'attacker_text': attacker_text,
                    **tactics
                })
            except Exception as e:
                pass
    
    return pd.DataFrame(rows)

# ============================================================================
# STEP 2: Multi-Task BERT Model
# ============================================================================
class ManipulationBERT(nn.Module):
    """BERT backbone + separate classifier heads for each manipulation tactic."""
    
    def __init__(self, bert_path, num_tactics=6, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        
        # One linear head per tactic (binary classification each)
        self.tactic_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_tactics)
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        pooled = self.dropout(pooled)
        
        # Predict each tactic independently
        logits = []
        for classifier in self.tactic_classifiers:
            logits.append(classifier(pooled))  # (batch, 1)
        
        # Stack to (batch, num_tactics)
        return torch.cat(logits, dim=1)

class TacticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(label)
        }

# ============================================================================
# STEP 3: Training
# ============================================================================
def train_manipulation_detector():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("Loading COVA-X with tactic labels...")
    df = load_cova_x_with_tactics()
    print(f"Loaded {len(df)} conversations")
    
    # Show label distribution
    print("\nTactic distribution (bootstrapped):")
    for tactic in TACTIC_LABELS:
        count = df[tactic].sum()
        print(f"  {tactic}: {count} ({count/len(df)*100:.1f}%)")
    
    # Prepare data
    texts = df['full_text'].tolist()
    labels = df[TACTIC_LABELS].values
    
    # Split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    # Tokenizer
    print(f"\nLoading tokenizer from {BERT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    
    # Datasets
    train_ds = TacticDataset(train_texts, train_labels, tokenizer)
    val_ds = TacticDataset(val_texts, val_labels, tokenizer)
    test_ds = TacticDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model
    print("Building multi-task ManipulationBERT...")
    model = ManipulationBERT(BERT_PATH, num_tactics=len(TACTIC_LABELS)).to(DEVICE)
    
    # Loss and optimizer
    # BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_batch = batch['labels'].to(DEVICE)
                
                logits = model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                val_preds.append(preds)
                val_labels.append(labels_batch.cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/manipulation_detector_best.pt')
    
    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/manipulation_detector_best.pt'))
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['labels'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            test_preds.append(preds)
            test_labels.append(labels_batch.cpu().numpy())
    
    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)
    
    print(f"\nTest Macro-F1: {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}")
    print("\nPer-tactic results:")
    for i, tactic in enumerate(TACTIC_LABELS):
        f1 = f1_score(test_labels[:, i], test_preds[:, i], zero_division=0)
        print(f"  {tactic}: F1={f1:.4f}")
    
    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    
    return model, tokenizer

# ============================================================================
# STEP 4: Predict on new text
# ============================================================================
def predict_tactics(text, model_path=OUTPUT_DIR):
    """Predict manipulation tactics in a single conversation text."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = ManipulationBERT(BERT_PATH, num_tactics=len(TACTIC_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(f'{model_path}/manipulation_detector_best.pt', map_location=DEVICE))
    model.eval()
    
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
    results = {}
    for i, tactic in enumerate(TACTIC_LABELS):
        results[tactic] = {
            'present': bool(preds[i]),
            'confidence': float(probs[i])
        }
    
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the manipulation detector')
    parser.add_argument('--predict', type=str, help='Predict tactics in a text file')
    args = parser.parse_args()
    
    if args.train:
        train_manipulation_detector()
    elif args.predict:
        with open(args.predict, 'r') as f:
            text = f.read()
        results = predict_tactics(text)
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python manipulation_detector.py --train")
        print("       python manipulation_detector.py --predict conversation.txt")