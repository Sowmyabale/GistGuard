"""
train_layer2_profiler.py
Layer 2: Cognitive Profiler - Sentence-Transformers + BiLSTM
Predicts victim cognitive state from conversation turns.
Run on: Google Colab (free T4 GPU) or university GPU cluster
"""

# ============================================================================
# STEP 0: Install dependencies (run once in Colab)
# ============================================================================
# !pip install transformers sentence-transformers torch scikit-learn pandas numpy

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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# CONFIGURATION
# ============================================================================
SENTENCE_MODEL = 'all-MiniLM-L6-v2'  # 384-dim embeddings, fast, ~80MB
EMBED_DIM = 384
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PATIENCE = 5
OUTPUT_DIR = './layer2_profiler_results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# STEP 1: Load Full COVA-X with Victim Configs
# ============================================================================
def load_cova_x_with_profiles(base_dir="dataset/COVA-X Dataset/COVA-X_Dataset"):
    """Load ALL conversations with victim cognitive profiles."""
    rows = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                
                victim = conv.get('victim_config', {})
                turns = conv.get('turns', [])
                
                # Extract victim turns only (what the victim says)
                victim_turns = [t for t in turns if t.get('role') == 'victim']
                attacker_turns = [t for t in turns if t.get('role') == 'attacker']
                
                # Encode labels
                trust_map = {
                    'highly_skeptical': 0,
                    'skeptical': 0,
                    'cautious': 1,
                    'somewhat_trusting': 1,
                    'moderately_trusting': 1,
                    'very_trusting': 2,
                    'highly_trusting': 2,
                }
                tech_map = {'low': 0, 'moderate': 1, 'high': 2}
                awareness_map = {'unaware': 0, 'vaguely_aware': 1, 'aware': 2}
                
                rows.append({
                    'conversation_id': conv.get('conversation_id'),
                    'scam_type': conv.get('scam_type'),
                    'victim_turns': [t.get('content', '') for t in victim_turns],
                    'attacker_turns': [t.get('content', '') for t in attacker_turns],
                    'num_victim_turns': len(victim_turns),
                    'num_attacker_turns': len(attacker_turns),
                    'total_turns': len(turns),
                    # Targets
                    'trust_level': trust_map.get(victim.get('trust_level'), 1),
                    'tech_savviness': tech_map.get(victim.get('tech_savviness'), 0),
                    'scam_awareness': awareness_map.get(victim.get('scam_awareness'), 1),
                    'age': victim.get('age', '70-74'),
                    'gender': victim.get('gender', 'female'),
                    'living_situation': victim.get('living_situation', 'lives_alone'),
                })
            except Exception as e:
                pass
    
    return pd.DataFrame(rows)

# ============================================================================
# STEP 2: Encode Conversations with Sentence-Transformers
# ============================================================================
class ConversationEncoder:
    def __init__(self, model_name=SENTENCE_MODEL):
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode_conversation(self, victim_turns, attacker_turns, max_turns=25):
        """
        Encode a conversation as a sequence of turn embeddings.
        Returns: (max_turns, embed_dim) array
        """
        # Combine turns with role markers
        all_turns = []
        for t in victim_turns[:max_turns//2]:
            all_turns.append(f"VICTIM: {t}")
        for t in attacker_turns[:max_turns//2]:
            all_turns.append(f"ATTACKER: {t}")
        
        # Sort by conversation order (interleave victim/attacker)
        # For simplicity, just encode up to max_turns turns in order
        all_turns = all_turns[:max_turns]
        
        # Encode each turn
        if len(all_turns) == 0:
            embeddings = np.zeros((max_turns, EMBED_DIM))
        else:
            embeddings = self.model.encode(all_turns, convert_to_numpy=True, show_progress_bar=False)
            # Pad to max_turns
            if len(embeddings) < max_turns:
                pad = np.zeros((max_turns - len(embeddings), EMBED_DIM))
                embeddings = np.vstack([embeddings, pad])
            else:
                embeddings = embeddings[:max_turns]
        
        return embeddings

# ============================================================================
# STEP 3: PyTorch Dataset
# ============================================================================
class ConversationDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ============================================================================
# STEP 4: BiLSTM Model
# ============================================================================
class CognitiveProfilerBiLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM outputs 2*hidden_dim (forward + backward)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        # hidden: (num_layers*2, batch, hidden_dim)
        # Concatenate final forward and backward hidden states
        final_forward = hidden[-2]  # (batch, hidden_dim)
        final_backward = hidden[-1]  # (batch, hidden_dim)
        final_hidden = torch.cat([final_forward, final_backward], dim=1)
        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits

# ============================================================================
# STEP 5: Training Function
# ============================================================================
def train_profiler(target_column='trust_level', num_classes=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("Loading full COVA-X dataset...")
    df = load_cova_x_with_profiles()
    print(f"Loaded {len(df)} conversations")
    
    # Encode conversations
    print("Encoding conversations with Sentence-Transformers...")
    encoder = ConversationEncoder()
    
    embeddings = []
    for _, row in df.iterrows():
        emb = encoder.encode_conversation(row['victim_turns'], row['attacker_turns'])
        embeddings.append(emb)
    
    X = np.array(embeddings)  # (N, max_turns, embed_dim)
    y = df[target_column].values
    
    print(f"Embeddings shape: {X.shape}")
    print(f"Target distribution: {Counter(y)}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Datasets
    train_ds = ConversationDataset(X_train, y_train)
    val_ds = ConversationDataset(X_val, y_val)
    test_ds = ConversationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model
    model = CognitiveProfilerBiLSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/{target_column}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{target_column}_best.pt'))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(batch_y.numpy())
    
    print(f"\nTest Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(classification_report(test_labels, test_preds, digits=3))
    
    return model

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    # Train for each cognitive dimension
    targets = [
        ('trust_level', 3),
        ('tech_savviness', 3),
        ('scam_awareness', 3),
    ]
    
    for target, n_classes in targets:
        print(f"\n{'='*60}")
        print(f"Training Cognitive Profiler for: {target}")
        print(f"{'='*60}")
        model = train_profiler(target, n_classes)
    
    print("\nLayer 2 training complete!")
