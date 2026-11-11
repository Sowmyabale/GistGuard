"""
Generate the comprehensive GXAI-Elderly comparison notebook.
Run: python generate_notebook.py
Output: GXAI_Elderly_Framework_Comparison.ipynb
"""

import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n"),
    }

cells = []

# ============================================================
# TITLE
# ============================================================
cells.append(md("""# GXAI-Elderly: Protecting Older Adults from Financial Scams

## A Three-Version Framework Evolution: Supervised → Zero-Shot → Hybrid ML

**Author:** Sowmya Hemanth  
**Date:** September 2025  
**Goal:** Compare three approaches for scam detection and gist-based protection for elderly victims

---
"""))

# ============================================================
# PROBLEM STATEMENT
# ============================================================
cells.append(md("""## 1. Problem Statement

Elderly adults lose **billions of dollars annually** to phone, text, and email scams. Key challenges:

| Challenge | Why it matters |
|-----------|---------------|
| **Cognitive decline** | Older adults process information differently (Fuzzy Trace Theory) — they need gist, not details |
| **Theory of Mind decline** | Reduced ability to detect deception with age (Calso et al., 2020) |
| **No training data for new scams** | Zero-day scams appear faster than labeled datasets can be built |
| **No funds for human subjects** | Cannot recruit elderly participants for user studies |

**Our solution:** A 3-layer protective framework that detects manipulation, infers mental states, and generates plain-language warnings.
"""))

# ============================================================
# DATASET
# ============================================================
cells.append(md("""## 2. Dataset: COVA-X

| Property | Value |
|----------|-------|
| Total conversations | 10,985 |
| Scam types | 8 |
| Format | Multi-turn JSON (scammer + victim turns) |
| Split | 64% train / 16% val / 20% test |
| Source | Simulated elderly-victim scam conversations |

**Scam type distribution:**
- romance: 1,934
- government_impersonation: 1,545
- medicare: 1,490
- grandparent: 1,376
- bank: 1,360
- virtual_kidnapping: 1,330
- lottery: 975
- investment: 975
"""))

cells.append(code("""# Dataset exploration
import json, os, glob
from collections import Counter

DATASET_DIR = "dataset/COVA-X Dataset/COVA-X_Dataset"

counts = Counter()
for d in glob.glob(os.path.join(DATASET_DIR, "*")):
    if os.path.isdir(d):
        scam_type = os.path.basename(d)
        counts[scam_type] = len(glob.glob(os.path.join(d, "*.json")))

print("Scam type distribution:")
for scam, n in counts.most_common():
    print(f"  {scam:30s}: {n}")
print(f"\\nTotal: {sum(counts.values())}")

# Sample conversation structure
sample = glob.glob(os.path.join(DATASET_DIR, "bank", "*.json"))[0]
with open(sample) as f:
    conv = json.load(f)
print(f"\\nSample keys: {list(conv.keys())}")
print(f"Sample turns: {len(conv.get('turns', []))}")
if conv.get('turns'):
    print(f"First turn: {conv['turns'][0]}")
"""))

# ============================================================
# FRAMEWORK ARCHITECTURE
# ============================================================
cells.append(md("""## 3. Framework Architecture (All Versions)

All three versions share the same 3-layer conceptual structure:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT                                │
│   Multi-turn conversation + Victim sign-up profile      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Manipulation Detection                        │
│  ─────────────────────────────                          │
│  Detects: scam type, manipulation tactic, confidence    │
│  v1: Fine-tuned BERT classifier                         │
│  v2: Zero-shot (semantic similarity + linguistic rules) │
│  v3: Supervised transformer with hierarchical attention │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: Theory of Mind (ToM) Inference                │
│  ─────────────────────────────                          │
│  Infers: scammer intent, victim susceptibility,         │
│          truth bias, emotional state                    │
│  (Same across all versions — psychological rules)       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: Gist-Based Explanation Generator              │
│  ─────────────────────────────                          │
│  Generates: plain-language warning + action item        │
│  v1: DialoGPT-medium fine-tuned                         │
│  v2: Few-shot rule-based selection                      │
│  v3: LLaMA-3 fine-tuned with LoRA                       │
└─────────────────────────────────────────────────────────┘
```
"""))

# ============================================================
# V1 — SUPERVISED BERT
# ============================================================
cells.append(md("""## 4. V1 — Supervised BERT (Summer 2025)

**Approach:** Fine-tune pre-trained BERT on COVA-X for 8-class scam classification.

**Why it was built first:** Standard supervised baseline, maximum accuracy.

| Component | Model | Training |
|-----------|-------|----------|
| Layer 1 | DistilBERT-base | Fine-tuned on 7,689 conversations |
| Layer 2 | all-MiniLM-L6-v2 + MLP | Trained on victim features |
| Layer 3 | DialoGPT-medium | Fine-tuned on gist explanations |
| Data split | 70% / 15% / 15% | train / val / test |
"""))

cells.append(code("""# V1: BERT Training Skeleton (from experiments/train_layer1_bert.py)
# This is the architecture — full training code is in experiments/ folder

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np

MODEL = "distilbert-base-uncased"
NUM_CLASSES = 8

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)

# Model: 8-class classification
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL, num_labels=NUM_CLASSES
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

# Training arguments (v1 configuration)
training_args = TrainingArguments(
    output_dir="./results_v1",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    logging_steps=50,
    save_steps=500,
    eval_strategy="epoch",      # transformers >= 4.41
    save_strategy="epoch",
    load_best_model_at_end=True,
)

print("V1 BERT model loaded. Training config ready.")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""))

cells.append(md("""**V1 Results (from summer training run):**

| Metric | Value |
|--------|-------|
| Scam classification accuracy | ~98% (train) |
| Detection confidence | 0.994 avg |
| Per-scam-type accuracy | 100% (all 8 types) |

**Limitations of V1:**
- Requires labeled training data — cannot handle zero-day scams
- Fine-tuning requires GPU hours and COVA-X labels
- Not explainable — BERT is a black box
- Cannot adapt to individual victim profiles without retraining
"""))

# ============================================================
# V2 — ZERO-SHOT
# ============================================================
cells.append(md("""## 5. V2 — Zero-Shot Framework (August–September 2025)

**Motivation:** After professor left, no funds for GPU/training. Pivoted to zero-shot: **no training on scam data, no LLM prompting, no API costs.**

| Design Principle | Implementation |
|-----------------|----------------|
| No training | Pre-trained sentence encoder + linguistic rules |
| No LLM prompting | Few-shot example selection, not generation |
| Psychologically grounded | ToM inference from Calso, Ebner, Ruffman, Wang & Lighthall |
| Explainable | Every decision traceable to rules or similarity scores |
"""))

cells.append(md("""### 5.1 Layer 1: Zero-Shot Manipulation Detector

**Hybrid approach:**
- **Semantic similarity** (sentence-transformers): cosine similarity between input text and 90+ curated manipulation phrases
- **Linguistic deception markers** (Hancock et al., 2007): urgency verbs, certainty words, fear words, pronoun patterns

**8 manipulation tactics detected:**
urgency_pressure, authority_impersonation, fear_threat, social_proof, reciprocity_gift, request_sensitive_info, isolation_secrecy, over_familiarity
"""))

cells.append(code("""# V2: Layer 1 — Zero-Shot Manipulation Detector
import sys
sys.path.insert(0, 'v2_gpu_run')

from v2_gpu_run.config import MANIPULATION_TACTICS, DECEPTION_MARKERS
from v2_gpu_run.layer1_manipulation_detector import ManipulationDetector

detector = ManipulationDetector(MANIPULATION_TACTICS, DECEPTION_MARKERS)

test_cases = {
    "IRS Scam": "This is the IRS. You owe $5,000 in unpaid taxes. You must pay immediately or a warrant will be issued for your arrest.",
    "Grandparent Scam": "Hi grandma, it's me. I'm in jail and I need bail money. Please don't tell mom and dad. Wire me $2,000 right now.",
    "Lottery Scam": "Congratulations! You have been selected as a lucky winner of a $10,000 prize. Call now to claim your reward.",
    "Normal Message": "Hey, just checking in. Are we still meeting for lunch tomorrow?",
}

for name, text in test_cases.items():
    r = detector.detect(text)
    status = "SCAM" if r["detected"] else "SAFE"
    print(f"{name:20s} -> {status:5s} | Tactic: {r['primary_tactic']:28s} | Conf: {r['confidence']:.2f}")
"""))

cells.append(md("""### 5.2 Layer 2: Theory of Mind (ToM) Inference

**No ML — pure psychological rule-based inference.**

Infers **two mental states simultaneously:**

**Scammer's state:**
- Intent (money, info, access, combination)
- Deception strategy (primary tactic)
- False belief they want victim to accept
- Urgency level

**Victim's state:**
- Susceptibility (0–1) — from age, living situation, prior scam experience, trust tendency, urgency response
- Truth bias (0–1) — tendency to believe
- Emotional state (fear, trust, confusion, excitement, neutral)
- Cognitive load (low, medium, high)
- Protection level (low, moderate, high)

**Psychological grounding:**
- Calso et al. (2020): Age-related ToM decline in deception detection
- Ebner (2023): Social isolation increases scam vulnerability
- Ruffman et al. (2012): Prior scam experience → awareness
- Wang & Lighthall (2024): Older adults rely on gist processing
"""))

cells.append(code("""# V2: Layer 2 — ToM Inference
from v2_gpu_run.layer2_tom_inference import ToMInferenceEngine, SignUpProfile

engine = ToMInferenceEngine()

# Elderly profile: 75-year-old, lives alone, low tech comfort, no prior scam experience
profile = SignUpProfile(
    age_group="70-79",
    living_situation="alone",
    tech_comfort=2,
    prior_scam="no",
    trust_tendency=4,
    urgency_response=4,
    financial_control="self",
)

# Simulate Layer 1 output for IRS scam
layer1_result = detector.detect(test_cases["IRS Scam"])
tom = engine.infer(layer1_result, profile)

print("SCAMMER ToM:")
for k, v in tom["scammer_tom"].items():
    print(f"  {k:25s}: {v}")

print("\\nVICTIM ToM:")
for k, v in tom["victim_tom"].items():
    print(f"  {k:25s}: {v}")
"""))

cells.append(md("""### 5.3 Layer 3: Gist-Based Explanation Generator

**Fuzzy Trace Theory (Reyna & Brainerd, 1995):** Older adults rely on **gist** (bottom-line meaning) over verbatim details. Our gist generator:

- Adapts length to cognitive load (25/50/75 words)
- Adapts tone to emotional state (reassuring for fear/confusion)
- Avoids technical terms for low tech-comfort users
- Always includes a concrete action item

**Two modes:**
1. **Rule-based** (default): selects best-matching example from curated bank
2. **Neural** (optional): DialoGPT-medium generates novel gist from few-shot prompt
"""))

cells.append(code("""# V2: Layer 3 — Gist Generator
from v2_gpu_run.layer3_gist_generator import GistGenerator

# model_name=None -> rule-based (fast, deterministic)
# model_name="microsoft/DialoGPT-medium" -> neural (slow, novel text)
generator = GistGenerator(model_name=None)

for name, text in test_cases.items():
    l1 = detector.detect(text)
    tom = engine.infer(l1, profile)
    gist = generator.generate(l1, tom, profile.tech_comfort)
    print(f"\\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"  Gist:   {gist['gist']}")
    print(f"  Action: {gist['action_item']}")
"""))

# ============================================================
# V3 — HYBRID ML (PROPOSED)
# ============================================================
cells.append(md("""## 6. V3 — Hybrid ML Model (Proposed)

**Motivation:** V1 is accurate but requires training data and is a black box. V2 is explainable and zero-shot but less accurate. V3 combines the best of both.

**Proposed architecture: MHT-ESP (Multi-task Hierarchical Transformer for Elderly Scam Protection)**

```
┌─────────────────────────────────────────────────────────────┐
│  MHT-ESP Architecture                                        │
│                                                              │
│  Input: Conversation turns + Victim profile                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  HIERARCHICAL ENCODER                                │   │
│  │  ───────────────────                                 │   │
│  │  Word Level:    RoBERTa-large embeddings             │   │
│  │  Turn Level:    BiLSTM + Self-Attention              │   │
│  │  Conv Level:    Scammer-turn weighted attention      │   │
│  │  Profile:       Demographics embedding (concat)      │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MULTI-TASK HEADS                                    │   │
│  │  ──────────────                                      │   │
│  │  Head 1: Scam Type Classifier (8-way)                │   │
│  │  Head 2: Victim Susceptibility (regression, 0-1)     │   │
│  │  Head 3: Victim Truth Bias (regression, 0-1)         │   │
│  │  Head 4: Gist Decoder (T5 / LLaMA-3 with LoRA)       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Training:                                                   │
│  Phase 1: Contrastive pre-training (scam vs benign)         │
│  Phase 2: Multi-task fine-tuning on COVA-X                  │
│  Phase 3: RLHF for gist quality (optional)                  │
└─────────────────────────────────────────────────────────────┘
```
"""))

cells.append(md("""### V3: What Makes It Novel

| Component | Novelty |
|-----------|---------|
| Hierarchical attention over scammer turns | No prior work explicitly weights scammer vs victim turns differently in multi-turn scam detection |
| Joint 4-task optimization | Detection + susceptibility + truth bias + generation trained simultaneously |
| Victim profile fusion | Demographics embedded into conversation representation, not just appended |
| Contrastive pre-training | Learns scam vs benign embedding space before task-specific fine-tuning |
| Curriculum learning | Easy scams (grandparent) → hard scams (investment, romance) |
"""))

cells.append(code("""# V3: Model Architecture Skeleton (PyTorch)
# This is the proposed architecture — to be implemented with new ML advisor

import torch
import torch.nn as nn

class MHTESP(nn.Module):
    \"\"\"Multi-task Hierarchical Transformer for Elderly Scam Protection\"\"\"

    def __init__(self, roberta_dim=1024, profile_dim=64, hidden_dim=512, num_classes=8):
        super().__init__()

        # Word-level encoder (pre-trained, frozen initially)
        # self.roberta = RobertaModel.from_pretrained('roberta-large')

        # Turn-level encoder
        self.turn_lstm = nn.LSTM(
            input_size=roberta_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Conversation-level attention (scammer turns weighted higher)
        self.conv_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,   # bidirectional
            num_heads=8,
            dropout=0.1,
        )

        # Victim profile encoder
        self.profile_encoder = nn.Sequential(
            nn.Linear(8, profile_dim),   # 8 profile features
            nn.ReLU(),
            nn.Linear(profile_dim, profile_dim),
            nn.ReLU(),
        )

        # Fusion
        self.fusion = nn.Linear(hidden_dim * 2 + profile_dim, hidden_dim)

        # Task heads
        self.classifier = nn.Linear(hidden_dim, num_classes)       # Head 1
        self.susceptibility_head = nn.Linear(hidden_dim, 1)        # Head 2
        self.truth_bias_head = nn.Linear(hidden_dim, 1)            # Head 3
        # Head 4: T5 decoder for gist generation (separate model)

    def forward(self, turn_embeddings, speaker_mask, profile_features):
        # turn_embeddings: [batch, num_turns, roberta_dim]
        # speaker_mask: [batch, num_turns] — 1 for scammer, 0 for victim

        # Turn-level encoding
        turn_out, _ = self.turn_lstm(turn_embeddings)  # [batch, num_turns, hidden*2]

        # Weight scammer turns higher
        weights = speaker_mask.unsqueeze(-1).float() * 2.0 + 1.0  # scammer: 3x, victim: 1x
        turn_weighted = turn_out * weights

        # Conversation-level attention
        conv_out, _ = self.conv_attention(
            turn_weighted.transpose(0, 1),
            turn_weighted.transpose(0, 1),
            turn_weighted.transpose(0, 1),
        )
        conv_repr = conv_out.mean(dim=0)  # [batch, hidden*2]

        # Profile encoding + fusion
        profile_repr = self.profile_encoder(profile_features)  # [batch, profile_dim]
        fused = self.fusion(torch.cat([conv_repr, profile_repr], dim=-1))
        fused = torch.relu(fused)

        # Task outputs
        return {
            "scam_type": self.classifier(fused),           # [batch, 8]
            "susceptibility": torch.sigmoid(self.susceptibility_head(fused)),  # [batch, 1]
            "truth_bias": torch.sigmoid(self.truth_bias_head(fused)),          # [batch, 1]
        }

# Multi-task loss
def multi_task_loss(outputs, targets, alpha=1.0, beta=0.5, gamma=0.5):
    ce_loss = nn.CrossEntropyLoss()(outputs["scam_type"], targets["scam_type"])
    mse_susc = nn.MSELoss()(outputs["susceptibility"].squeeze(), targets["susceptibility"])
    mse_bias = nn.MSELoss()(outputs["truth_bias"].squeeze(), targets["truth_bias"])
    return alpha * ce_loss + beta * mse_susc + gamma * mse_bias

model = MHTESP()
print(f"MHT-ESP parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Architecture ready for training with new ML advisor.")
"""))

# ============================================================
# COMPARISON TABLE
# ============================================================
cells.append(md("""## 7. V1 vs V2 vs V3 Comparison

| Dimension | V1 (Supervised BERT) | V2 (Zero-Shot) | V3 (Hybrid ML — Proposed) |
|-----------|----------------------|----------------|---------------------------|
| **Training data needed** | Yes (7,689 labeled) | No | Yes (contrastive + fine-tune) |
| **GPU hours** | ~4 hours | 0 | ~8–12 hours |
| **API costs** | None | None | None (open models) |
| **Accuracy** | ~98% (seen types) | ~70–80% (estimated) | Target: ~95% + generalization |
| **Zero-day detection** | No | Yes | Partial (contrastive helps) |
| **Explainability** | Low (black box) | High (rules + scores) | Medium (attention maps) |
| **Personalization** | No | Yes (ToM rules) | Yes (profile embedding) |
| **Gist quality** | Template-based | Rule-based or DialoGPT | LLaMA-3 + LoRA fine-tuned |
| **Novelty for publication** | Low (standard BERT) | Medium (zero-shot + ToM) | **High** (custom architecture) |
| **Reproducibility** | Good | Excellent (deterministic) | Good (seed controlled) |
| **Time to results** | 1 day | **30 minutes** | 1–2 weeks |
"""))

cells.append(md("""## 8. Evaluation Metrics

All versions evaluated on the same 64/16/20 split:

| Metric | Description |
|--------|-------------|
| Detection Rate | % of scam conversations correctly flagged |
| Top-1 Tactic Accuracy | Predicted primary tactic matches ground truth |
| Top-3 Tactic Accuracy | Ground truth in top-3 predicted tactics |
| Per-class F1 | For each of 8 scam types |
| Avg Processing Time | Seconds per conversation |
"""))

cells.append(code("""# Load and compare evaluation results
import json, glob
import pandas as pd

def load_eval_results(results_dir="results"):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "v2_eval_*.json")):
        with open(path) as f:
            data = json.load(f)
        if "metrics" in data:
            split = data.get("split", os.path.basename(path))
            m = data["metrics"]
            rows.append({
                "Split": split,
                "Total": m["total"],
                "Detection Rate": f"{m['detection_rate']:.2%}",
                "Top-1 Acc": f"{m['top1_tactic_accuracy']:.2%}",
                "Top-3 Acc": f"{m['top3_tactic_accuracy']:.2%}",
                "Avg Time (s)": m["avg_time_sec"],
            })
    return pd.DataFrame(rows)

df = load_eval_results("results")
print("V2 Zero-Shot Evaluation Results:")
print(df.to_string(index=False))
"""))

# ============================================================
# VISUALIZATION
# ============================================================
cells.append(code("""# Framework architecture diagram
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(x, y, w, h, color, label, sublabel=""):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        alpha=0.8,
        linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, label,
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha='center', va='center', fontsize=8, color='white', style='italic')

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Title
ax.text(5, 9.7, 'GXAI-Elderly: 3-Layer Protective Framework',
        ha='center', fontsize=16, fontweight='bold')

# Input
draw_box(3, 8.5, 4, 0.7, '#2C3E50', 'INPUT', 'Multi-turn conversation + Victim sign-up profile')
draw_arrow(5, 8.5, 5, 7.8)

# Layer 1
draw_box(1.5, 6.2, 7, 1.5, '#E74C3C', 'LAYER 1: Manipulation Detection',
         'Scam type | Tactic | Confidence')
ax.text(5, 6.5, 'V1: BERT  |  V2: Zero-shot (semantic + linguistic)  |  V3: Hierarchical Transformer',
        ha='center', fontsize=8, color='white')
draw_arrow(5, 6.2, 5, 5.5)

# Layer 2
draw_box(1.5, 4, 7, 1.5, '#F39C12', 'LAYER 2: Theory of Mind Inference',
         'Scammer intent | Victim susceptibility | Truth bias | Emotional state')
ax.text(5, 4.3, 'Psychological rules: Calso (2020), Ebner (2023), Ruffman (2012)',
        ha='center', fontsize=8, color='white')
draw_arrow(5, 4.0, 5, 3.3)

# Layer 3
draw_box(1.5, 1.8, 7, 1.5, '#27AE60', 'LAYER 3: Gist-Based Explanation Generator',
         'Plain-language warning + Action item')
ax.text(5, 2.1, 'V1: DialoGPT  |  V2: Few-shot selection  |  V3: LLaMA-3 + LoRA',
        ha='center', fontsize=8, color='white')

# Output
draw_box(3.5, 0.3, 3, 0.7, '#8E44AD', 'OUTPUT', 'Gist + Action for elderly user')

plt.tight_layout()
plt.savefig('framework_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Architecture diagram saved: framework_architecture.png")
"""))

# ============================================================
# CONCLUSIONS
# ============================================================
cells.append(md("""## 9. Conclusions & Next Steps

### What we built (Summer–Fall 2025)
1. **V1 Supervised BERT** — proof of concept, ~98% accuracy, but black box
2. **V2 Zero-Shot** — explainable, no training, psychologically grounded, works end-to-end
3. **V3 Hybrid ML** — proposed architecture for new ML advisor, combines V1 accuracy + V2 explainability

### Key Findings
- Zero-shot detection works but is less accurate than supervised (expected)
- ToM inference adds psychological grounding that pure ML approaches lack
- Gist-based explanations are essential for elderly users (Fuzzy Trace Theory)
- The framework is modular — each layer can be independently improved

### Next Steps (with new ML advisor)
1. Implement MHT-ESP (V3) with PyTorch + LoRA
2. Contrastive pre-training on scam vs benign conversations
3. Multi-task fine-tuning with curriculum learning
4. Human evaluation of gist quality (with psychology collaborator)
5. Multimodal extension: voice features from YouTube scam calls

### Publication Target
- **Venue:** AAAI AI for Social Good / ACM CHI / Journal of Applied Gerontology
- **Contribution:** First framework to jointly optimize scam detection + ToM inference + gist generation for elderly protection
"""))

# ============================================================
# REFERENCES
# ============================================================
cells.append(md("""## 10. References

1. Calso, C., Besnard, J., & Allain, P. (2020). Study of the theory of mind in normal aging: focus on the deception detection and its links with other cognitive functions. *Aging, Neuropsychology, and Cognition*, 27(4), 430–452.

2. Cialdini, R. B. (2001). *Influence: Science and Practice* (4th ed.). Allyn & Bacon.

3. Hancock, J. T., Curry, L. E., Goorha, S., & Woodworth, M. (2007). On lying and being lied to: A linguistic analysis of deception. *Discourse Processes*, 45(1), 1–23.

4. Ebner, N. C. (2023). Social and emotional aging. *Current Opinion in Psychology*, 34, 101–106.

5. Reyna, V. F., & Brainerd, C. J. (1995). Fuzzy-trace theory: An interim synthesis. *Learning and Individual Differences*, 7(1), 1–75.

6. Ruffman, T., Murray, J., Halberstadt, J., & Vater, T. (2012). Age-related differences in deception. *Psychology and Aging*, 27(3), 543–549.

7. Wang, L., & Lighthall, N. R. (2024). Age differences in decision making: The role of memory and cognition in social judgments. *Psychology and Aging*.

8. Adkins, S., et al. (2025). Linguistic markers of deception in scam communications. *Journal of Language and Social Psychology*.

9. Stanley, J. T., & Blanchard-Fields, F. (2008). Challenges older adults face in detecting deceit: The role of emotion recognition. *Psychology and Aging*, 23(1), 24–32.
"""))

# ============================================================
# Build notebook JSON
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "GXAI_Elderly_Framework_Comparison.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
