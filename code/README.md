# GXAI-Elderly v2 — GPU Run Package

Self-contained folder for running the v2 zero-shot framework on GPU (HPC / Google Colab).

## What's Inside

| File | Purpose |
|------|---------|
| `config.py` | Tactic DB, deception markers, ToM attributes, sign-up mapping |
| `layer1_manipulation_detector.py` | **Layer 1:** Zero-shot manipulation detection (semantic + linguistic) |
| `layer2_tom_inference.py` | **Layer 2:** Theory of Mind inference (scammer + victim mental states) |
| `layer3_gist_generator.py` | **Layer 3:** Gist-based explanation generator (few-shot) |
| `pipeline.py` | End-to-end pipeline connecting all 3 layers |
| `split_dataset.py` | Split COVA-X into train/val/test with stratification |
| `run_evaluation.py` | **Main evaluation script** — runs pipeline on all splits, computes metrics |
| `requirements.txt` | Python dependencies |

## Quick Start (Colab / HPC)

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Split the dataset (if not already split)

```bash
# Two-step split: 80/20 → 80/20  →  64% train, 16% val, 20% test
python split_dataset.py --two_step --seed 42 \
  --dataset_dir "/path/to/COVA-X_Dataset" \
  --output_dir "/path/to/COVA-X_Split"
```

### Step 3: Run evaluation

```bash
# Evaluate all splits (train + val + test)
python run_evaluation.py --split all \
  --dataset_dir "/path/to/COVA-X_Split" \
  --output_dir "results"

# Or evaluate only test split
python run_evaluation.py --split test \
  --dataset_dir "/path/to/COVA-X_Split" \
  --output_dir "results"
```

### Step 4: Check outputs

Results are saved to `results/`:
- `v2_eval_train.json` — per-conversation results + metrics for train split
- `v2_eval_val.json` — per-conversation results + metrics for val split
- `v2_eval_test.json` — per-conversation results + metrics for test split
- `v2_eval_summary.json` — combined summary across all splits

## Metrics Computed

| Metric | Description |
|--------|-------------|
| **Detection Rate** | % of scam conversations correctly flagged as scams |
| **Top-1 Tactic Accuracy** | % where predicted primary tactic matches true tactic |
| **Top-3 Tactic Accuracy** | % where true tactic is in top-3 predicted tactics |
| **Avg Time** | Seconds per conversation (end-to-end) |

## Expected Runtime

| Setup | Per conversation | Full COVA-X (~11K) |
|-------|------------------|---------------------|
| CPU only (no sentence-transformers) | ~0.1s | ~20 min |
| GPU with sentence-transformers | ~0.5s | ~90 min |
| GPU + neural gist (DialoGPT) | ~3-5s | ~8-12 hours |

**Tip:** For fast evaluation, Layer 3 falls back to rule-based gist generation automatically if `transformers` is not installed or the model fails to load.

## Demo (single text)

```bash
python pipeline.py --demo
```

This runs 4 test cases (IRS, Grandparent, Lottery, Normal) through all 3 layers and prints results.

## Customizing the Sign-Up Profile

Edit the default profile in `run_evaluation.py`:

```python
profile = SignUpProfile(
    age_group="70-79",          # under_60, 60-69, 70-79, 80+
    living_situation="alone",   # alone, spouse, family, community
    tech_comfort=2,             # 1-5 (1=low)
    prior_scam="no",            # lost, stopped, unsure, no
    trust_tendency=4,           # 1-5 (5=high trust)
    urgency_response=4,         # 1-5 (5=acts immediately)
    financial_control="self",   # self, mostly_self, family, advisor
)
```

## Zero-Shot = No Training Required

This framework does **NOT** require:
- Fine-tuning on scam data
- LLM prompting / API keys
- Labeled training sets

It runs entirely from:
- Pre-trained sentence encoders (Layer 1, optional)
- Linguistic deception markers (Layer 1)
- Psychological ToM models (Layer 2)
- Few-shot curated examples (Layer 3)

## Citation Support

The framework components are grounded in:
- **Layer 1:** Cialdini (2001), Hancock et al. (2007), Adkins et al. (2025)
- **Layer 2:** Calso et al. (2020), Ebner (2023), Ruffman et al. (2012), Stanley & Blanchard-Fields (2008)
- **Layer 3:** Wang & Lighthall (2024) — Fuzzy Trace Theory / gist processing in older adults
