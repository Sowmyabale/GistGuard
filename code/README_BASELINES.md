# Baselines + V3 Hybrid — HPC Run Guide

Date: 2026-10-06
Purpose: produce the B1–B5 + V3 rows of the Paper-1 comparison table
(`GXAI_Baselines_Plan_Sep21.docx`). Same COVA-X_Split (60/20/20, seed 42) as v1.5/v2.

## Files in this folder (`v1_rerun/`)

| File | What it produces |
|------|------------------|
| `train_baselines.py` | B1 TF-IDF+LogReg, B2 TF-IDF+SVM, B3 TF-IDF+RandomForest (CPU, minutes) |
| `train_single_task.py` | B4 DistilBERT single-task, B5 RoBERTa single-task (GPU, ~1-2 h each) |
| `train_v3_hybrid.py` | V3 hybrid: MiniLM embeddings + tactic-similarity + linguistic markers -> LogReg/RF/MLP; also a pure zero-shot-similarity reference (GPU encoding ~10-20 min, rest CPU) |

All three reuse `dataset_utils.py` + `config.py` (copy the whole `v1_rerun/` folder).

## 0. Activate env

```bash
conda activate aging
cd ~/Harika/code/v1_rerun        # wherever you copied the folder
```

## 1. CPU baselines first (fast win)

```bash
python train_baselines.py \
  --output_dir results_baselines
```

Output: `results_baselines/tfidf_baselines_summary.json` + confusion matrices.

## 2. Single-task encoders (GPU, run one after the other)

```bash
# B4 — DistilBERT
python train_single_task.py \
  --dataset_dir ".../COVA-X_Split" \
  --model_name distilbert-base-uncased \
  --output_dir models/b4_distilbert_single

# B5 — RoBERTa single-task
python train_single_task.py \
  --dataset_dir ".../COVA-X_Split" \
  --model_name roberta-base \
  --output_dir models/b5_roberta_single
```

Each prints the test classification report and saves `test_summary.json`.
B5 vs v1.5 (multi-task, 99.86%) isolates the tactic-auxiliary contribution.
B4 vs B5 isolates the encoder contribution.

## 3. V3 hybrid (GPU for encoding, then CPU)

```bash
python train_v3_hybrid.py \
  --dataset_dir ".../COVA-X_Split" \
  --output_dir results_v3hybrid
```

Output: `results_v3hybrid/v3_hybrid_summary.json`
Rows: `zero_shot_similarity` (no training), `hybrid_logreg`, `hybrid_rf`, `hybrid_mlp`.

## 4. Overnight everything (recommended)

```bash
python train_baselines.py --dataset_dir ".../COVA-X_Split" --output_dir results_baselines && \
python train_single_task.py --dataset_dir ".../COVA-X_Split" --model_name distilbert-base-uncased --output_dir models/b4_distilbert_single && \
python train_single_task.py --dataset_dir ".../COVA-X_Split" --model_name roberta-base --output_dir models/b5_roberta_single && \
python train_v3_hybrid.py --dataset_dir ".../COVA-X_Split" --output_dir results_v3hybrid
' > baselines_run.log 2>&1 &
tail -f baselines_run.log
```

## 5. Paper comparison table (fill after runs)

| Model | Test acc | Macro-F1 | Source |
|-------|----------|----------|--------|
| TF-IDF + LogReg | ? | ? | B1 |
| TF-IDF + SVM | ? | ? | B2 |
| TF-IDF + RF | ? | ? | B3 |
| DistilBERT single-task | ? | ? | B4 |
| RoBERTa single-task | ? | ? | B5 |
| Zero-shot v2 (LLM) | 99.3% det. | — | done |
| **v1.5 RoBERTa multi-task (ours)** | **99.86%** | — | done |
| V3 hybrid LogReg | ? | ? | V3 |
| SmishX (external ref) | 98.8% (3-class SMS) | — | SOUPS'25 |

## Notes

- If HPC has no internet, pre-download `distilbert-base-uncased`, `roberta-base`,
  and `sentence-transformers/all-MiniLM-L6-v2` on a machine that does
  (`huggingface-cli download <name>`) or reuse the existing HF cache
  (`~/.cache/huggingface`) — v1.5 models are already cached there.
- Everything is seeded (config.SEED = 42) and resumes are not needed: each script
  runs to completion in one go and writes JSON summaries at the end.
