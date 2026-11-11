# Model Weights — Availability and Checksums

Trained weights are **not committed** to this repository (size; base models are
public HF checkpoints). They are shared **on request** for review/reproduction.
This file lists each model, its role, and a SHA-256 checksum so a received copy
can be verified. Checksums marked *(training host)* are recorded on the GPU
training machine and will be added here when transferred.

## Base models (public, Hugging Face)

| Component | Base checkpoint |
|---|---|
| Layer 1 encoder | `roberta-base` |
| Gist generator (Layers 3 / all gist variants) | `meta-llama/Meta-Llama-3.1-8B-Instruct` (local/4-bit for annotation; LoRA adapters below) |
| BERT detection baseline | `bert-base-uncased` |
| DistilBERT baseline | `distilbert-base-uncased` |

## Trained weights

All checksums below are now recorded — no placeholders remain.

| Model | Role | File | SHA-256 | Size |
|---|---|---|---|---|
| Layer 1 multitask (scam 8-class + tactic head) | detection | `layer1_multitask/layer1_multitask.pt` | `013e010b91cb3f7c…` | 498.7 MB |
| Layer 1 full checkpoint (ckpt-2472) | detection | `layer1_multitask/trainer/checkpoint-2472/model.safetensors` | `3ab6e4c966fd744b…` | 498.6 MB |
| Layer 1 tactic-v2 (7-class, LLM-annotation retrain; full RoBERTa model, not a LoRA adapter) | detection | `layer1_multitask_tacticv2/model.safetensors` | `eb5f3dc6e24099985…` | 498.7 MB |
| GistGuard adapter = SFT-v2s (LoRA on Llama-3.1-8B, `silver_v2_short`) | gist generation | `l3_gist_lora_v4b/adapter_model.safetensors` | `fcc396c0404867c9…` | 167.8 MB |
| GistGuard adapter config | — | `l3_gist_lora_v4b/adapter_config.json` | `688799ea2b1fe0cf…` | 1 KB |
| SFT-v1 adapter (LoRA, `silver_train`, ckpt-824) | gist generation | `l3_gist_lora/trainer/checkpoint-824/adapter_model.safetensors` | `dd5fd523b3c61cf0…` | 167.8 MB |
| SFT-v2 adapter (LoRA, `silver_v2`) | gist generation | `l3_gist_lora_v4/adapter_model.safetensors` | `052e321107a9e8c73…` | 167.8 MB |
| DistilBERT baseline (ckpt-824) | detection baseline | `b4_distilbert_single/trainer/checkpoint-824/model.safetensors` | `51b8def753cb0afa…` | 265.5 MB |
| BERT baseline | detection baseline | *(see `results_v1rerun/bert_baseline_summary.json`; weights on training host)* | — | — |

Full (untruncated) checksums: `models_checksums.json` at the repository root of
the training workspace, or recompute with `sha256sum <file>`.

## Reproduction without weights

All reported numbers can be reproduced end-to-end from the data + configs in
this repository by retraining with the commands in `RUN_MANIFEST.md`
(`train_layer1_tactic_v2.py`, `train_gist_lora.py`). Generation scripts accept
`--adapter <path>`; with no adapter they run the Prompt-LM (untuned) variant.
