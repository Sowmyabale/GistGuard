# GXAI-Elderly: Google Colab Setup Guide

Because PyTorch cannot be installed locally on your Windows machine (path-length error), the fastest way to train on the **full 10,985-conversation COVA-X dataset** is **Google Colab** (free T4 GPU) or your university GPU cluster.

---

## 1. Quick Start (Colab)

1. **Upload your project folder to Google Drive**  
   Place the entire `Research in Aging` folder (with `dataset/` and `paper1_gxai_elderly/`) under `MyDrive`.

2. **Open the notebook**  
   In Colab, upload and open:  
   `paper1_gxai_elderly/experiments/GXAI_Elderly_Colab.ipynb`

3. **Run cells top-to-bottom**  
   - Mount Drive  
   - Install dependencies  
   - Verify dataset  
   - Train Layer 1 → Layer 2 → Layer 3  
   - Run end-to-end inference

4. **Models are saved automatically** back to your Drive.

---

## 2. Files You Need

| File | Purpose |
|------|---------|
| `requirements.txt` | All pip dependencies |
| `GXAI_Elderly_Colab.ipynb` | One notebook that runs everything |
| `train_layer1_bert.py` | Layer 1: BERT scam classifier |
| `train_layer2_profiler.py` | Layer 2: BiLSTM cognitive profiler |
| `train_layer3_llm_lora.py` | Layer 3: LoRA LLM gist generator |
| `evaluate_end_to_end.py` | Chains L1→L2→L3 for inference |

---

## 3. Using Llama-3.1-8B or Mistral-7B (Optional)

The default Layer-3 model is **TinyLlama-1.1B** so it fits the free Colab T4.  
For stronger paper results you should switch to **Llama-3.1-8B-Instruct** or **Mistral-7B-Instruct**.

### Steps

1. **Get a HuggingFace token**  
   - Go to https://huggingface.co/settings/tokens  
   - Create a **Read** token  
   - Accept the model license on the model page (e.g. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

2. **Login in Colab**  
   Uncomment the HuggingFace login cell in the notebook:
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```
   Paste your token when prompted.

3. **Edit the model name** in `train_layer3_llm_lora.py`:
   ```python
   MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
   ```
   Then re-run Layer 3.

---

## 4. Expected Runtime (Colab T4)

| Layer | Time | Output size |
|-------|------|-------------|
| Layer 1 (DistilBERT) | ~20–30 min | ~250 MB |
| Layer 2 (BiLSTM ×3) | ~15–25 min | ~3 MB each |
| Layer 3 (TinyLlama + LoRA) | ~30–45 min | ~200 MB adapter |
| **Total** | **~70–100 min** | **~500 MB** |

---

## 5. Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `BATCH_SIZE` or `MAX_SEQ_LENGTH` in the script |
| `ModuleNotFoundError` | Re-run the `pip install` cell |
| Dataset not found | Check that `dataset/COVA-X Dataset/COVA-X_Dataset` exists relative to the project root |
| `gated repo` error | Make sure you accepted the license on HuggingFace and ran `notebook_login()` |

---

## 6. After Training

- Use `evaluate_end_to_end.py` to generate results on test conversations.
- Copy the JSON outputs into `paper1_gxai_elderly/results/` for your paper tables.
- Download the zipped models if you need to submit them as supplementary material.

**Need help?** Message me with the exact Colab error and I’ll fix it immediately.
