# GXAI-Elderly: HPC Implementation Guide
# October 1st Deadline — Complete Execution Plan
# Created: August 27, 2026

## WHAT YOU ALREADY HAVE (Do NOT retrain these)

These models were trained on Colab T4 and are saved locally.
You will COPY these to HPC, NOT retrain them:

1. **Layer 1 BERT** (`layer1_bert_results/best_model/`)
   - distilbert-base-uncased fine-tuned on COVA-X
   - 99.6% accuracy
   - ~268MB

2. **Layer 2 BiLSTM** (`layer2_profiler_results/`)
   - 3 models: trust_level_best.pt, tech_savviness_best.pt, scam_awareness_best.pt
   - 94-96% F1
   - ~35MB each

3. **Layer 3 LoRA Adapter** (`layer3_llm_lora_results/lora_adapter/`)
   - LoRA adapter for TinyLlama-1.1B
   - ~18MB

4. **Datasets** (copy to HPC)
   - `dataset/COVA-X Dataset/COVA-X_Dataset/` (10,985 conversations)
   - `dataset/Youtube transcripts/converted_json/` (243 transcripts)

---

## WHAT YOU NEED TO BUILD ON HPC (New Work)

### Job 1: Manipulation Tactic Detection (Priority: HIGH)
**New model**: Multi-task BERT head for 6 manipulation tactics
**Input**: Your existing BERT + COVA-X dataset
**Output**: `manipulation_detector_results/`
**Time**: ~2-3 hours on V100/A100
**GPU Memory**: ~8GB

### Job 2: Retrain Layer 3 with LLaMA-3-8B (Priority: HIGHEST)
**New model**: LLaMA-3.1-8B-Instruct + LoRA for gist generation
**Input**: COVA-X dataset with exemplar-based training data
**Output**: `layer3_llama3_results/lora_adapter/`
**Time**: ~6-8 hours on A100
**GPU Memory**: ~16GB (4-bit quantization)

### Job 3: Generate All Gists with LLaMA-3 (Priority: HIGH)
**Input**: Trained LLaMA-3 + COVA-X test set
**Output**: `results/all_gists_llama3.json` (10,985 gists)
**Time**: ~4-6 hours on A100
**GPU Memory**: ~16GB

### Job 4: YouTube Validation Pipeline (Priority: HIGH)
**Input**: All 3 layers + 243 YouTube transcripts
**Output**: `youtube_validation_results.json`
**Time**: ~2-3 hours on V100/A100
**GPU Memory**: ~16GB

### Job 5: RL Feedback Loop (Priority: MEDIUM)
**Input**: Existing pipeline + simulated feedback
**Output**: `rl_feedback_results.json`
**Time**: ~1 hour on CPU/GPU
**GPU Memory**: ~8GB

---

## EXECUTION ORDER

```
Day 1 (Today):
  1. Copy existing models + datasets to HPC
  2. Submit Job 1: Manipulation Detector
  3. Submit Job 2: LLaMA-3 Training (starts in parallel)

Day 2:
  4. Job 1 completes → verify manipulation detector results
  5. Job 2 running → monitor training

Day 3:
  6. Job 2 completes → verify LLaMA-3 gist quality
  7. Submit Job 3: Generate All Gists
  8. Submit Job 4: YouTube Validation

Day 4:
  9. Jobs 3 & 4 complete → download results
  10. Submit Job 5: RL Feedback Loop

Day 5:
  11. All experiments complete
  12. Start human evaluation recruitment
```

---

## FILES TO COPY TO HPC

### From your local machine:
```
paper1_gxai_elderly/
  layer1_bert_results/best_model/     <-- COPY
  layer2_profiler_results/            <-- COPY
  layer3_llm_lora_results/lora_adapter/  <-- COPY (for comparison, not for main use)
  experiments/                        <-- COPY all new scripts
  
dataset/
  COVA-X Dataset/COVA-X_Dataset/      <-- COPY
  Youtube transcripts/converted_json/ <-- COPY
```

### Total size to copy: ~350MB (models) + ~50MB (data) = ~400MB

---

## HPC SETUP STEPS

### Step 1: Login to HPC
```bash
ssh your_username@hpc.university.edu
```

### Step 2: Create workspace
```bash
mkdir -p ~/gxai-elderly
cd ~/gxai-elderly
```

### Step 3: Copy files from local (run on your local machine)
```bash
# Use scp or rsync
scp -r paper1_gxai_elderly/layer1_bert_results your_username@hpc.university.edu:~/gxai-elderly/
scp -r paper1_gxai_elderly/layer2_profiler_results your_username@hpc.university.edu:~/gxai-elderly/
scp -r paper1_gxai_elderly/experiments your_username@hpc.university.edu:~/gxai-elderly/
scp -r dataset your_username@hpc.university.edu:~/gxai-elderly/
```

### Step 4: Create Python environment on HPC
```bash
cd ~/gxai-elderly
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets peft accelerate bitsandbytes sentence-transformers scikit-learn pandas numpy tqdm
```

### Step 5: Request interactive GPU session (for testing)
```bash
salloc --gres=gpu:1 --partition=gpu --time=01:00:00
```

---

## SLURM JOB SCRIPTS

Save these as `.slurm` files and submit with `sbatch`.

---

### Job 1: Manipulation Detector (`job_manipulation.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=manipulation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/manipulation_%j.out
#SBATCH --error=logs/manipulation_%j.err

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python manipulation_detector.py --train

echo "Manipulation detector training complete!"
```

---

### Job 2: LLaMA-3 Training (`job_llama3_train.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=llama3_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/llama3_train_%j.out
#SBATCH --error=logs/llama3_train_%j.err

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python train_layer3_llama3.py

echo "LLaMA-3 training complete!"
```

---

### Job 3: Generate All Gists (`job_generate_gists.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=generate_gists
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/generate_gists_%j.out
#SBATCH --error=logs/generate_gists_%j.err

# Dependency: only run after LLaMA-3 training completes
#SBATCH --dependency=afterok:JOB_ID_OF_LLAMA3_TRAIN

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python generate_all_gists_hpc.py --model llama3 --output all_gists_llama3.json

echo "Gist generation complete!"
```

---

### Job 4: YouTube Validation (`job_youtube.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=youtube_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/youtube_%j.out
#SBATCH --error=logs/youtube_%j.err

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python evaluate_youtube_pipeline.py --input_dir "../../dataset/Youtube transcripts/converted_json"

echo "YouTube validation complete!"
```

---

### Job 5: RL Feedback Loop (`job_rl.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=rl_loop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/rl_%j.out
#SBATCH --error=logs/rl_%j.err

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python rl_feedback_loop.py --batch_eval --num_samples 200

echo "RL feedback loop complete!"
```

---

## SUBMITTING JOBS

```bash
# Create logs directory
mkdir -p logs

# Submit all jobs
sbatch job_manipulation.slurm
sbatch job_llama3_train.slurm
sbatch job_youtube.slurm

# For generate_gists, wait for llama3_train to finish, then:
# Get the job ID from squeue, then submit:
sbatch --dependency=afterok:JOB_ID job_generate_gists.slurm

# RL can run anytime after layer models are loaded
sbatch job_rl.slurm
```

---

## MONITORING JOBS

```bash
# Check job status
squeue -u your_username

# Check GPU usage on your node
ssh node_name nvidia-smi

# View live logs
tail -f logs/manipulation_*.out

# Cancel a job
scancel JOB_ID
```

---

## DOWNLOADING RESULTS

After all jobs complete:

```bash
# On HPC, zip results
cd ~/gxai-elderly
zip -r results.zip results/ manipulation_detector_results/ layer3_llama3_results/ youtube_validation_results.json rl_feedback_results.json

# On your local machine
scp your_username@hpc.university.edu:~/gxai-elderly/results.zip .
```

---

## HUGGINGFACE TOKEN (Required for LLaMA-3)

LLaMA-3 requires HuggingFace authentication:

```bash
# On HPC
huggingface-cli login
# Enter your token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

---

## TROUBLESHOOTING

### Out of Memory
- Reduce batch size in scripts
- Use `--mem=64G` in SLURM script
- Use A100 instead of V100

### Job Fails Immediately
- Check `logs/*.err` for error messages
- Verify all files were copied correctly
- Check `module load` commands match your HPC

### GPU Not Available
- Check `sinfo` for available partitions
- Try `--partition=gpu-debug` for quick tests
- Contact HPC admin if no GPUs available

---

## EXPECTED OUTPUTS

After all jobs complete, you should have:

```
~/gxai-elderly/
  manipulation_detector_results/
    manipulation_detector_best.pt
    config.json
  layer3_llama3_results/
    lora_adapter/
      adapter_model.safetensors
      adapter_config.json
  results/
    all_gists_llama3.json          (10,985 gists)
  youtube_validation_results.json   (243 evaluations)
  rl_feedback_results.json          (200 feedback loops)
```

---

## NEXT STEPS AFTER HPC JOBS COMPLETE

1. **Quality Check**: Manually inspect 20-30 generated gists
2. **Human Evaluation**: Recruit evaluators using `human_eval_protocol.md`
3. **Paper Writing**: Start writing results section
4. **Figures**: Generate confusion matrix, per-type accuracy, tactic detection F1

---

## QUESTIONS?

If any job fails, send me:
1. The `.err` log file
2. The `.out` log file
3. The SLURM job ID

I will debug and fix the script.