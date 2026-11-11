#!/bin/bash
#SBATCH --job-name=generate_gists
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/generate_gists_%j.out
#SBATCH --error=logs/generate_gists_%j.err

# This job should only run after LLaMA-3 training completes
# Uncomment and set JOB_ID after llama3_train is submitted:
#SBATCH --dependency=afterok:JOB_ID_OF_LLAMA3_TRAIN

module load python/3.10
source ~/gxai-elderly/venv/bin/activate

cd ~/gxai-elderly/experiments
python generate_all_gists_hpc.py --model llama3 --output all_gists_llama3.json

echo "Gist generation complete!"
echo "Results saved to: results/all_gists_llama3.json"