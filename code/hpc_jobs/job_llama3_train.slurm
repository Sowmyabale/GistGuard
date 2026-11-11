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
echo "Results saved to: layer3_llama3_results/lora_adapter/"