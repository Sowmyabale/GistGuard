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
echo "Results saved to: rl_feedback_results.json"