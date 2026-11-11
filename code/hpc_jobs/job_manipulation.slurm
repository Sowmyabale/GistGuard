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
echo "Results saved to: manipulation_detector_results/"