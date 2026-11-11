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
python evaluate_youtube_pipeline.py \
    --input_dir "../../dataset/Youtube transcripts/converted_json" \
    --output "../../youtube_validation_results.json"

echo "YouTube validation complete!"
echo "Results saved to: youtube_validation_results.json"