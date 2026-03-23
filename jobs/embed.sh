#!/bin/bash
#SBATCH --job-name=esm2-embed
#SBATCH --partition=shared-gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/embed-%j.out

cd ~/work/amp_proj
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm-amp
python src/embed.py
