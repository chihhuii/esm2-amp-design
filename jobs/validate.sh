#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --partition=shared-gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/validate-%j.out

cd ~/work/amp_proj
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm-amp
python src/validate.py
