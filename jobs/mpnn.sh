#!/bin/bash
#SBATCH --job-name=mpnn-amp
#SBATCH --partition=shared-gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/mpnn-%j.out

cd ~/work/amp_proj
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlfold

python ~/work/ProteinMPNN/protein_mpnn_run.py \
    --pdb_path data/pdb/2LXZ.pdb \
    --pdb_path_chains A \
    --out_folder outputs/mpnn_results \
    --path_to_model_weights ~/work/ProteinMPNN/vanilla_model_weights \
    --model_name v_48_020 \
    --num_seq_per_target 50 \
    --sampling_temp "0.1 0.2 0.3" \
    --batch_size 1
