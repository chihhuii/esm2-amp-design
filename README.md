# esm2-amp-design

De novo antimicrobial peptide (AMP) candidate generation using ESM2 embeddings and ProteinMPNN.

## Overview

This project implements a generative pipeline for designing novel AMP candidates:

1. **Embed** — Extract sequence representations using ESM2-650M
2. **Cluster** — Identify AMP subfamilies via UMAP + HDBSCAN
3. **Generate** — Design new sequences with ProteinMPNN conditioned on target cluster structure
4. **Validate** — Verify generated sequences in ESM2 embedding space

## Results

- 2,963 AMP sequences embedded (DBAASP, ribosomal, non-hemolytic)
- 6 clusters identified; Cluster 1 = cysteine-rich defensin-like family (71 sequences)
- 150 novel candidate sequences generated from Human Defensin 5 (PDB: 2LXZ)
- Generated sequences form a coherent new region in ESM2 embedding space

## Usage

### 1. Embed sequences
sbatch jobs/embed.sh

### 2. Cluster
python src/cluster.py

### 3. Generate with ProteinMPNN
sbatch jobs/mpnn.sh

### 4. Validate
sbatch jobs/validate.sh

## Data

Download AMP sequences from DBAASP (https://dbaasp.org) with filters:
- Synthesis Type: Ribosomal
- Target: Gram+, Gram-
- Hemolytic Activity: Non-hemolytic

Place files in data/.

## Requirements

torch
transformers
umap-learn
hdbscan
matplotlib
seaborn
numpy

## Limitations & Next Steps

**v0 baseline limitations:**
- ProteinMPNN input structure (2LXZ) is not a perfect representative of Cluster 1
- Generated sequences land near but outside Cluster 1 boundary

**v1 roadmap:**
- Use Cluster 1 native structures as ProteinMPNN input for tighter generation
- Add physicochemical filter (net charge, hydrophobicity)
- Streamlit demo for interactive sequence scoring

## References

- ESM2: Lin et al. (2023) Science
- ProteinMPNN: Dauparas et al. (2022) Science
- DBAASP: Pirtskhalava et al. (2021) Antimicrob Agents Chemother
