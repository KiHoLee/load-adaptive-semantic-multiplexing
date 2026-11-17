#!/usr/bin/env bash
# Evaluations of stored checkpoints (no training): Learned ODS-4 and Learned ODS-6 with the energy d_s/U
# (data/extra/energy_ortho_U{4,6}_s*.json) and the division of the transmitted energy by the learned masks
# (data/extra/mask_partition.json, read by pca_numbers.py). Run after rev_queue.sh has produced the checkpoints
# in ../data/ckpt/.  Usage (from the repository root): bash Code/extra_runs.sh
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=${PY:-python}
C=../data/ckpt
mkdir -p ../data/extra
# Learned ODS-4 and Learned ODS-6 at the per-user energy d_s/U
$PY rev_extra.py energy $C/ortho_U4_s1.pt $C/ortho_U4_s2.pt $C/ortho_U4_s3.pt $C/ortho_U6_s1.pt $C/ortho_U6_s2.pt $C/ortho_U6_s3.pt
# Division of the transmitted energy by the learned masks
$PY mask_partition.py
