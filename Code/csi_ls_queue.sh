#!/bin/bash
# 2026-11-04: local search over the slot counts and positions after the greedy grant (pca_csi_ls.py), CPU only.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for U in 2 3 4 5 6; do
  for snr in 0 10 20; do
    $PY pca_csi_ls.py --U $U --nearfar 6 --snrs $snr --maxit 20 > logs/ls_U${U}_${snr}.log 2>&1 &
  done
done
wait
