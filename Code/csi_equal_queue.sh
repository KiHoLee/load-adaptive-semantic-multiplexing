#!/bin/bash
# 2026-11-04: equal-count control of the channel-aware runs (pca_csi.py --modes csiqe), written with the
# suffix _e on the draws of the first queue: greedy placement with every slot count capped at that of the
# load masks, which separates the free slot counts from the greedy placement. CPU only, all in parallel.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for U in 2 3 4 5 6; do
  $PY pca_csi.py --U $U --k 6 --nearfar 6 --modes csiqe --suffix _e > logs/pca_csi_e$U.log 2>&1 &
done
wait
