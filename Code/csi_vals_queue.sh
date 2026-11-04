#!/bin/bash
# 2026-11-04: mask support against mask values (pca_csi.py modes fixedb, csiqw, csiqb), CPU only.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
for U in 1 2 3 4 5 6; do
  for nf in "--nearfar 6" ""; do
    $PY pca_csi.py --U $U --k 6 $nf --modes fixed fixedb csiq csiqw csiqb --suffix _vals > logs/vals_U${U}_${nf:+nf6}.log 2>&1 &
  done
done
wait
