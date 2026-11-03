#!/bin/bash
# 2026-11-04: channel-aware dynamic masks of the closed-form code (pca_csi.py), CPU only, N_PAR at a time.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
N_PAR=${N_PAR:-14}
J=()
for U in 1 2 3 4 5 6; do J+=("--U $U --k 6 --nearfar 6"); J+=("--U $U --k 6"); done
for U in 1 2 3 4; do J+=("--U $U --k 4 --nearfar 6 --modes ods odsfix"); done
for d in 0.9 0.7; do for U in 1 2 3 4 5 6; do J+=("--U $U --k 6 --nearfar 6 --delay $d --modes fixed csi csiq ods"); done; done
J+=("--U 4 --k 6 --nearfar 6 --genie")
for r in 0.3 0.95; do J+=("--U 4 --k 6 --nearfar 6 --rho $r --modes fixed csi csiq ods"); done
i=0
for j in "${J[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 10; done
  $PY pca_csi.py $j > logs/pca_csi_$i.log 2>&1 &
  i=$((i+1))
done
wait
echo done
