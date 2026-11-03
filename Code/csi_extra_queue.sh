#!/bin/bash
# 2026-11-04: extra mode of the channel-aware runs (pca_csi.py), written with the suffix _x so the first
# queue's files stay intact: odsm, Static ODS-k with the same scheduler and the energy d_s/U, on the same
# draws as the first queue (perfect and delayed CSI). CPU only, N_PAR at a time.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
N_PAR=${N_PAR:-12}
J=()
for U in 1 2 3 4 5; do J+=("--U $U --k 6 --nearfar 6 --modes odsm"); J+=("--U $U --k 6 --modes odsm"); done
for U in 1 2 3; do J+=("--U $U --k 4 --nearfar 6 --modes odsm"); done
for d in 0.9 0.7; do for U in 1 2 3 4 5; do J+=("--U $U --k 6 --nearfar 6 --delay $d --modes odsm"); done; done
i=0
for j in "${J[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 10; done
  echo "$(date +%H:%M) start $j" >> logs/pcacsi2_queue.log
  $PY pca_csi.py $j --suffix _x > logs/pca_csi2_$i.log 2>&1 &
  i=$((i+1))
done
wait
echo done >> logs/pcacsi2_queue.log
