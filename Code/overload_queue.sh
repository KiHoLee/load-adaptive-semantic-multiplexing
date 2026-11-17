#!/bin/bash
# overload_queue.sh — load sweep of the shared-content masks from light load through overload (2026-11-16), K = 4
# (d_s = 3072, full load U = 4), U = 2..8. Shows how the masks follow the load: below and at full load the overlap of
# Proposition 2 helps only at low SNR; beyond full load the masks of correlated users keep overlapping at high SNR.
# 1) shared_theory.py (Gaussian model, exact NMSE, 100 Rayleigh draws): --mix 0.8 and 0.4 at U = 2 3 5 6 7 8 (U = 4
#    comes from shared_theory_queue.sh; mix 0 gives exactly zero overlap gain at U = 4 for K = 4 and K = 2), SNR -10 -5
#    0 10 20. Outputs ../data/shared/theory_coco_mix<k>_K4_U<U>.json. Cost about U^3: U = 8 about 1 h per run.
# 2) shared_masks.py on the caption mixtures (real BERT embeddings, every test image): --mix 0.8 at U = 2..8 and
#    --mix 0 at U = 8, SNR -5 0 10 20, modes load_pu load_joint gen_joint, suffix _ol (no clash with the
#    shared queue). Outputs ../data/shared/shared_coco_mix<k>_K4_U<U>_ol.json/.npz.
# The U = 7 jobs (theory 0.8 and 0.4, masks 0.8) ran in a second pass with the same arguments; the order below
# differs from the stored run order, the outputs do not depend on it.
# CPU only, THREADS BLAS threads per run (default 2), nice 10, N_PAR runs at a time (default 2). A job whose
# logs/overload/<name>.done exists is skipped, so the queue can be restarted.
# Usage (from the repository root): N_PAR=2 bash Code/overload_queue.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
THREADS=${THREADS:-2}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
N_PAR=${N_PAR:-2}
J=()
for U in 2 3 5 6 7 8; do J+=("theory|--ds coco --mix 0.8 --K 4 --U $U --snrs -10 -5 0 10 20"); done
for U in 2 3 4 5 6 7 8; do J+=("masks|--dataset coco --mix 0.8 --K 4 --U $U --snrs -5 0 10 20 --suffix _ol --modes load_pu load_joint gen_joint"); done
for U in 2 3 5 6 7 8; do J+=("theory|--ds coco --mix 0.4 --K 4 --U $U --snrs -10 -5 0 10 20"); done
J+=("masks|--dataset coco --mix 0 --K 4 --U 8 --snrs -5 0 10 20 --suffix _ol --modes load_pu load_joint gen_joint")
mkdir -p logs/overload
echo "$(date '+%m-%d %H:%M') start overload queue, ${#J[@]} jobs, $N_PAR at a time" >> logs/overload_queue.log
for e in "${J[@]}"; do
  kind=${e%%|*}; j=${e#*|}
  name=${kind}_$(echo "$j" | sed -E 's/--modes.*//; s/ --snrs( -?[0-9]+)+//; s/--//g; s/ +/_/g; s/_+$//')
  [ -f "logs/overload/$name.done" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 30; done
  if [ "$kind" = theory ]; then script=shared_theory.py; else script=shared_masks.py; fi
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/overload_queue.log
  ( nice -n 10 $PY $script $j > "logs/overload/$name.log" 2>&1 && touch "logs/overload/$name.done"
    echo "$(date '+%m-%d %H:%M') end $name ($?)" >> logs/overload_queue.log ) &
  sleep 5
done
wait
echo "$(date '+%m-%d %H:%M') overload queue done" >> logs/overload_queue.log
