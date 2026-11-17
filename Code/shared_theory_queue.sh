#!/bin/bash
# shared_theory_queue.sh — analysis lines of the shared-content runs (shared_theory.py, 2026-11-16), exact Gaussian NMSE of load_pu, load_joint,
# gen_joint, gen_eq, gen_k0 with the full fitted C and with (A1), equal gains and 100 Rayleigh draws (seed 7),
# SNR -10..20 dB. Outputs ../data/shared/theory_<ds>[_mix<k>]_K<K>_U<U>.json (+ .npz).
# CPU only, THREADS BLAS threads per run (default 2), nice 10, N_PAR runs at a time (default 2). Measured cost (2
# threads, all five schemes, both models, one SNR, 100 draws): U = 2 about 25 s, U = 3 about 50 s, U = 4 about 85 s,
# U = 5 about 140-185 s (dense solves of dimension 3840); per run of 7 SNRs 3 / 6 / 10 / 20 min; whole queue (36 runs)
# about 5 h CPU, about 2.5 h wall at N_PAR = 2. Order: COCO (main figure), mixtures, Flickr.
# Run content_stats.py first (about 70 s) if ../data/shared/content_stats.json is missing (the queue does it).
# Usage (from the repository root): N_PAR=2 bash Code/shared_theory_queue.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
THREADS=${THREADS:-2}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
N_PAR=${N_PAR:-2}
mkdir -p logs
JOBS=()
for K in 4 2; do for U in 5 4 3 2; do JOBS+=("--ds coco --K $K --U $U"); done; done
for K in 4 2; do for k in 0 0.2 0.4 0.6 0.8 0.9 0.95; do JOBS+=("--ds coco --mix $k --K $K --U 4"); done; done
for K in 4 2; do for U in 5 4 3 2; do JOBS+=("--ds flickr --K $K --U $U"); done; done
[ -f ../data/shared/content_stats.json ] || nice -n 10 $PY content_stats.py > logs/content_stats.log 2>&1
echo "$(date +%H:%M) start theory queue, ${#JOBS[@]} runs, $N_PAR at a time"
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 20; done
  name=theory_$(echo "$j" | sed -E 's/--ds //; s/ --mix /_mix/; s/ --K /_K/; s/ --U /_U/')
  echo "$(date +%H:%M) start $name"
  nice -n 10 $PY shared_theory.py $j > logs/${name}.log 2>&1 &
  sleep 5
done
wait
echo "$(date +%H:%M) all done"
