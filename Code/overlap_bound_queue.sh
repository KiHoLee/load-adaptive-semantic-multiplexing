#!/bin/bash
# overlap_bound_queue.sh — numerical search against the closed-form masks (overlap_bound.py). CPU only, THREADS
# BLAS threads per run (default 2), nice 10, N_PAR (default 2) runs at a time.
# Stage original (2026-11-04): AG News, load masks against any real overlapping masks, U = 2, 4, 6, 8, equal gains and
#   offsets spread over +-6 dB, 0/10/20 dB, Adam 3000 steps, 3 random starts -> ../data/extra/overlap_bound_U<U>_nf<X>.json
# Stage run2 (2026-11-16, generalized real-valued masks):
# Every run: Adam 3000 steps (in energy units W = A sqrt(lam)) from the closed form + 3 random starts (+ the
# non-overlapping member, key disjoint in the JSON), joint LMMSE receiver. Outputs ../data/extra/overlap_bound2_<case>.json, logs
# logs/obound2_<case>.log (<case> as the script names it: gains equal | rayleigh-D20 | slot-rho0.7-D20).
#   (0)   sanity: indep content (kappa = 0), equal gains, U 2/4, K 4: the search must not beat the load masks
#   (i)   mix:0.5 and coco with (A1), equal gains, U 2/4, K 4: Proposition 2 vs search
#   (iv)  coco, overload K 2, U 3/4: pruned Proposition 2 (free placement, at most d_s modes) vs
#         search; at 10 dB the kept modes do not fit the original coefficient-to-use-index split (13 > 12 uses
#         in some problem), so closed and search run with the coefficients relabeled over the 128 problems
#         (split "lpt" in the JSON; load and the non-overlapping member keep the original split)
#   (iii) coco, rayleigh:20 (gains known to the search), U 2/4, K 4: gap of Proposition 2 at the unit design gain
#   (v)   indep, slot:0.7 (20 draws), offsets 0/6 dB, U 2/4, K 4: channel-aware masks (13) on the greedy
#         support (csiq) vs overlapping search (same per-user energies n_u x 128); the search may also move a
#         user to other slots: search_tx_outside_closed > 0 with search_shared_energy ~ 0 is a better slot
#         assignment than the greedy support, not overlap
# SNRs: -5 0 10 for (0), (i), (iii), (iv); 0 10 20 for (v) (the SNRs of the original overlap_bound runs).
# CPU time per SNR point at 3000 steps (two runs side by side): equal gains U 2
# 20-25 s, U 4 ~45 s; K 2 U 3 ~30 s, U 4 ~45 s (incl. 3-7 s pruning at 10 dB); rayleigh:20 U 2 ~6.5 min,
# U 4 ~14 min; slot:0.7 U 2 ~7.5 min, U 4 ~16 min. Total ~3.6 h CPU, ~2 h wall at N_PAR = 2.
# Usage (from the repository root): bash Code/overlap_bound_queue.sh [original|run2|all]   (default all)
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
THREADS=${THREADS:-2}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
STAGE=${1:-all}
N_PAR=${N_PAR:-2}
SA="--snrs -5 0 10"
SV="--snrs 0 10 20"
JOBS=()
if [ "$STAGE" = original ] || [ "$STAGE" = all ]; then
  for U in 2 4 6 8; do for nf in 0 6; do JOBS+=("original_U${U}_nf${nf}|--U $U --nearfar $nf"); done; done
fi
[ "$STAGE" = run2 ] || [ "$STAGE" = all ] && JOBS+=(
  # short runs first (results within ~20 min)
  "indep_equal_K4_U2|--content indep --gains equal --K 4 --U 2 $SA"
  "indep_equal_K4_U4|--content indep --gains equal --K 4 --U 4 $SA"
  "mix0.5_equal_K4_U2|--content mix:0.5 --gains equal --K 4 --U 2 $SA"
  "mix0.5_equal_K4_U4|--content mix:0.5 --gains equal --K 4 --U 4 $SA"
  "coco_equal_K4_U2|--content coco --gains equal --K 4 --U 2 $SA"
  "coco_equal_K4_U4|--content coco --gains equal --K 4 --U 4 $SA"
  "coco_equal_K2_U3|--content coco --gains equal --K 2 --U 3 $SA"
  "coco_equal_K2_U4|--content coco --gains equal --K 2 --U 4 $SA"
  # long runs, longest first
  "indep_slot-rho0.7-D20_K4_U4_nf6|--content indep --gains slot:0.7 --K 4 --U 4 --nearfar 6 $SV"
  "indep_slot-rho0.7-D20_K4_U4|--content indep --gains slot:0.7 --K 4 --U 4 $SV"
  "coco_rayleigh-D20_K4_U4|--content coco --gains rayleigh:20 --K 4 --U 4 $SA"
  "indep_slot-rho0.7-D20_K4_U2_nf6|--content indep --gains slot:0.7 --K 4 --U 2 --nearfar 6 $SV"
  "indep_slot-rho0.7-D20_K4_U2|--content indep --gains slot:0.7 --K 4 --U 2 $SV"
  "coco_rayleigh-D20_K4_U2|--content coco --gains rayleigh:20 --K 4 --U 2 $SA"
)
mkdir -p logs
echo "$(date +%H:%M) start overlap_bound queue ($STAGE), ${#JOBS[@]} runs, $N_PAR at a time"
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 20; done
  name=${j%%|*}; args=${j#*|}
  echo "$(date +%H:%M) start $name: $args"
  nice -n 10 $PY overlap_bound.py $args > logs/obound2_${name}.log 2>&1 &
  sleep 5
done
wait
echo "$(date +%H:%M) all done"
