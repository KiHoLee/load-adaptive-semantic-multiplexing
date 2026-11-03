#!/bin/bash
# =========================================================
# dyn_queue.sh — runs of the load-adaptive revision (2026-11-03), grouped by stage.
# The commands are the ones run for the paper on one NVIDIA GPU (Code/u24_*_queue.sh in the
# working folder), with the interpreter and the device made configurable.
#   bash dyn_queue.sh main      # Proposed (one model) and its SNR-independent profile, 3 seeds
#   bash dyn_queue.sh clamp     # re-evaluate the main models with the profile held above 25 dB
#   bash dyn_queue.sh ablate    # Table II ablations, U_max = 8, SST-2, DistilBERT (seed 1)
#   bash dyn_queue.sh eval      # no adaptation, robustness, probes, cost, effective width,
#                               # Static ODS-k with energy d_s/U
#   bash dyn_queue.sh history   # earlier designs reported in Table II and the response letter
#   bash dyn_queue.sh todma     # re-dimensioned ToDMA, 500 frames per point
#   bash dyn_queue.sh report    # data/dyn_numbers.md and Figs. 2, 3, R1.1 from data/ only
# Prerequisites: the embedding caches (python rev_cache.py) and, from rev_queue.sh, the
# Static ODS-k models ortho_U{1..6}_s{1,2,3} (stage main_ortho), ortho_U8_s1 (stage scale),
# the per-load retrains shared_U{1..6}_s{1,2,3} and Overlapping masks shared_agn_s{1,2,3}
# (stage main).
# =========================================================
cd "$(dirname "$0")"
PY=${PY:-python}
C=../data/ckpt
mkdir -p logs
COMMON="--K 4 --U 6 --U-max 6 --agnostic --epochs 150 --steps 100 --batch 64 --trials 2000 --eval-U 1 2 3 4 5 6 --out-dir ../data"
run() { name=$1; sch=$2; seed=$3; shift 3; $PY rev_run.py --name $name --scheme $sch --seed $seed "$@" > logs/${name}_s$seed.log 2>&1; }

case "$1" in
main)      # shared_dyns: SNR-dependent profile (the proposal); shared_dyn: profile without SNR
  for sch in shared_dyn shared_dyns; do
    for s in 1 2 3; do run $sch $sch $s $COMMON; done
  done ;;
clamp)     # writes data/shared_dyns_clamp_s{1,2,3}.json, the proposal of every figure and table
  $PY rev_extra.py reeval $C/shared_dyns_s1.pt $C/shared_dyns_s2.pt $C/shared_dyns_s3.pt --U 1 2 3 4 5 6 --suffix _clamp ;;
ablate)
  run shared_dyns_freezetx shared_dyns 1 $COMMON --freeze tx
  run shared_dyns_uniform  shared_dyns 1 $COMMON --freeze profile
  run shared_dyns_joint    shared_dyns 1 $COMMON --mode joint
  run shared_dyns_K1       shared_dyns 1 $COMMON --K 1
  run shared_dyns_K8       shared_dyns 1 $COMMON --K 8
  run shared_dyns_slots12  shared_dyns 1 $COMMON --n-slots 12
  run shared_dyns_slots48  shared_dyns 1 $COMMON --n-slots 48
  run shared_dyns_U8       shared_dyns 1 --K 4 --U 8 --U-max 8 --agnostic --epochs 150 --steps 100 --batch 64 --trials 2000 --eval-U 1 2 3 4 5 6 7 8 --out-dir ../data
  run sst2_shared_dyns     shared_dyns 1 $COMMON --dataset sst2 --probe
  run distil_shared_dyns   shared_dyns 1 $COMMON --model distilbert-base-uncased ;;
eval)
  $PY rev_extra.py noadapt $C/shared_dyns_s1.pt $C/shared_dyns_s2.pt $C/shared_dyns_s3.pt --U 1 2 3 4 5 6
  $PY rev_extra.py robust  $C/shared_dyns_s1.pt --U 1 4
  $PY rev_extra.py probe   $C/shared_dyns_s1.pt --U 1 4 6
  $PY rev_extra.py cost    $C/shared_dyns_s1.pt $C/ortho_U4_s1.pt $C/ortho_U6_s1.pt $C/shared_agn_s1.pt --U 4
  $PY rev_extra.py width   $C/shared_dyns_s1.pt $C/shared_dyns_s2.pt $C/shared_dyns_s3.pt $C/shared_dyn_s1.pt
  # Static ODS-k with energy d_s/U (k = 4 and 6 are written by extra_runs.sh)
  $PY rev_extra.py energy $C/ortho_U1_s1.pt $C/ortho_U1_s2.pt $C/ortho_U1_s3.pt $C/ortho_U2_s1.pt $C/ortho_U2_s2.pt $C/ortho_U2_s3.pt \
      $C/ortho_U3_s1.pt $C/ortho_U3_s2.pt $C/ortho_U3_s3.pt $C/ortho_U5_s1.pt $C/ortho_U5_s2.pt $C/ortho_U5_s3.pt ;;
history)   # idle masks reassigned (Table II, seed 1) and the design with a per-load receive offset
  run shared_abs shared_abs 1 $COMMON --per-user-rx
  for s in 1 2 3; do run shared_nest shared_nest $s $COMMON; done ;;
todma)
  $PY todma_redim.py --frames 500 4,24,128 4,32,96 4,48,64 1,24,128 1,32,96 1,48,64 2,32,96 2,48,64 6,24,128 6,32,96 6,48,64 ;;
report)
  $PY rev_dyn_numbers.py
  $PY rev_dynamic_fig.py ;;
*) echo "usage: bash dyn_queue.sh {main|clamp|ablate|eval|history|todma|report}"; exit 1 ;;
esac
