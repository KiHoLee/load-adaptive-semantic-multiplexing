#!/bin/bash
# =========================================================
# dyn_queue.sh — runs of the load-adaptive revision (2026-11-03), grouped by stage.
# The commands are the ones run for the paper on one NVIDIA GPU, with the interpreter (PY) and the
# GPU index (GPU, default 0) made configurable; without CUDA the scripts fall back to the CPU.
#   bash dyn_queue.sh main      # Learned nested code (shared_dyns) and its variant without the SNR profile, 3 seeds
#   bash dyn_queue.sh clamp     # re-evaluate the main models with the profile held above 25 dB
#   bash dyn_queue.sh ablate    # ablations, U_max = 8, SST-2, DistilBERT (seed 1; data/dyn_numbers.md, Section 8)
#   bash dyn_queue.sh eval      # no adaptation, robustness, probes, cost, effective width,
#                               # Learned ODS-k with energy d_s/U
#   bash dyn_queue.sh history   # earlier designs reported in the response letter (data/dyn_numbers.md)
#   bash dyn_queue.sh pcainit   # Learned nested code started from the PCA basis (seed 1)
#   bash dyn_queue.sh todma     # re-dimensioned ToDMA, 500 frames per point
#   bash dyn_queue.sh report    # data/dyn_numbers.md from data/ only
# Prerequisites: the embedding caches (python rev_cache.py) and, from rev_queue.sh, the
# Learned ODS-U models ortho_U{1..6}_s{1,2,3} (stage main_ortho), ortho_U8_s1 (stage scale),
# the per-load retrains shared_U{1..6}_s{1,2,3} and Learned shared embedding shared_agn_s{1,2,3}
# (stage main).
# =========================================================
cd "$(dirname "$0")"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
C=../data/ckpt
mkdir -p logs
COMMON="--K 4 --U 6 --U-max 6 --agnostic --epochs 150 --steps 100 --batch 64 --trials 2000 --eval-U 1 2 3 4 5 6 --out-dir ../data"
run() { name=$1; sch=$2; seed=$3; shift 3; $PY rev_run.py --name $name --scheme $sch --seed $seed "$@" > logs/${name}_s$seed.log 2>&1; }

case "$1" in
main)      # shared_dyns: SNR-dependent profile (Learned nested code); shared_dyn: profile without SNR
  for sch in shared_dyn shared_dyns; do
    for s in 1 2 3; do run $sch $sch $s $COMMON; done
  done ;;
clamp)     # writes data/shared_dyns_clamp_s{1,2,3}.json, the Learned nested code of Table I
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
  # Learned ODS-k with energy d_s/U (k = 4 and 6 are written by extra_runs.sh)
  $PY rev_extra.py energy $C/ortho_U1_s1.pt $C/ortho_U1_s2.pt $C/ortho_U1_s3.pt $C/ortho_U2_s1.pt $C/ortho_U2_s2.pt $C/ortho_U2_s3.pt \
      $C/ortho_U3_s1.pt $C/ortho_U3_s2.pt $C/ortho_U3_s3.pt $C/ortho_U5_s1.pt $C/ortho_U5_s2.pt $C/ortho_U5_s3.pt ;;
history)   # idle masks reassigned (seed 1), learned and fixed blocks, and the design with a per-load receive offset
  run shared_abs shared_abs 1 $COMMON --per-user-rx
  run shared_absblk shared_abs 1 $COMMON --per-user-rx --mask-init blocks --freeze mask
  for s in 1 2 3; do run shared_nest shared_nest $s $COMMON; done
  run shared_nest_e300 shared_nest 1 ${COMMON/--epochs 150/--epochs 300} ;;
pcainit)   # writes data/shared_dyns_pcainit_s1.json
  run shared_dyns_pcainit shared_dyns 1 $COMMON --init pca ;;
todma)
  $PY todma_redim.py --frames 500 4,24,128 4,32,96 4,48,64 1,24,128 1,32,96 1,48,64 2,32,96 2,48,64 6,24,128 6,32,96 6,48,64 ;;
report)
  $PY rev_dyn_numbers.py ;;
*) echo "usage: bash dyn_queue.sh {main|clamp|ablate|eval|history|pcainit|todma|report}"; exit 1 ;;
esac
