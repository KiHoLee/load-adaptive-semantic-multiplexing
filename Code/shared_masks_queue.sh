#!/bin/bash
# shared_masks_queue.sh — runs of shared_masks.py for users that describe one scene (2026-11-16), CPU only.
# N_PAR processes at a time (default 2), THREADS BLAS threads each (default 2), nice 10; longest jobs first. A job whose
# logs/shared/<name>.done exists is skipped, so the queue can be restarted. FR sets --frames for the
# full-test-set jobs (default: every test image, coco 2000, flickr 1000) and then adds the output suffix _fr<FR>,
# which also enters the job name, so a trial run with FR neither overwrites the full outputs nor marks the full
# jobs done; the rnd jobs always use 500 frames (suffix _rnd). All draws are prefix-stable in the frame count.
# Usage (from the repository root): N_PAR=2 bash Code/shared_masks_queue.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
THREADS=${THREADS:-2}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
N_PAR=${N_PAR:-2}
FRA=${FR:+--frames $FR --suffix _fr$FR}
M6="load_pu load_joint gen_joint gen_eq gen_k0 gen_pc"
J=()
# 1) COCO grid: U = 2..5, K = 4 and 2, SNR -10..20 (default --snrs -10 -5 0 5 10 20)
for U in 5 4 3 2; do for K in 4 2; do J+=("--dataset coco --U $U --K $K $FRA --modes $M6"); done; done
# 2) control frames (captions of U distinct images): C = 0 at the design and receiver, and the fitted C (mismatch)
for K in 4 2; do
  J+=("--dataset coco --U 4 --K $K --control $FRA --modes $M6")
  J+=("--dataset coco --U 4 --K $K --control --control_c fit $FRA --modes $M6")
done
# 3) Flickr30k
for K in 4 2; do J+=("--dataset flickr --U 4 --K $K --snrs -5 0 10 $FRA --modes $M6"); done
# 4) controlled mixture kappa
for k in 0 0.2 0.4 0.6 0.8 0.9 0.95; do for K in 4 2; do
  J+=("--dataset coco --mix $k --U 4 --K $K --snrs 0 10 $FRA --modes $M6")
done; done
# 5) robustness (COCO, U = 4): residual phase with |h| or the effective gain at the receiver; C from the other
#    corpus or from 100 / 1000 fit images; pilot-based gains for the overlapping members (one real pilot use per
#    user, energy Ep = 1 (stress case: gain-error variance = s2), 8 and 32 taken from d_s/U; the load masks are not
#    charged and use the effective gain, so load_pu / load_joint equal the grid run frame by frame) with the
#    genie-gain reference gen_joint_true (the same pilot-charged masks, true gain)
MP="load_pu load_joint gen_joint gen_joint_true gen_pc"
for K in 4 2; do
  for ph in 10 30; do for rx in abs eff; do
    J+=("--dataset coco --U 4 --K $K --snrs -5 0 10 --phase $ph --rxgain $rx $FRA --modes $M6")
  done; done
  # the gain-estimation runs are in pilot_queue.sh (shared_pilot.py)
  for cf in other 100 1000; do J+=("--dataset coco --U 4 --K $K --snrs -5 0 10 --c_from $cf $FRA --modes $M6"); done
done
# 5b) U = 5 load masks with the manuscript's rank counts (pca_eval.per_load: 640,640,640,640,512 at K = 4 and
#     384,384,256,256,256 at K = 2; suffix _rank), next to the default floor(d_s/U) of the grid (same draws)
for K in 4 2; do J+=("--dataset coco --U 5 --K $K --load_rule rank $FRA --modes load_pu load_joint"); done
# 6) dense random masks (U <= K), independent (AG News) and shared (COCO) content, 500 frames; suffix _rnd so the
#    COCO outputs do not overwrite shared_coco_K4_U<U>.json of the grid
for ds in agnews coco; do for U in 4 2; do
  J+=("--dataset $ds --U $U --K 4 --snrs 0 10 20 --frames 500 --suffix _rnd --modes load_pu load_joint gen_joint rnd")
done; done
mkdir -p logs/shared
echo "$(date '+%m-%d %H:%M') start shared queue, ${#J[@]} jobs, $N_PAR at a time" >> logs/shared_queue.log
for j in "${J[@]}"; do
  name=$(echo "$j" | sed -E 's/--modes.*//; s/--frames [0-9]+//; s/--//g; s/ +/_/g; s/_+$//')
  [ -f "logs/shared/$name.done" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 30; done
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/shared_queue.log
  ( nice -n 10 $PY shared_masks.py $j > "logs/shared/$name.log" 2>&1 && touch "logs/shared/$name.done"
    echo "$(date '+%m-%d %H:%M') end $name ($?)" >> logs/shared_queue.log ) &
  sleep 5
done
wait
echo "$(date '+%m-%d %H:%M') shared queue done" >> logs/shared_queue.log
