#!/bin/bash
# pilot_queue.sh — gain-estimation runs of shared_pilot.py (2026-11-16), CPU only. Every receiver estimates its gains: the load masks blind by the manuscript's (8) (all d_u entries) with the
# transmit normalization (5); the pilot-charged Proposition 2 masks by the raw one-use pilot, by the Gaussian-ML EM
# over the whole frame (unnormalized users) and by the energy-matched EM with the same normalization (5) as the load
# masks (*_rho); the unpiloted Proposition 2 masks by both EMs from a common blind gain; the load masks by both EMs
# (like-for-like references); genie references with the true gain, unnormalized (load_pu, load_joint,
# gen_joint_true(0)) and normalized (load_pu_rho, load_joint_rho, gen_joint_true(0)_rho).
# COCO, Rayleigh gains, K in {4, 2}, U in {4, 2}, Ep in {1, 8}, SNR -5 0 10 20, every test image (2000 frames),
# plus one residual-phase point (--phase 10, U 4, K 4, Ep 1). The Ep 8 jobs drop the modes that do not depend on Ep
# (gen_joint_true0(_rho), gen_joint_em0(_rho), load_joint_em(_rho): the same draws give the same numbers frame by frame
# as in the Ep 1 job of the point, so they can be paired across the two files) but keep the cheap load references, so
# the bootstrap CIs against load_joint_blind, load_joint and load_joint_rho are computed in every file.
# EM options (EMOPT, default below): Fisher scoring with the line search on the exact log-likelihood for the
# Gaussian-ML modes, the extrapolated energy-matched step for the *_rho modes, at most 30 iterations (a frame stops at
# a relative change < 1e-4), the exact posterior covariance at -5 and 0 dB and
# the A1 one at 10 and 20 dB (--em_cov auto); the options enter the output tag (_emauto0_fs_it30) and the job name, so
# runs with other options neither overwrite these outputs nor are skipped. Modes: MALL (Ep 1 jobs) and MEP (Ep 8 jobs)
# can be overridden from the environment.
# Outputs ../data/shared/pilot/shared_pilot_<tag>[_snr<S>].json/.npz, logs logs/pilot/<name>.log. N_PAR processes at
# a time (default 2), THREADS BLAS threads each (default 2), nice 10; longest jobs first. A job whose logs/pilot/<name>.done exists is
# skipped, so the queue can be restarted. FR sets --frames and adds the output suffix _fr<FR> (also in the job name),
# so a trial run neither overwrites the full outputs nor marks them done; SUF adds an output suffix. SPLIT=1 (default, as for
# the stored files) runs every point as one job per SNR (output suffix _snr<SNR>; the SNRs are independent in the script, the same draws), which
# shortens the longest job from about 18.5 h to about 7 h. The md5 of shared_pilot.py and shared_masks.py is written to
# logs/pilot_queue.log at start (and into every output JSON).
# Cost at 2 threads per process: about 59 process-hours in all; the longest job (U = K = 4, Ep = 1) takes about
# 18.5 h, or about 7 h per SNR with SPLIT=1; the exact posterior covariance at 0 dB costs about 2.5 h of it.
# Usage (from the repository root): N_PAR=4 bash Code/pilot_queue.sh
#   (trial: FR=200 ...; other EM options: EMOPT="--em_step em --em_iters 3" ...)
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
THREADS=${THREADS:-2}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
N_PAR=${N_PAR:-2}
SPLIT=${SPLIT:-1}
EMOPT=${EMOPT:-"--em_step fisher --em_iters 30 --em_cov auto"}
MALL=${MALL:-"load_pu load_joint load_pu_rho load_joint_rho load_pu_blind load_joint_blind gen_joint_true gen_joint_true0 gen_joint_true_rho gen_joint_true0_rho gen_joint_pilot gen_joint_em gen_joint_em0 load_joint_em gen_joint_em_rho gen_joint_em0_rho load_joint_em_rho"}
MEP=${MEP:-"load_pu load_joint load_pu_rho load_joint_rho load_pu_blind load_joint_blind gen_joint_true gen_joint_true_rho gen_joint_pilot gen_joint_em gen_joint_em_rho"}
if [ "$SPLIT" = 1 ]; then SNRL="20 10 0 -5"; else SNRL="all"; fi
J=()
for S in $SNRL; do
  if [ "$S" = all ]; then SN="--snrs -5 0 10 20"; SX="${FR:+_fr$FR}${SUF:-}"; else SN="--snrs $S"; SX="_snr$S${FR:+_fr$FR}${SUF:-}"; fi
  FRA="${FR:+--frames $FR} ${SX:+--suffix $SX}"
  J+=("--dataset coco --U 4 --K 4 --pilot_energy 1 $EMOPT $SN $FRA --modes $MALL")
  J+=("--dataset coco --U 4 --K 4 --pilot_energy 1 --phase 10 $EMOPT $SN $FRA --modes $MALL")
  J+=("--dataset coco --U 4 --K 4 --pilot_energy 8 $EMOPT $SN $FRA --modes $MEP")
done
for S in $SNRL; do
  if [ "$S" = all ]; then SN="--snrs -5 0 10 20"; SX="${FR:+_fr$FR}${SUF:-}"; else SN="--snrs $S"; SX="_snr$S${FR:+_fr$FR}${SUF:-}"; fi
  FRA="${FR:+--frames $FR} ${SX:+--suffix $SX}"
  for UK in "4 2" "2 4" "2 2"; do
    set -- $UK
    J+=("--dataset coco --U $1 --K $2 --pilot_energy 1 $EMOPT $SN $FRA --modes $MALL")
    J+=("--dataset coco --U $1 --K $2 --pilot_energy 8 $EMOPT $SN $FRA --modes $MEP")
  done
done
mkdir -p logs/pilot ../data/shared/pilot
echo "$(date '+%m-%d %H:%M') start pilot queue, ${#J[@]} jobs, $N_PAR at a time, SPLIT=$SPLIT, EMOPT=$EMOPT" >> logs/pilot_queue.log
(md5sum shared_pilot.py shared_masks.py 2>/dev/null || md5 -r shared_pilot.py shared_masks.py) >> logs/pilot_queue.log
for j in "${J[@]}"; do
  name=pilot_$(echo "$j" | sed -E 's/--modes.*//; s/--snrs( -?[0-9]+)+//; s/--frames [0-9]+//; s/--//g; s/ +/_/g; s/_+$//')
  [ -f "logs/pilot/$name.done" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 30; done
  echo "$(date '+%m-%d %H:%M') start $name" >> logs/pilot_queue.log
  ( nice -n 10 $PY shared_pilot.py $j > "logs/pilot/$name.log" 2>&1 && touch "logs/pilot/$name.done"
    echo "$(date '+%m-%d %H:%M') end $name ($?)" >> logs/pilot_queue.log ) &
  sleep 5
done
wait
echo "$(date '+%m-%d %H:%M') pilot queue done" >> logs/pilot_queue.log
