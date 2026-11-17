#!/bin/bash
# =========================================================
# coherence_queue.sh — coherence sweep of the channel-aware masks (pca_csi.py --coh), 2026-11-16.
# Model: per user an AR(1) complex Gaussian gain over the N = 24 subbands (adjacent correlation rho_f), gains
# reported one block earlier, h | h' ~ CN(eta h', (1 - eta^2) R_F), [R_F]_{ss'} = rho_f^{|s-s'|} (stationary
# innovation, the default of pca_csi.py), optional near-far offsets uniform in +-6 dB.
# Schemes: fixed (load masks), csiq, switch (coherence-aware choice csiq/fixed, 32 conditional samples),
#   csiqp (conditional-mean gains), ods (Static ODS-k with k = U, energy d_s/k = d_s/U), csiqb and csiqw (support
#   of csiq with equal values / with the unit-gain values (12): support vs values). Not run because identical to
#   ods at k = U for U in {1, 2, 4, 6} (U divides N): odsm (energy d_s/U = d_s/k) and csi (quotas N/U = N/k, same
#   Hungarian sets and energy). At U = 1, ods also equals csiq. SNR 0, 10, 20 dB; FRAMES frames (default 2000);
#   seeds 123 + SNR (draws), 999 + SNR (switch samples), 555 + SNR (bootstrap CIs).
# Order: (1) reproduction gate: rho_f 0.7, eta 1, +-6 dB, U = 1 and 4; fixed/csiq/csi at 0 dB must equal the
#   stored data/pilot_corr/pca_csi_U<U>_k6_rho0.7_nf6.json within 1e-9 (2 threads change the last bit only),
#   otherwise the queue stops. (2) +-6 dB offsets, eta = 1, rho_f in {0.3, 0.5, 0.7, 0.9, 0.95, 1.0}.
#   (3) +-6 dB offsets, eta in {0.9, 0.7, 0.5, 0.3, 0.0} at rho_f in {0.3, 0.7, 0.95}. (2) and (3) for
#   U in {6, 4, 2, 1}, largest U first (84 runs with the gate). (4) No offsets, rho_f = 0.7, eta in
#   {1.0, 0.9, 0.7, 0.5, 0.3, 0.0}, U in {4, 1} (12 runs). (5) No offsets, rho_f = 1.0, eta = 1, U in {4, 1},
#   blind and with known gains (--genie): tests the rotation-equivalence statement (4 runs). 100 runs in all.
# A run is skipped only if its JSON and .npz exist and the JSON args match (frames, k = U, SNRs, nsamp, genie,
# stationary innovation, all modes); otherwise it is rerun and pca_csi.py moves the old files to
# ../data/pilot_corr/old/ first (nothing is overwritten), so the queue can be restarted.
# CPU only, THREADS BLAS threads (default 2) and nice 10 per run, N_PAR runs at a time (default 2). The stored
# pca_csi runs used 1 thread; THREADS=1 reproduces them bit for bit.
# Estimated time per run at 2000 frames (2 threads, measured at 200 frames): eta < 1 about 1150 / 870 /
# 490 / 270 s and eta = 1 about 630 / 480 / 280 / 160 s for U = 6 / 4 / 2 / 1; about 16 CPU-hours in all,
# about 8 h wall at N_PAR = 2.
# Output: ../data/pilot_corr/pca_csi_<tag>.json and .npz (per-frame CosSim of every scheme, see pca_csi.py),
#   tag = coh_rho<rho>_eta<eta>_U<U>[_nf6][_genie]; logs logs/<tag>.log, progress (start, done with exit code and
#   wall time, skips, gate) logs/coh_queue.log.
# Environment: PY (python), N_PAR, FRAMES, THREADS; DRY=1 prints the decisions (and the gate check of existing
#   files) without running; POINTS="U rho eta nf flag;..." (flag - or genie) replaces the point list and
#   turns the gate off unless GATE=1.
# Usage: PY=/path/to/python N_PAR=2 bash coherence_queue.sh
# =========================================================
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
N_PAR=${N_PAR:-2}
FRAMES=${FRAMES:-2000}
THREADS=${THREADS:-2}
DRY=${DRY:-0}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
MODES="fixed csiq switch csiqp ods csiqb csiqw"
OUT=../data/pilot_corr
mkdir -p logs

log() {
  if [ "$DRY" = 1 ]; then echo "DRY $*"; else echo "$(date '+%m-%d %H:%M') $*" >> logs/coh_queue.log; fi
}

GATE_PTS=("1 0.7 1.0 6 -" "4 0.7 1.0 6 -")      # "U rho_f eta nearfar flag"
if [ -n "${POINTS+x}" ]; then
  IFS=';' read -ra J <<< "$POINTS"; GATE=${GATE:-0}
else
  GATE=${GATE:-1}
  J=()
  for U in 6 4 2 1; do for r in 0.3 0.5 0.7 0.9 0.95 1.0; do J+=("$U $r 1.0 6 -"); done; done
  for U in 6 4 2 1; do for r in 0.3 0.7 0.95; do for e in 0.9 0.7 0.5 0.3 0.0; do J+=("$U $r $e 6 -"); done; done; done
  for U in 4 1; do for e in 1.0 0.9 0.7 0.5 0.3 0.0; do J+=("$U 0.7 $e 0 -"); done; done
  for U in 4 1; do for fl in - genie; do J+=("$U 1.0 1.0 0 $fl"); done; done
fi

check() {   # $1 tag, $2 U, $3 flag -> "ok", "missing" or the mismatches of an existing run
  "$PY" - "$OUT/pca_csi_$1" "$FRAMES" "$2" "$3" $MODES <<'EOF'
import json, os, sys
base, frames, U, flag, modes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
if not os.path.exists(base + ".json"):
    print("missing"); sys.exit()
a, res = (lambda d: (d["args"], d["res"]))(json.load(open(base + ".json")))
bad = [] if os.path.exists(base + ".npz") else ["no npz"]
for key, want in (("frames", frames), ("k", U), ("snrs", [0, 10, 20]), ("nsamp", 32), ("genie", flag == "genie"),
                  ("legacy_delay", False)):
    if a.get(key) != want:
        bad.append(f"{key} {a.get(key)} != {want}")
miss = [m for m in modes if m not in res]
if miss:
    bad.append("no modes " + ",".join(miss))
print("ok" if not bad else "; ".join(bad))
EOF
}

launch() {  # $1 = "U rho_f eta nearfar flag"
  local U r e nf fl tag st
  read -r U r e nf fl <<< "$1"
  tag="coh_rho${r}_eta${e}_U${U}$([ "$nf" != 0 ] && echo "_nf${nf}")$([ "$fl" = genie ] && echo "_genie")"
  st=$(check "$tag" "$U" "$fl")
  if [ "$st" = ok ]; then log "skip $tag (complete)"; return; fi
  [ "$st" != missing ] && log "rerun $tag ($st); pca_csi.py moves the old files to old/"
  if [ "$DRY" = 1 ]; then log "would run $tag"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 20; done
  log "start $tag"
  ( t0=$(date +%s)
    nice -n 10 "$PY" pca_csi.py --coh --U "$U" --k "$U" --rho "$r" --eta "$e" --nearfar "$nf" --frames "$FRAMES" \
      --snrs 0 10 20 --modes $MODES $([ "$fl" = genie ] && echo --genie) > "logs/${tag}.log" 2>&1
    rc=$?; log "done $tag rc=$rc $(( $(date +%s) - t0 )) s" ) &
  sleep 2
}

gate_check() {
  "$PY" - "$OUT" <<'EOF'
import json, os, sys
out, ok = sys.argv[1], True
for U in (1, 4):
    fn, fo = f"{out}/pca_csi_coh_rho0.7_eta1.0_U{U}_nf6.json", f"{out}/pca_csi_U{U}_k6_rho0.7_nf6.json"
    if not (os.path.exists(fn) and os.path.exists(fo)):
        print(f"U={U}: missing {fn if not os.path.exists(fn) else fo}"); ok = False; continue
    new, old = json.load(open(fn)), json.load(open(fo))
    if new["args"]["frames"] != old["args"]["frames"]:
        print(f"U={U}: frames {new['args']['frames']} != stored {old['args']['frames']}"); ok = False; continue
    for m in ("fixed", "csiq", "csi"):
        if m not in new["res"] or m not in old["res"]:
            continue
        for q in ("cos", "weakest"):
            x, y = new["res"][m]["0"][q], old["res"][m]["0"][q]
            print(f"U={U} {m} {q} 0 dB: {x!r} stored {y!r} diff {x - y:.1e}")
            ok = ok and abs(x - y) <= 1e-9
print("GATE PASSED" if ok else "GATE FAILED")
EOF
}

log "start coherence sweep, ${#J[@]} points (gate $GATE first), $N_PAR at a time, $FRAMES frames, $THREADS threads"
if [ "$GATE" = 1 ]; then
  for j in "${GATE_PTS[@]}"; do launch "$j"; done
  wait
  g=$(gate_check 2>&1)
  if [ "$DRY" = 1 ]; then echo "$g"; else echo "$g" >> logs/coh_queue.log; fi
  if ! grep -q "GATE PASSED" <<< "$g"; then
    log "reproduction gate failed"
    [ "$DRY" = 1 ] || exit 1
  fi
fi
for j in "${J[@]}"; do launch "$j"; done
wait
log "all done"
