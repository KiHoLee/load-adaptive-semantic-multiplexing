#!/bin/bash
# 2026-11-03 pilot: channel-aware dynamic masks under slot-correlated fading (pilot_corr.py, seed 1,
# rho = 0.7). Primary channel: independent users with near-far offsets of up to 6 dB (tag _nf6);
# secondary: the same fading without near-far. One GPU (variable GPU, default 0), N_PAR runs at a time, primary runs first.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
N_PAR=${N_PAR:-5}
JOBS=(
  # primary: near-far 6 dB
  "--scheme prop --mode netqpw --nearfar 6 --tag _nf6"
  "--scheme prop --mode csisort --nearfar 6 --tag _nf6"
  "--scheme prop --mode fixed --nearfar 6 --tag _nf6"
  "--scheme ods6 --mode netpw --ods-all-loads --nearfar 6 --tag _all_nf6"
  "--scheme ods4 --mode netpw --ods-all-loads --nearfar 6 --tag _all_nf6"
  "--scheme prop --mode netq --nearfar 6 --tag _nf6"
  "--scheme prop --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods1 --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods2 --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods3 --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods4 --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods6 --mode netpw --nearfar 6 --tag _nf6"
  "--scheme ods1 --mode block --nearfar 6 --tag _nf6"
  "--scheme ods2 --mode block --nearfar 6 --tag _nf6"
  "--scheme ods3 --mode block --nearfar 6 --tag _nf6"
  "--scheme ods4 --mode block --nearfar 6 --tag _nf6"
  "--scheme ods6 --mode block --nearfar 6 --tag _nf6"
  "--scheme ods4 --mode csisort --nearfar 6 --tag _nf6"
  "--scheme ods6 --mode csisort --nearfar 6 --tag _nf6"
  # secondary: no near-far
  "--scheme prop --mode netqpw"
  "--scheme prop --mode csisort"
  "--scheme prop --mode fixed"
  "--scheme ods6 --mode netpw --ods-all-loads --tag _all"
  "--scheme ods4 --mode netpw"
  "--scheme ods4 --mode block"
  "--scheme ods6 --mode block"
)
echo "$(date +%H:%M) start corr pilot, ${#JOBS[@]} runs, $N_PAR at a time"
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 20; done
  name=$(echo "$j" | sed -E 's/--scheme ([^ ]+) --mode ([^ ]+).*/\1_\2/')$(echo "$j" | sed -nE 's/.*--tag ([^ ]+).*/\1/p')
  echo "$(date +%H:%M) start $name"
  $PY pilot_corr.py $j --seed 1 > logs/corr_${name}_s1.log 2>&1 &
  sleep 5
done
wait
echo "$(date +%H:%M) all done"
