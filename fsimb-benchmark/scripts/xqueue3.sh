#!/bin/bash
# MF-CSO re-implementation, repeats 0-1, both suites. Restartable (per-dataset pickles).
for d in "$(dirname "$0")/../hdi" "$(dirname "$0")/../hdi_ho"; do
  [ -f $d/.done_mfcso ] && continue
  echo "== $(date +%H:%M) $d mfcso"
  (cd $d && R0=0 NREP=2 python run_mfcso.py data/v2 results_mfcso 2) && touch $d/.done_mfcso || exit 1
done
echo "== $(date +%H:%M) QUEUE3 DONE"
