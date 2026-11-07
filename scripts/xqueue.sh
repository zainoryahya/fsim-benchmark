#!/bin/bash
# Restartable queue: extra repeats (r = 2..4) and new baselines for both suites. Each step leaves a marker.
step() {  # dir name command...
  local d=$1 n=$2; shift 2
  [ -f "$d/.done_$n" ] && return 0
  echo "== $(date +%H:%M) $d $n"
  (cd $d && eval "$@") && touch "$d/.done_$n" || { echo "FAILED $d $n"; exit 1; }
}
X="R0=2 NREP=3"
for d in "$(dirname "$0")/../hdi" "$(dirname "$0")/../hdi_ho"; do
  step $d wrapx  "$X POOL=F CONFIGS=acc_a99 python run_wrapper_x.py data/v2 results_wrap_x"
  step $d wraphx "$X POOL=H CONFIGS=acc_a99,bbrier_a99 python run_wrapper_x.py data/v2 results_wrapH_x"
  step $d capx   "$X python run_cap_x.py data/v2 results_cap_x"
  step $d emtx   "$X python run_emt_x.py data/v2 results_emt_x"
  step $d xclfx  "$X SUF=_x python run_xclf_x.py data/v2 results_xclf_x"
  step $d thrx   "$X python launch_thr_x.py data/v2 results_thr_x.pkl 2"
  step $d tomekx "$X python launch_tomek_x.py data/v2 results_tomek_x.pkl 2"
  step $d ridgex "$X SUF=_x python launch_ridge_x.py data/v2 results_ridge_x"
done
for d in "$(dirname "$0")/../hdi" "$(dirname "$0")/../hdi_ho"; do
  step $d base "NREP=5 python run_base.py data/v2 results_base 2"
done
echo "== $(date +%H:%M) QUEUE DONE"
