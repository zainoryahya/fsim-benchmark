#!/bin/bash
# Dorothea: extra repeats (r = 2..4) and baselines (5 repeats). Restartable via markers.
step() { local d=$1 n=$2; shift 2; [ -f "$d/.done_$n" ] && return 0; echo "== $(date +%H:%M) $d $n"
  (cd $d && eval "$@") && touch "$d/.done_$n" || { echo "FAILED $d $n"; exit 1; }; }
X="R0=2 NREP=3"; d="$(dirname "$0")/../hdi_ho"
step $d d_wrapx  "$X POOL=F CONFIGS=acc_a99 python run_wrapper_x.py data/doro results_wrap_dx"
step $d d_wraphx "$X POOL=H CONFIGS=acc_a99,bbrier_a99 python run_wrapper_x.py data/doro results_wrapH_dx"
step $d d_capx   "$X python run_cap_x.py data/doro results_cap_dx"
step $d d_emtx   "$X python run_emt_x.py data/doro results_emt_dx"
step $d d_xclfx  "$X SUF=_dx python run_xclf_x.py data/doro results_xclf_dx"
step $d d_thrx   "$X python launch_thr_x.py data/doro results_doro_thr_x.pkl 1"
step $d d_tomekx "$X python launch_tomek_x.py data/doro results_doro_tomek_x.pkl 1"
step $d d_base   "NREP=5 python run_base.py data/doro results_doro_base 1"
echo "== $(date +%H:%M) QUEUE2 DONE"
