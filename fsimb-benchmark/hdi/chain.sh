#!/bin/bash
# Held-out validation chain. Stops at the first failing step.
set -e
cd "$(dirname "$0")/../hdi_ho"
step() { echo "== $(date +%H:%M) $*" ; }
step wrapper-F;  POOL=F CONFIGS=acc_a99 python run_wrapper.py data/v2 results_wrap
step wrapper-H;  POOL=H CONFIGS=acc_a99,bbrier_a99 python run_wrapper.py data/v2 results_wrapH
step cap;        python run_cap.py data/v2 results_cap
step emt;        python run_emt.py data/v2 results_emt
step xclf;       python run_xclf.py data/v2 results_xclf
step ridge;      python launch_ridge.py data/v2 results_ridge
step leak;       python launch_leak2_ho.py
step thr;        python launch_thr_ho.py
echo "== $(date +%H:%M) ALL DONE"
