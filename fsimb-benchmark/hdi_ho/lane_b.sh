#!/bin/bash
set -e; cd "$(dirname "$0")"; D=data/doro
step() { echo "== $(date +%H:%M) $*" ; }
step wrapper-H; POOL=H CONFIGS=acc_a99,bbrier_a99 python run_wrapper.py $D results_doro_wrapH
step emt;       python run_emt.py $D results_doro_emt
step thr;       python launch_thr_doro.py
echo "== $(date +%H:%M) LANE B DONE"
