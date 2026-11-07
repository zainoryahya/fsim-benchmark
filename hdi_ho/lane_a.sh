#!/bin/bash
set -e; cd "$(dirname "$0")"; D=data/doro
step() { echo "== $(date +%H:%M) $*" ; }
step wrapper-F; POOL=F CONFIGS=acc_a99 python run_wrapper.py $D results_doro_wrap
step cap;       python run_cap.py $D results_doro_cap
step leak;      python launch_leak2_doro.py
echo "== $(date +%H:%M) LANE A DONE"
