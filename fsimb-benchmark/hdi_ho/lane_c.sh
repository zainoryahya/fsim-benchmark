#!/bin/bash
set -e; cd "$(dirname "$0")"; D=data/doro
step() { echo "== $(date +%H:%M) $*" ; }
step xclf;  python run_xclf_doro.py $D results_doro_xclf
step ridge; python launch_ridge_doro.py $D results_doro_ridge
echo "== $(date +%H:%M) LANE C DONE"
