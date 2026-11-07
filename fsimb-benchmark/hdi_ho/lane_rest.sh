#!/bin/bash
# After lane B finishes: leakage (one worker, memory), then classifier re-scoring and ridge.
set -e; cd "$(dirname "$0")"
until grep -q -E "LANE B DONE|Traceback" lane_b.log; do sleep 30; done
step() { echo "== $(date +%H:%M) $*" ; }
step leak; python launch_leak2_doro.py

echo "== $(date +%H:%M) LEAK DONE"
