#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../hdi" && python3 run_ssshd.py data/v2 results_ssshd 2 >> "$(dirname "$0")/../"ssshd_disc.log 2>&1
cd "$(dirname "$0")/../hdi_ho" && python3 run_ssshd.py data/v2 results_ssshd 2 >> "$(dirname "$0")/../"ssshd_ho.log 2>&1
echo ALLDONE >> "$(dirname "$0")/../"ssshd_ho.log
