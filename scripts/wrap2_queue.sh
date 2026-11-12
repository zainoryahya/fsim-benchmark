#!/bin/bash
# Further wrapper checks (GA, 4x budget, ANOVA-F pool, search-seed stability) on both suites.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd hdi && python3 run_wrap2.py data/v2 results_wrap2 2 && cd ..
cd hdi_ho && python3 run_wrap2.py data/v2 results_wrap2 2 && cd ..
