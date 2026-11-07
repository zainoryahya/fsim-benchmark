# fsimb: evaluation benchmark for feature selection on high-dimensional imbalanced data

Code for the manuscript "Evaluating feature selection for high-dimensional imbalanced data". `hdi/` holds the
discovery suite (34 analysed tasks) and `hdi_ho/` the held-out suite (18 tasks) and Dorothea. The library
`fsimb/` and the run scripts are byte-identical in the two folders; only data and result directories differ.

Contents: code, the result files of every run (`results_*`, `*.pkl`), the summary files written by the analysis
scripts (`summary_*.json`), the dataset inventories (`data/*/inventory.csv`), the held-out pre-registration of the
Dorothea predictions (`hdi_ho/dorothea_prereg.md`) and `paper/make_assets.py`, which rebuilds every table and
figure of the manuscript from the stored results (`python paper/make_assets.py; python paper/make_doro.py`).
The datasets themselves are public and are not redistributed here; `build_data.py`, `build_heldout.py` and
`build_dorothea.py` rebuild them from the original sources (see below). `scripts/` holds the queue scripts used
to run repeats 2-4 (`xqueue*.sh`) and sssHD (`ssshd_queue.sh`).

## Requirements

Python 3.13 with the versions in `requirements.txt` (`pip install -r requirements.txt`). Use scikit-learn 1.8 or
later: the L1 and elastic-net baselines set only `l1_ratio`, which older releases ignore (they fit an L2 model).

## Library (`fsimb/`)

| File | Content |
| --- | --- |
| `measures.py` | G-mean, MCC (stored as (MCC+1)/2), AUC, AUC-PR (average precision, step-wise), balanced Brier and log-loss and other measures; checked against scikit-learn |
| `indicators.py` | Filter scores: Hellinger (equal-frequency bins on ranks, b = max(3, floor(sqrt(n)))), ANOVA F, \|AUC-0.5\|, balanced Fisher, Relief-F |
| `innerest.py` | Inner estimates on the training fold: leave-one-out, 5x2 and 5x5 CV, out-of-bag bootstrap; 5-NN with inverse-distance weights |
| `optimizer.py` | Binary grey wolf optimiser (single task) and the multitask variant with UCB1 source selection |
| `ridge.py` | Closed-form leave-one-out ridge fitness (exact for fixed class weights) |
| `baselines.py` | Embedded and other filter baselines: L1 and elastic-net logistic, mRMR, random-forest impurity and permutation importance |
| `mfcso.py` | Re-implementation of MF-CSO |
| `ssshd.py` | Re-implementation of sssHD (Fu et al. 2020) from the paper and the authors' R code; deviations in the docstring |
| `stats.py` | Nogueira stability, Holm, Bayesian signed-rank test |

## Reproducing the manuscript

Raw data: `.RData` files from github.com/ramhiser/datamicroarray (data/) into `data_raw/`, and `.mat` files from
github.com/jundongl/scikit-feature (gh-pages branch, files/datasets/) into `data_raw/skf/`.

| Manuscript item | Run (in `hdi/`, then the same in `hdi_ho/`) | Analysis |
| --- | --- | --- |
| Data, Table inventory | `build_data.py data_raw data/v2`; held-out: `build_heldout.py`; `hdi_ho/build_dorothea.py` | `paper/make_assets.py` |
| Ceiling rule | `baseline_v2.py` (prototype 3-NN pipeline, filters top-20) | |
| Leakage (Table leak) | `launch_leak2.py`, `launch_leak2_ho.py` (`run_leak2.py`) | `analyze_leak2.py` |
| Threshold vs SMOTE, Tomek (Table thr) | `launch_thr*.py`, `launch_tomek*.py` (`run_thr.py`, `run_tomek.py`) | `analyze_thr.py`, `analyze_tomek.py`, `analyze_dep.py` |
| Further filter and embedded baselines (Table base) | `run_base.py data/v2 results_base` (NREP=5) | `analyze_base.py` |
| Fitness fidelity (Table fid) | `run_fidelity.py data/v2 results_fid` | `analyze_fidelity.py` |
| Wrapper, caps, other classifiers, ridge fitness | `POOL=H run_wrapper.py`, `run_cap.py`, `run_xclf.py`, `launch_ridge.py` | `analyze_cap.py`, `analyze_xclf.py`, `analyze_ridge.py`, `analyze_dep.py` |
| EMT (Table emt) | `run_emt.py` | `analyze_emt.py` |
| MF-CSO (Table mfcso) | `R0=0 NREP=2 run_mfcso.py data/v2 results_mfcso` | `analyze_mfcso.py` |
| sssHD | `run_ssshd.py data/v2 results_ssshd` (NREP=2, K_SUB=50) | `analyze_ssshd.py` |
| Held-out hypotheses (Table prereg) and sensitivity analysis | `hdi_ho/chain.sh` | `hdi_ho/analyze_ho.py`, `hdi_ho/sens_ho.py` |
| Dorothea (Table doro) | `xqueue2.sh`, `hdi_ho/*_doro.py` | `hdi_ho/merge_doro.py`, `hdi_ho/analyze_doro.py`, `paper/make_doro.py` |
| Run time (Table time) | `time_methods.py`, `time_ssshd.py` (one core, idle machine) | `paper/make_assets.py` |

Repeats 0-1 were run first and repeats 2-4 later with the `*_x.py` scripts (`xqueue.sh`, R0=2 NREP=3);
`merge_x.py <suite dir>` merges them into the 5-repeat results (weighted means 2/5 and 3/5, subsets concatenated
in repeat order). Seeds: outer folds `StratifiedKFold(5, shuffle=True, random_state=2027 + r)`; generators of
the selection methods are seeded from (2027, repeat, fold); permutation j of the leakage runs uses seed 100 + j.
The final random forest uses `random_state=2027` in every fold, the same for all methods compared.

## Prototype files (not used for any reported result)

`run_main.py`, `run_leakage.py`, `analyze.py`, `load_data.py`, `make_synth.py`, `make_variants.py`,
`fsimb/pipeline.py`, `fsimb/fitness.py`, `run_ctrl.py`/`analyze_ctrl.py`, `run_wrapper.py` with `POOL=F` and
`analyze_wrapper.py` are from the first prototype and exploratory runs. They use a 3-NN fitness, an ANOVA-F
candidate pool and other settings that differ from the manuscript. The prototype was run on early copies of the
chowdary and singh data, which later entered the held-out suite (see the manuscript).
