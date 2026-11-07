"""Adds the single-core selection time of sssHD (first outer fold, seed 2027) to time_methods.json.

Same six datasets and fold as time_methods.py; K = 50 subsamples, as in run_ssshd.py. Run on an otherwise idle machine, from hdi/.
"""
import json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from fsimb.ssshd import ssshd_rank

TM = json.load(open("time_methods.json"))
for name in TM:
    path = f"data/v2/{name}.npz"
    if not os.path.exists(path):
        path = f"../hdi_ho/data/v2/{name}.npz"
    d = np.load(path); X = d["X"].astype(float); y = d["y"].astype(int)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=2027).split(X, y))
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
    Xtr, ytr = (X[tr] - mu) / sd, y[tr]
    s = time.perf_counter(); ssshd_rank(Xtr, ytr, np.random.default_rng(0), K=50)
    TM[name]["times"]["filter/embedded:sssHD"] = time.perf_counter() - s
    print(name, round(TM[name]["times"]["filter/embedded:sssHD"], 2), flush=True)
json.dump(TM, open("time_methods.json", "w"), indent=1)
