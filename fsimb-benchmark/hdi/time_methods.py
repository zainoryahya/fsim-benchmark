"""Computational cost: single-core wall time of each selection method for one outer training fold.

First outer fold (seed 2027) of six datasets spanning n and p. Selection time only (ranking or search on the
training fold, including the Hellinger pre-ranking for the wrappers); evaluation of the final classifier is excluded.
Run on an otherwise idle machine. Usage: python time_methods.py <out.json>
"""
import json, sys, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from fsimb import indicators as ind
from fsimb.optimizer import bgwo, mt_bgwo
from run_wrapper import MeasureFitness, POOL
from run_cap import Capped, repair
from run_ridge import RidgeFit
from run_emt import tasks_for
from run_base import rankings

SETS = ["data/v2/alon.npz", "data/v2/gravier.npz", "data/v2/PCMAC_ir5.npz", "data/v2/yeoh.npz",
        "data/v2/burczynski.npz", "../hdi_ho/data/v2/sun.npz"]
out = {}
for path in SETS:
    if not os.path.exists(path):
        path = path.replace("data/v2/", "../hdi_ho/data/v2/")
    d = np.load(path); X = d["X"].astype(float); y = d["y"].astype(int)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=2027).split(X, y))
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
    Xtr, ytr = (X[tr] - mu) / sd, y[tr]
    T = {}
    _, tr_times, _ = rankings(Xtr, ytr, 0)
    T.update({f"filter/embedded:{k}": v for k, v in tr_times.items()})

    def pool():
        order = np.argsort(-ind.hellinger(Xtr, ytr)); return order[: min(POOL, X.shape[1])]
    s = time.perf_counter(); p = pool(); Xp = Xtr[:, p]
    bgwo(Capped(MeasureFitness(Xp, ytr, "bbrier", 0.99), 20), Xp.shape[1], np.random.default_rng(1))
    T["wrapper:BGWO 5-NN cap 20"] = time.perf_counter() - s
    s = time.perf_counter(); p = pool(); Xp = Xtr[:, p]
    bgwo(MeasureFitness(Xp, ytr, "bbrier", 0.99), Xp.shape[1], np.random.default_rng(1))
    T["wrapper:BGWO 5-NN no cap"] = time.perf_counter() - s
    s = time.perf_counter(); p = pool(); Xp = Xtr[:, p]
    bgwo(RidgeFit(Xp, ytr, 20), Xp.shape[1], np.random.default_rng(1))
    T["wrapper:BGWO ridge cap 20"] = time.perf_counter() - s
    s = time.perf_counter(); p = pool(); Xp = Xtr[:, p]
    rng = np.random.default_rng(1); tasks, sel = tasks_for("emt_min3", Xp, ytr, rng)
    mt_bgwo(MeasureFitness(Xp, ytr, "bbrier", 0.99), Xp.shape[1], tasks, rng, selector=sel)
    T["wrapper:EMT 3 tasks"] = time.perf_counter() - s
    out[os.path.basename(path)[:-4]] = dict(n_train=int(len(tr)), p=int(X.shape[1]), times=T)
    print(os.path.basename(path), {k: round(v, 2) for k, v in T.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
