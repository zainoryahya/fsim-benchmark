"""Option 1: size-capped single-task wrapper, sweep of the cap and an automatic choice of it.

Same fitness (balanced Brier, alpha 0.99, LOO 5-NN), Hellinger top-200 pool, ~2000 evaluations
per search. The subset is repaired to its `cap` best-ranked (Hellinger) members.
  cap10/20/30/40   fixed caps
  auto_loo         cap with the best inner LOO balanced-Brier of the subset it found
  auto_cv          cap with the best inner CV 5x5 balanced-Brier (different estimator from the search)
  H10/20/30/40     Hellinger top-k filters, same k
Usage: python run_cap.py <data_dir> <out_dir>
"""
import sys, os, time, glob, pickle, numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob, estimate
from fsimb.measures import all_measures, MEASURES
from fsimb.optimizer import bgwo
from run_wrapper_x import MeasureFitness, CEILING, POOL
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

CAPS = [10, 20, 30, 40]
IB = MEASURES.index("bbrier")
METHODS = [f"cap{c}" for c in CAPS] + ["auto_loo", "auto_cv"] + [f"H{c}" for c in CAPS]


def repair(mask, cap):
    m = np.zeros_like(mask)
    m[np.flatnonzero(mask)[:cap]] = True      # columns are ordered best-first
    return m


class Capped:
    def __init__(self, fit, cap): self.fit, self.cap = fit, cap
    def __call__(self, mask): return self.fit(repair(mask, self.cap))


def run_one(path, out_dir, n_splits=5, n_repeats=NREP, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {m: [np.zeros(len(y)) for _ in range(n_repeats)] for m in METHODS}
    subsets = {m: [] for m in METHODS}
    chosen = {"auto_loo": [], "auto_cv": []}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r + R0).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = y[tr]
            order = np.argsort(-ind.hellinger(Xtr, ytr))
            pool = order[: min(POOL, X.shape[1])]
            Xp = Xtr[:, pool]
            cols_of, sc_loo, sc_cv = {}, {}, {}
            for c in CAPS:
                rng = np.random.default_rng([seed, r + R0, fi, 999])
                mask, _, _ = bgwo(Capped(MeasureFitness(Xp, ytr, "bbrier", 0.99), c), Xp.shape[1], rng)
                cols = pool[repair(mask, c)]
                cols_of[f"cap{c}"] = cols
                D = sqdist(Xtr[:, cols])
                est = estimate(D, ytr, np.random.default_rng([seed, r + R0, fi, 5]), which=["loo", "cv5x5"])
                sc_loo[c], sc_cv[c] = est["loo"][IB], est["cv5x5"][IB]
                cols_of[f"H{c}"] = order[:c]
            cols_of["auto_loo"] = cols_of[f"cap{max(CAPS, key=lambda c: sc_loo[c])}"]
            cols_of["auto_cv"] = cols_of[f"cap{max(CAPS, key=lambda c: sc_cv[c])}"]
            chosen["auto_loo"].append(max(CAPS, key=lambda c: sc_loo[c]))
            chosen["auto_cv"].append(max(CAPS, key=lambda c: sc_cv[c]))
            for m in METHODS:
                cols = cols_of[m]
                prob[m][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                subsets[m].append(np.asarray(cols, dtype=np.int32))
    res = {}
    for m in METHODS:
        rows = [all_measures(y, prob[m][r]) for r in range(n_repeats)]
        res[m] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc", "acc", "f1")}
        res[m]["size"] = float(np.mean([len(s) for s in subsets[m]]))
    pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), res=res, subsets=subsets, chosen=chosen),
                open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
