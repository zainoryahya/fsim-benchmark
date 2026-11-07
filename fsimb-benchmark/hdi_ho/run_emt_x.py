"""Phase 3: do minority-aware auxiliary tasks help a wrapper search?

Fixed from Phase 2: fitness = balanced Brier (alpha = 0.99) by LOO 5-NN, and the
candidate pool = top-200 Hellinger features of the outer training fold.
Only the search changes. Budget ~2000 evaluations for every method.

Methods (pre-declared):
  single      single-task BGWO (taken from results_wrapH, same splits and pool)
  emt_f       1 auxiliary task: top-50 pool features by ANOVA F
  emt_relief  1 auxiliary task: top-50 by Relief-F  (literature baseline)
  emt_min3    3 tasks: Hellinger, |AUC-0.5|, balanced Fisher (uniform source choice)
  emt_min3_rl same 3 tasks, UCB1 bandit chooses the source task
  emt_rand3   3 random 50-feature tasks (control: is it the tasks or just the structure?)

Usage: python run_emt.py <data_dir> <out_dir>
"""
import sys, os, time, glob, pickle, numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from fsimb.optimizer import mt_bgwo
from run_wrapper_x import MeasureFitness, CEILING, POOL
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

GROUP = 50
METHODS = ["emt_f", "emt_relief", "emt_min3", "emt_min3_rl", "emt_rand3"]


def top_group(score, K):
    m = np.zeros(K, bool)
    m[np.argsort(-score)[:GROUP]] = True
    return m


def tasks_for(method, Xp, ytr, rng):
    K = Xp.shape[1]
    if method == "emt_f":
        return [top_group(ind.anova_f(Xp, ytr), K)], "uniform"
    if method == "emt_relief":
        return [top_group(ind.relieff(Xp, ytr), K)], "uniform"
    if method in ("emt_min3", "emt_min3_rl"):
        return [top_group(fn(Xp, ytr), K) for fn in ind.MINORITY_AWARE.values()], \
               ("ucb" if method.endswith("rl") else "uniform")
    if method == "emt_rand3":
        out = []
        for _ in range(3):
            m = np.zeros(K, bool)
            m[rng.choice(K, min(GROUP, K), replace=False)] = True
            out.append(m)
        return out, "uniform"
    raise ValueError(method)


def run_one(path, out_dir, n_splits=5, n_repeats=NREP, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {m: [np.zeros(len(y)) for _ in range(n_repeats)] for m in METHODS}
    subsets = {m: [] for m in METHODS}
    infos = {m: [] for m in METHODS}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r + R0).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = y[tr]
            pool = np.argsort(-ind.hellinger(Xtr, ytr))[: min(POOL, X.shape[1])]
            Xp = Xtr[:, pool]
            for m in METHODS:
                rng = np.random.default_rng([seed, r + R0, fi, 999])
                tasks, sel = tasks_for(m, Xp, ytr, rng)
                fit = MeasureFitness(Xp, ytr, "bbrier", 0.99)
                mask, _, info = mt_bgwo(fit, Xp.shape[1], tasks, rng, selector=sel)
                cols = pool[mask]
                prob[m][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                subsets[m].append(cols.astype(np.int32))
                infos[m].append(info)
    res = {}
    for m in METHODS:
        rows = [all_measures(y, prob[m][r]) for r in range(n_repeats)]
        res[m] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc", "acc", "f1")}
        res[m]["size"] = float(np.mean([len(s) for s in subsets[m]]))
    with open(f"{out_dir}/{name}.pkl", "wb") as f:
        pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()),
                         res=res, subsets=subsets, infos=infos), f)
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)
    return name


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(f"{data_dir}/*.npz"))
    paths = [p for p in paths if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
