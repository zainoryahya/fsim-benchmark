"""Phase 3 control: is the EMT gain just a smaller search space / subset?

Single-task BGWO, same fitness (balanced Brier, alpha 0.99, LOO 5-NN), same Hellinger pool,
same ~2000-evaluation budget, but:
  single_top50  searches only the 50 best Hellinger features (the EMT auxiliary reach)
  single_cap20  searches all 200 but any subset is repaired to its 20 best-ranked members
Usage: python run_ctrl.py <data_dir> <out_dir>
"""
import sys, os, time, glob, pickle, numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from fsimb.optimizer import bgwo
from run_wrapper import MeasureFitness, CEILING, POOL

METHODS = ["single_top50", "single_cap20"]
CAP = 20


class Capped:
    def __init__(self, fit): self.fit = fit
    def __call__(self, mask): return self.fit(repair(mask))


def repair(mask):
    m = np.zeros_like(mask)
    m[np.flatnonzero(mask)[:CAP]] = True      # columns are ordered best-first
    return m


def run_one(path, out_dir, n_splits=5, n_repeats=2, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {m: [np.zeros(len(y)) for _ in range(n_repeats)] for m in METHODS}
    subsets = {m: [] for m in METHODS}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = y[tr]
            pool = np.argsort(-ind.hellinger(Xtr, ytr))[: min(POOL, X.shape[1])]
            for m in METHODS:
                rng = np.random.default_rng([seed, r, fi, 999])
                if m == "single_top50":
                    pl = pool[:50]
                    mask, _, _ = bgwo(MeasureFitness(Xtr[:, pl], ytr, "bbrier", 0.99), len(pl), rng)
                    cols = pl[mask]
                else:
                    Xp = Xtr[:, pool]
                    mask, _, _ = bgwo(Capped(MeasureFitness(Xp, ytr, "bbrier", 0.99)), Xp.shape[1], rng)
                    cols = pool[repair(mask)]
                prob[m][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                subsets[m].append(cols.astype(np.int32))
    res = {}
    for m in METHODS:
        rows = [all_measures(y, prob[m][r]) for r in range(n_repeats)]
        res[m] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc", "acc", "f1")}
        res[m]["size"] = float(np.mean([len(s) for s in subsets[m]]))
    pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), res=res, subsets=subsets),
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
