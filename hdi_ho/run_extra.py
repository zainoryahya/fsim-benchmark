"""Further baselines requested in review, on the same outer folds as run_base.py.

* ALL   - no selection: every classifier on all features (stored under k = 0).
* RFE   - SVM-RFE (Guyon et al. 2002): linear SVM (C = 1), remove the 10% of remaining features with the smallest
          w_j^2 per step until 40 remain, then one feature per step, giving a full order of the last 40.
* RFEw  - the same with balanced class weights (cost-sensitive SVM-RFE).
* Hs    - Hellinger filter, ties in stable index order (as in run_base.py, for pairing).
* Hf    - Hellinger filter, ties broken by the ANOVA F statistic.
* Hr    - Hellinger filter, ties broken at random (seeded per fold), to measure how much tie order matters.
Top k = 10, 20, 30, 40 scored with 5-NN, logistic regression, linear SVM and random forest as in run_base.py.
Usage: python run_extra.py <data_dir> <out_dir> [n_jobs]   (env NREP, default 5)
"""
import sys, os, glob, time, pickle, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from run_xclf import score
from run_wrapper import CEILING

warnings.filterwarnings("ignore")
NREP = int(os.environ.get("NREP", "5"))
KS = (10, 20, 30, 40)
RANKERS = ["RFE", "RFEw", "Hs", "Hf", "Hr"]
CLFS = ["knn", "lr", "svm", "rf"]


def svm_rfe(X, y, weighted, keep=40, frac=0.1):
    """Return the indices of the last `keep` surviving features, best first."""
    alive = np.arange(X.shape[1])
    cw = "balanced" if weighted else None
    while len(alive) > keep:
        w = LinearSVC(C=1.0, class_weight=cw, max_iter=5000, dual="auto").fit(X[:, alive], y).coef_.ravel()
        drop = max(1, min(int(frac * len(alive)), len(alive) - keep))
        alive = alive[np.argsort(w ** 2, kind="stable")[drop:]]
    order = []
    while len(alive) > 0:
        w = LinearSVC(C=1.0, class_weight=cw, max_iter=5000, dual="auto").fit(X[:, alive], y).coef_.ravel()
        i = int(np.argmin(w ** 2))
        order.append(alive[i]); alive = np.delete(alive, i)
    return np.array(order[::-1])


def rankings(Xtr, ytr, seed):
    out, t = {}, {}
    s = time.perf_counter(); out["RFE"] = svm_rfe(Xtr, ytr, False); t["RFE"] = time.perf_counter() - s
    s = time.perf_counter(); out["RFEw"] = svm_rfe(Xtr, ytr, True); t["RFEw"] = time.perf_counter() - s
    h = ind.hellinger(Xtr, ytr)
    s = time.perf_counter(); out["Hs"] = np.argsort(-h, kind="stable"); t["Hs"] = time.perf_counter() - s
    f = ind.anova_f(Xtr, ytr)
    out["Hf"] = np.lexsort((-f, -h)); t["Hf"] = t["Hs"]
    u = np.random.default_rng(seed).random(len(h))
    out["Hr"] = np.lexsort((u, -h)); t["Hr"] = t["Hs"]
    ties = int(40 - len(np.unique(h[out["Hs"][:40]]))) if len(h) >= 40 else 0
    return out, t, dict(ties_top40=ties, n_tied_at_40=int((h == h[out["Hs"][39]]).sum()) if len(h) >= 40 else 0)


def run_one(path, out_dir, n_splits=5, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {(m, k, c): [np.zeros(len(y)) for _ in range(NREP)] for m in RANKERS for k in KS for c in CLFS}
    prob.update({("ALL", 0, c): [np.zeros(len(y)) for _ in range(NREP)] for c in CLFS})
    subsets = {(m, k): [] for m in RANKERS for k in KS}
    times = {m: [] for m in RANKERS}
    info = []
    for r in range(NREP):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
            prob[("ALL", 0, "knn")][r][te] = knn_prob(sqdist(Xte, Xtr), ytr)
            for c in ("lr", "svm", "rf"):
                prob[("ALL", 0, c)][r][te] = np.clip(score(c, Xtr, ytr, Xte, seed), 0.02, 0.98)
            rk, tm, inf = rankings(Xtr, ytr, seed + 100 * r + fi)
            info.append(inf)
            for m in RANKERS:
                times[m].append(tm[m])
                for k in KS:
                    cols = rk[m][:k]
                    subsets[(m, k)].append(cols.astype(np.int32))
                    prob[(m, k, "knn")][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                    for c in ("lr", "svm", "rf"):
                        prob[(m, k, c)][r][te] = np.clip(score(c, Xtr[:, cols], ytr, Xte[:, cols], seed), 0.02, 0.98)
    res = {}
    for key, ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[key] = {q: float(np.mean([x[q] for x in rows])) for q in ("gmean", "mcc", "aucpr", "auc")}
    pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), nrep=NREP, res=res,
                     subsets=subsets, times=times, info=info), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: os.path.getsize(p))
    Parallel(n_jobs=nj)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
