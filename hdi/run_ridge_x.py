"""Phase 5b: a wrapper whose fitness uses a classifier matched to the final one.

Fitness: class-balanced kernel ridge (linear) with exact closed-form LOO, balanced-Brier quality,
alpha = 0.99, lam = LAM_MULT * |S| (set before running, not tuned). Same Hellinger top-200 pool,
~2000 evaluations, same folds as before. Final classifiers: the same ridge (rls), logistic regression
and linear SVM on the outer test folds.

  ridge_free      no size cap (size penalty only)
  ridge_cap10/20/30   subset repaired to its 10/20/30 best-ranked (Hellinger) members
Saved comparators (from earlier phases, re-scored): H10/20/30/40, F20, cap20 (5-NN-matched).
Usage: python run_ridge.py <data_dir> <out_dir>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, glob, time, pickle, warnings, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.special import expit
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.measures import all_measures
from fsimb.optimizer import bgwo
from fsimb.ridge import loo_prob, fit_predict
from run_wrapper_x import CEILING, POOL
from run_cap import repair
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

warnings.filterwarnings("ignore")
LAM_MULT, ALPHA = 1.0, 0.99
CAPS = [10, 20, 30]
NEW = ["ridge_free"] + [f"ridge_cap{c}" for c in CAPS]
SAVED = {"H10": "H10", "H20": "H20", "H30": "H30", "H40": "H40", "cap20_knn": "cap20"}   # name -> key in results_cap
CLFS = ["rls", "lr", "svm"]


class RidgeFit:
    def __init__(self, X, y, cap=None):
        self.X, self.y, self.cap, self.K = X, y, cap, X.shape[1]
        self.bm = None

    def __call__(self, mask):
        if self.cap is not None:
            mask = repair(mask, self.cap)
        k = int(mask.sum())
        if k == 0:
            return 1.0
        p = loo_prob(self.X[:, mask], self.y, LAM_MULT * k)
        pos, neg = p[self.y == 1], p[self.y == 0]
        q = 1.0 - 0.5 * (np.mean((1 - pos) ** 2) + np.mean(neg ** 2))
        return ALPHA * (1.0 - q) + (1.0 - ALPHA) * k / self.K


def score(clf, Xtr, ytr, Xte):
    if clf == "rls":
        return fit_predict(Xtr, ytr, Xte, LAM_MULT * Xtr.shape[1])
    if clf == "lr":
        return np.clip(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1], .02, .98)
    return np.clip(expit(LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, dual="auto").fit(Xtr, ytr).decision_function(Xte)), .02, .98)


def run_one(path, out_dir, n_splits=5, n_repeats=NREP, seed=2027):
    name = os.path.basename(path)[:-4]
    t0 = time.time()
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    old = pickle.load(open(f"results_cap{SUF}/{name}.pkl", "rb"))["subsets"]
    f20 = pickle.load(open(f"results_wrap{SUF}/{name}.pkl", "rb"))["subsets"]["F-top20"]
    methods = NEW + list(SAVED) + ["F20"]
    prob = {(m, c): [np.zeros(len(y)) for _ in range(n_repeats)] for m in methods for c in CLFS}
    subsets = {m: [] for m in methods}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r + R0).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = y[tr]
            pool = np.argsort(-ind.hellinger(Xtr, ytr))[: min(POOL, X.shape[1])]
            Xp = Xtr[:, pool]
            cols = {}
            for m in NEW:
                cap = None if m == "ridge_free" else int(m.replace("ridge_cap", ""))
                rng = np.random.default_rng([seed, r + R0, fi, 999])
                mask, _, _ = bgwo(RidgeFit(Xp, ytr, cap), Xp.shape[1], rng)
                cols[m] = pool[repair(mask, cap) if cap else mask]
            for m, key in SAVED.items():
                cols[m] = old[key][r * n_splits + fi]
            cols["F20"] = f20[r * n_splits + fi]
            for m in methods:
                subsets[m].append(np.asarray(cols[m], dtype=np.int32))
                for c in CLFS:
                    prob[(m, c)][r][te] = score(c, Xtr[:, cols[m]], ytr, Xte[:, cols[m]])
    res = {}
    for (m, c), ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[(m, c)] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc")}
    for m in methods:
        res[(m, "size")] = float(np.mean([len(s) for s in subsets[m]]))
    pickle.dump(dict(name=name, p=X.shape[1], minority=int(y.sum()), res=res, subsets=subsets), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
