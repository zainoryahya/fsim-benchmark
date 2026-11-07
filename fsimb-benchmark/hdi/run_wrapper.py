"""Phase 2, part B: does fitness fidelity carry over to a real wrapper search?

Binary GWO (50 wolves x 40 iterations = 2000 evaluations) over the top-200
features of the outer training fold, ranked by ANOVA F (env POOL=F, the default) or by the Hellinger
distance (POOL=H, used for all results reported in the manuscript; see chain.sh). The fitness is
    f(S) = alpha * (1 - Q(S)) + (1 - alpha) * |S| / K,
with Q the chosen measure estimated by leave-one-out 5-NN on the training fold.
The best subset found is then scored once on the outer test fold.

All configurations share the same initial population within an outer split
(common random numbers), so differences between them are paired.

Outcome per (dataset, repeat): G-mean, MCC (normalised), AUC-PR, AUC on the
pooled out-of-fold test predictions, plus mean subset size and the selected
subsets (for stability).

Usage: python -I run_wrapper.py <data_dir> <out_dir> [name ...]
"""
import sys, os, time, glob, pickle, numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from fsimb.optimizer import bgwo

POOL = 200
CEILING = {"chiaretti", "CLL_SUB_111", "lymphoma", "Carcinom"}
CONFIGS = {  # name: (measure, alpha)
    "acc_a99": ("acc", 0.99), "gmean_a99": ("gmean", 0.99), "mcc_a99": ("mcc", 0.99),
    "bbrier_a99": ("bbrier", 0.99), "bll_a99": ("bll", 0.99), "auc_a99": ("auc", 0.99),
    "aucpr_a99": ("aucpr", 0.99),
    "acc_a95": ("acc", 0.95), "gmean_a95": ("gmean", 0.95), "bbrier_a95": ("bbrier", 0.95),
    "acc_a90": ("acc", 0.90), "gmean_a90": ("gmean", 0.90), "bbrier_a90": ("bbrier", 0.90),
}
FILTERS = ["F-top20", "H-top20"]


class MeasureFitness:
    def __init__(self, X, y, measure, alpha):
        self.X, self.y, self.m, self.a, self.K = X, y, measure, alpha, X.shape[1]

    def __call__(self, mask):
        k = int(mask.sum())
        if k == 0:
            return 1.0
        D = sqdist(self.X[:, mask])
        np.fill_diagonal(D, np.inf)
        q = all_measures(self.y, knn_prob(D, self.y))[self.m]
        return self.a * (1.0 - q) + (1.0 - self.a) * k / self.K


def run_one(path, out_dir, n_splits=5, n_repeats=2, seed=2027, pool_score="F", configs=None, with_filters=True):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    cfg = {k: CONFIGS[k] for k in (configs or CONFIGS)}
    methods = list(cfg) + (FILTERS if with_filters else [])
    prob = {m: [np.zeros(len(y)) for _ in range(n_repeats)] for m in methods}
    subsets = {m: [] for m in methods}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr = y[tr]
            score = ind.hellinger(Xtr, ytr) if pool_score == "H" else ind.anova_f(Xtr, ytr)
            pool = np.argsort(-score)[: min(POOL, X.shape[1])]
            Xp = Xtr[:, pool]
            for m in methods:
                if m in cfg:
                    meas, alpha = cfg[m]
                    rng = np.random.default_rng([seed, r, fi, 777])
                    mask, _, _ = bgwo(MeasureFitness(Xp, ytr, meas, alpha), Xp.shape[1], rng)
                    cols = pool[mask]
                elif m == "F-top20":
                    cols = np.argsort(-ind.anova_f(Xtr, ytr))[:20]
                else:
                    cols = np.argsort(-ind.hellinger(Xtr, ytr))[:20]
                prob[m][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                subsets[m].append(cols.astype(np.int32))
    res = {}
    for m in methods:
        rows = [all_measures(y, prob[m][r]) for r in range(n_repeats)]
        res[m] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc", "acc", "f1")}
        res[m]["size"] = float(np.mean([len(s) for s in subsets[m]]))
    with open(f"{out_dir}/{name}.pkl", "wb") as f:
        pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), res=res, subsets=subsets), f)
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)
    return name


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    names = sys.argv[3:]
    paths = sorted(glob.glob(f"{data_dir}/*.npz")) if not names else [f"{data_dir}/{n}.npz" for n in names]
    paths = [p for p in paths if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    pool_score = os.environ.get("POOL", "F")
    configs = os.environ["CONFIGS"].split(",") if os.environ.get("CONFIGS") else None
    with_filters = pool_score == "F"
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir, pool_score=pool_score, configs=configs,
                                        with_filters=with_filters) for p in paths)
    print("done")
