"""Phase 2, part A: fitness fidelity.

For each dataset, 5-fold x 2 outer splits. In every outer training fold:
  1. standardise with training statistics; build a candidate pool of subsets
     (see candidates()), all selected using the training fold only;
  2. score every candidate with each of the 12 measures under each of the 4
     inner estimators, on the training fold only;
  3. fit the classifier on the training fold with that subset and score the
     outer test fold (the 'truth'): G-mean, MCC (normalised), AUC-PR.
Analysis (analyze_fidelity.py) then asks how well each inner score ranks the
candidates by their truth, and what the inner-best candidate really delivers.

Pre-declared protocol: primary criterion = within-size-stratum Spearman
correlation, averaged over the three truths; secondary = global Spearman and
the percentile of the inner-best pick. Ceiling datasets (filter G-mean >= 0.99)
are run but excluded from the main analysis.

Usage: python -I run_fidelity.py <data_dir> <out_dir> [name ...]
"""
import sys, os, time, glob, numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

from fsimb import indicators as ind
from fsimb.innerest import estimate, sqdist, knn_prob, ESTIMATORS
from fsimb.measures import MEASURES, all_measures

POOL = 200
SIZES = [3, 5, 10, 20, 40]
TRUTHS = ["gmean", "mcc", "aucpr"]


def candidates(Xtr, ytr, rng):
    """~80 candidate subsets of mixed quality at five sizes.

    base:      top-k of three label-aware filters (F, Hellinger, |AUC-0.5|)
    perturbed: base subsets with 25/50/75% of features swapped for random
               features from the filter pools
    random:    random subsets of the pooled features, and a few from all
               features (almost surely uninformative)
    """
    p = Xtr.shape[1]
    scores = [ind.anova_f(Xtr, ytr), ind.hellinger(Xtr, ytr), ind.auc_score(Xtr, ytr)]
    order = [np.argsort(-s) for s in scores]
    U = np.unique(np.concatenate([o[: min(POOL, p)] for o in order]))
    cands = []

    def add(idx, kind):
        cands.append((np.asarray(idx, int), kind))

    for k in SIZES:
        base = [o[:k] for o in order]
        for b in base:
            add(b, "base")
            for frac in (0.25, 0.5, 0.75):
                m = max(1, int(round(frac * k)))
                keep = rng.choice(b, size=k - m, replace=False)
                free = np.setdiff1d(U, keep)
                add(np.concatenate([keep, rng.choice(free, size=m, replace=False)]), "perturbed")
        for _ in range(3):
            add(rng.choice(U, size=k, replace=False), "random")
        if k in (5, 10, 20):
            for _ in range(2):
                add(rng.choice(p, size=k, replace=False), "random_all")
    return cands


def run_one(path, out_dir, n_splits=5, n_repeats=2, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    inner, truth, sizes, kinds, splits = [], [], None, None, []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for fi, (tr, te) in enumerate(skf.split(X, y)):
            rng = np.random.default_rng([seed, r, fi])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ytr, yte = y[tr], y[te]
            cands = candidates(Xtr, ytr, rng)
            if sizes is None:
                sizes = np.array([len(c[0]) for c in cands])
                kinds = np.array([c[1] for c in cands])
            I = np.zeros((len(cands), len(MEASURES), len(ESTIMATORS)), np.float32)
            T = np.zeros((len(cands), len(TRUTHS)), np.float32)
            for ci, (cols, _) in enumerate(cands):
                Z = Xtr[:, cols]
                est = estimate(sqdist(Z), ytr, rng)
                for ei, e in enumerate(ESTIMATORS):
                    I[ci, :, ei] = est[e]
                prob = knn_prob(sqdist(Xte[:, cols], Z), ytr)
                m = all_measures(yte, prob)
                T[ci] = [m[t] for t in TRUTHS]
            inner.append(I)
            truth.append(T)
            splits.append((r, fi, int(yte.sum())))
    np.savez_compressed(f"{out_dir}/{name}.npz", inner=np.array(inner), truth=np.array(truth),
                        sizes=sizes, kinds=kinds, splits=np.array(splits),
                        measures=np.array(MEASURES), estimators=np.array(ESTIMATORS),
                        truths=np.array(TRUTHS), n=len(y), p=X.shape[1], minority=int(y.sum()))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)
    return name


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    names = sys.argv[3:]
    paths = sorted(glob.glob(f"{data_dir}/*.npz")) if not names else [f"{data_dir}/{n}.npz" for n in names]
    paths = [p for p in paths if not os.path.exists(f"{out_dir}/{os.path.basename(p)}")]
    # largest first keeps both workers busy to the end
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
