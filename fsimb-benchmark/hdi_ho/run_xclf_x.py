"""Phase 5a: do the subsets chosen with a 5-NN fitness transfer to other classifiers?

Re-scores the saved subsets (same outer folds, same seeds) with logistic regression, linear SVM
and random forest on the outer test folds. The search is NOT repeated; only the final classifier
changes. Class weights are balanced for all three.
Usage: python run_xclf.py <data_dir> <out_dir>
"""
import sys, os, glob, time, pickle, warnings, numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from fsimb.measures import all_measures
from run_wrapper_x import CEILING
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

warnings.filterwarnings("ignore")
CLFS = ["lr", "svm", "rf"]
SRC = {  # method -> (results dir, key inside the pickle)
    "single": ("results_wrapH" + SUF, "bbrier_a99"), "F20": ("results_wrap" + SUF, "F-top20"),
    **{m: ("results_emt" + SUF, m) for m in ("emt_f", "emt_relief", "emt_min3", "emt_min3_rl", "emt_rand3")},
    **{m: ("results_cap" + SUF, m) for m in ("cap10", "cap20", "cap30", "cap40", "auto_loo", "auto_cv", "H10", "H20", "H30", "H40")},
}


def score(clf, Xtr, ytr, Xte, seed):
    if clf == "lr":
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if clf == "svm":
        m = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, dual="auto").fit(Xtr, ytr)
        return expit(m.decision_function(Xte))
    m = RandomForestClassifier(100, class_weight="balanced_subsample", min_samples_leaf=1, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_one(path, out_dir, n_splits=5, n_repeats=NREP, seed=2027):
    name = os.path.basename(path)[:-4]
    t0 = time.time()
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    saved = {}
    for m, (dr, key) in SRC.items():
        saved[m] = pickle.load(open(f"{dr}/{name}.pkl", "rb"))["subsets"][key]
    prob = {(m, c): [np.zeros(len(y)) for _ in range(n_repeats)] for m in SRC for c in CLFS}
    for r in range(n_repeats):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r + R0).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            for m in SRC:
                cols = saved[m][r * n_splits + fi]
                for c in CLFS:
                    prob[(m, c)][r][te] = np.clip(score(c, Xtr[:, cols], y[tr], Xte[:, cols], seed), 0.02, 0.98)
    res = {}
    for (m, c), ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[(m, c)] = {k: float(np.mean([x[k] for x in rows])) for k in ("gmean", "mcc", "aucpr", "auc")}
    pickle.dump(dict(name=name, minority=int(y.sum()), res=res), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: -os.path.getsize(p))
    Parallel(n_jobs=2)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
