"""Phase 5d-2: Tomek links and SMOTE+Tomek versus threshold moving (added after reviewer-type question).

Same setting, folds and seed as run_thr.py (top-20 filter inside the fold, 5-fold x 2 repeats, seed 7, pooled
out-of-fold metrics, filters F and H, classifiers 5-NN and logistic regression). All resampling on the
training fold only. Variants:
  base, prior      as in run_thr (recomputed here so every contrast is paired within this run)
  tomek_fs         Tomek-link removal (majority member of each link) -> FS, threshold 0.5
  fs_tomek         FS -> Tomek on the selected columns, threshold 0.5
  smtk_fs          SMOTE + Tomek (imblearn SMOTETomek) -> FS, threshold 0.5
  fs_smtk          FS -> SMOTE + Tomek on the selected columns, threshold 0.5
Also records the share of training samples Tomek removed.
Usage: imported by launch_tomek.py
"""
import numpy as np, warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold
from run_thr_x import FS, K, predict, metrics
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

warnings.filterwarnings("ignore")
VARIANTS = ["base", "prior", "tomek_fs", "fs_tomek", "smtk_fs", "fs_smtk"]


def tomek(X, y):
    return TomekLinks(sampling_strategy="majority").fit_resample(X, y)


def smtk(X, y, seed):
    sm = SMOTE(k_neighbors=min(5, int(y.sum()) - 1), random_state=seed)
    return SMOTETomek(smote=sm, tomek=TomekLinks(sampling_strategy="all"), random_state=seed).fit_resample(X, y)


def job(path, seed=7, repeats=NREP):
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    out, removed = {}, {"full": [], "selected": []}
    for fs in FS:
        acc = {}
        for r in range(repeats):
            P = {(v, c): np.zeros(len(y)) for v in VARIANTS for c in ("knn", "lr")}
            T = {k: np.full(len(y), 0.5) for k in P}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r + R0).split(X, y):
                mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
                Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
                cols = np.argsort(-FS[fs](Xtr, ytr))[:K]
                Xt, yt = tomek(Xtr, ytr)
                cols_t = np.argsort(-FS[fs](Xt, yt))[:K]
                Xft, yft = tomek(Xtr[:, cols], ytr)
                Xs, ys = smtk(Xtr, ytr, seed + r + R0)
                cols_s = np.argsort(-FS[fs](Xs, ys))[:K]
                Xfs, yfs = smtk(Xtr[:, cols], ytr, seed + r + R0)
                removed["full"].append(1 - len(yt) / len(ytr)); removed["selected"].append(1 - len(yft) / len(ytr))
                pi = ytr.mean()
                for c in ("knn", "lr"):
                    p = predict(c, Xtr[:, cols], ytr, Xte[:, cols])
                    P[("base", c)][te] = p
                    P[("prior", c)][te] = p; T[("prior", c)][te] = pi
                    P[("tomek_fs", c)][te] = predict(c, Xt[:, cols_t], yt, Xte[:, cols_t])
                    P[("fs_tomek", c)][te] = predict(c, Xft, yft, Xte[:, cols])
                    P[("smtk_fs", c)][te] = predict(c, Xs[:, cols_s], ys, Xte[:, cols_s])
                    P[("fs_smtk", c)][te] = predict(c, Xfs, yfs, Xte[:, cols])
            for k in P:
                acc.setdefault(k, []).append(metrics(y, P[k], T[k]))
        for k, v in acc.items():
            out[(fs,) + k] = {m: float(np.mean([x[m] for x in v])) for m in v[0]}
    out["removed"] = {k: float(np.mean(v)) for k, v in removed.items()}
    return path.split("/")[-1][:-4], out
