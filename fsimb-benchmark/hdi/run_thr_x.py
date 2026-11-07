"""Phase 5d-1: is SMOTE better than simply moving the decision threshold?

Same setting as run_leak2 (top-20 filter inside the fold, 5-fold x 2 repeats, pooled out-of-fold metrics,
filters F and H, classifiers 5-NN and logistic regression) on the 34 non-ceiling v2 datasets, real labels.
Variants (all with the filter fitted inside the training fold):
  base        threshold 0.5, no resampling
  prior       threshold = training minority share  (predict minority when p > share)
  tuned       threshold chosen on inner 5-fold CV probabilities of the training fold to maximise G-mean
  smote_fs    SMOTE -> FS, threshold 0.5
  fs_smote    FS -> SMOTE, threshold 0.5
  weighted    (logistic regression only) class_weight = balanced, threshold 0.5
"""
import numpy as np, warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob
import os as _os
R0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')

warnings.filterwarnings("ignore")
K = 20
FS = {"F": ind.anova_f, "H": ind.hellinger}
VARIANTS = ["base", "prior", "tuned", "smote_fs", "fs_smote", "weighted"]
GRID = np.arange(0.05, 0.951, 0.025)


def smote(X, y, seed):
    return SMOTE(k_neighbors=min(5, int(y.sum()) - 1), random_state=seed).fit_resample(X, y)


def predict(clf, Xtr, ytr, Xte, weighted=False):
    if clf == "knn":
        return knn_prob(sqdist(Xte, Xtr), ytr)
    return LogisticRegression(C=1.0, max_iter=500, class_weight="balanced" if weighted else None).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def gm(y, pred):
    return np.sqrt(pred[y == 1].mean() * (~pred[y == 0]).mean())


def tune_threshold(clf, X, y, seed):
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[b] = predict(clf, X[a], y[a], X[b])
    sc = np.array([gm(y, p > t) for t in GRID])
    best = np.flatnonzero(sc >= sc.max() - 1e-12)
    pi = y.mean()
    return float(GRID[best[np.argmin(np.abs(GRID[best] - pi))]])


def metrics(y, p, t):
    pred = p > t
    return dict(mcc=float(matthews_corrcoef(y, pred)), auc=float(roc_auc_score(y, p)), gmean=float(gm(y, pred)),
                recall=float(pred[y == 1].mean()))


def job(path, seed=7, repeats=NREP):
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    out = {}
    for fs in FS:
        acc = {}
        for r in range(repeats):
            P = {(v, c): np.zeros(len(y)) for v in VARIANTS for c in ("knn", "lr")}
            T = {k: np.full(len(y), 0.5) for k in P}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r + R0).split(X, y):
                mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
                Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
                cols = np.argsort(-FS[fs](Xtr, ytr))[:K]
                Xs, ys = smote(Xtr, ytr, seed + r + R0)
                cols_s = np.argsort(-FS[fs](Xs, ys))[:K]
                Xf, yf = smote(Xtr[:, cols], ytr, seed + r + R0)
                pi = ytr.mean()
                for c in ("knn", "lr"):
                    p = predict(c, Xtr[:, cols], ytr, Xte[:, cols])
                    P[("base", c)][te] = p
                    P[("prior", c)][te] = p; T[("prior", c)][te] = pi
                    P[("tuned", c)][te] = p; T[("tuned", c)][te] = tune_threshold(c, Xtr[:, cols], ytr, seed + r + R0)
                    P[("smote_fs", c)][te] = predict(c, Xs[:, cols_s], ys, Xte[:, cols_s])
                    P[("fs_smote", c)][te] = predict(c, Xf, yf, Xte[:, cols])
                    if c == "lr":
                        P[("weighted", c)][te] = predict(c, Xtr[:, cols], ytr, Xte[:, cols], weighted=True)
            for k in P:
                if k == ("weighted", "knn"):
                    continue
                acc.setdefault(k, []).append(metrics(y, P[k], T[k]))
        for k, v in acc.items():
            out[(fs,) + k] = {m: float(np.mean([x[m] for x in v])) for m in v[0]}
    return path.split("/")[-1][:-4], out
