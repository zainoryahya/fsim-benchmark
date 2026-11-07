"""Phase 2: inner estimators of a subset's quality, all sharing one distance matrix.

Classifier: 5-NN with inverse-distance weights. prob = weighted share of
minority neighbours, clipped to [0.02, 0.98] so log loss is finite. The label
is prob > 0.5. (The prototype used plain 3-NN; its probabilities take only
four values, which would make every probability-based measure tie heavily.)

Estimators (each returns the 12 measures, averaged over its repeats):
  loo    leave-one-out
  cv5x2  stratified 5-fold, 2 repeats
  cv5x5  stratified 5-fold, 5 repeats
  boot   out-of-bag bootstrap (b0), B = 25 resamples; every sample's
         out-of-bag probabilities are averaged, then the measures are taken
         on those pooled scores. The .632+ estimator needs the apparent
         (resubstitution) score; for a lazy learner that score is perfect,
         the relative overfitting rate is clipped to 1 and .632+ equals b0,
         so b0 is what is implemented.

Measures on a pooled prediction set follow Forman & Scholz (2010): pool the
out-of-fold predictions of one repeat, take the measure once, then average
over repeats.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .measures import measure_vector, MEASURES

K_NN = 5
ESTIMATORS = ["loo", "cv5x2", "cv5x5", "boot"]
PCLIP = (0.02, 0.98)


def sqdist(Xa, Xb=None):
    """Squared Euclidean distances between rows."""
    if Xb is None:
        s = (Xa ** 2).sum(1)
        D = s[:, None] + s[None, :] - 2.0 * Xa @ Xa.T
    else:
        D = (Xa ** 2).sum(1)[:, None] + (Xb ** 2).sum(1)[None, :] - 2.0 * Xa @ Xb.T
    np.maximum(D, 0.0, out=D)
    return D


def knn_prob(D, yref):
    """D: queries x references squared distances."""
    k = min(K_NN, D.shape[1])
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]
    d = np.sqrt(np.take_along_axis(D, nn, axis=1))
    w = 1.0 / (d + 1e-6)
    p = (w * yref[nn]).sum(1) / w.sum(1)
    return np.clip(p, *PCLIP)


def estimate(D, y, rng, which=ESTIMATORS):
    """D: n x n squared distances of the training set under one subset.
    Returns {estimator: vector of 12 measures}."""
    n = len(y)
    out = {}
    if "loo" in which:
        Dl = D.copy()
        np.fill_diagonal(Dl, np.inf)
        out["loo"] = measure_vector(y, knn_prob(Dl, y))
    for name, reps in (("cv5x2", 2), ("cv5x5", 5)):
        if name not in which:
            continue
        acc = []
        for r in range(reps):
            p = np.zeros(n)
            seed = int(rng.integers(1 << 30))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
                p[te] = knn_prob(D[np.ix_(te, tr)], y[tr])
            acc.append(measure_vector(y, p))
        out[name] = np.mean(acc, axis=0)
    if "boot" in which:
        ssum, cnt = np.zeros(n), np.zeros(n)
        for _ in range(25):
            b = rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), b)
            if len(oob) == 0:
                continue
            ssum[oob] += knn_prob(D[np.ix_(oob, b)], y[b])
            cnt[oob] += 1
        ok = cnt > 0
        pb = np.full(n, 0.5)
        pb[ok] = ssum[ok] / cnt[ok]
        out["boot"] = measure_vector(y, np.clip(pb, *PCLIP))
    return out
