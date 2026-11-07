"""Per-feature relevance indicators for binary, imbalanced, high-dimensional data.

Convention: y is 0/1 with 1 = minority class. Every function returns a
non-negative score per column of X; larger = more relevant.

Indicators split into two families:
  * prevalence-weighted (ANOVA F): the majority class dominates the estimate.
  * minority-aware (Hellinger, |AUC-0.5|, balanced Fisher): each class
    contributes equally, whatever the imbalance ratio.
Relief-F is included as the literature baseline for EMT task formation.
"""
import warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore", message=r"(?s)Features .* are constant")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


def anova_f(X, y):
    """Classic ANOVA F statistic (prevalence-weighted)."""
    f, _ = f_classif(X, y)
    return np.nan_to_num(f, nan=0.0)


def _ranks(X):
    """Column-wise average ranks 1..n. Ties share a rank, which matters for
    sparse or discrete features (text counts, binary pixels): breaking ties
    by row order would inject label-independent noise into the scores."""
    return rankdata(X, axis=0, method="average")


def auc_score(X, y):
    """|AUC - 0.5| per feature. Rank based and prevalence independent."""
    r = _ranks(X)
    m = y.sum()
    n0 = len(y) - m
    u = r[y == 1].sum(axis=0) - m * (m + 1) / 2.0
    return np.abs(u / (m * n0) - 0.5)


def balanced_fisher(X, y):
    """(mu1 - mu0)^2 / (s1^2 + s0^2): both classes weigh the same."""
    a, b = X[y == 1], X[y == 0]
    num = (a.mean(0) - b.mean(0)) ** 2
    den = a.var(0, ddof=1) + b.var(0, ddof=1) + 1e-12
    return num / den


def hellinger(X, y, bins=None):
    """Hellinger distance between the two class-conditional histograms
    (Yin et al. 2013). Bins are equal-frequency on the pooled data, so the
    score is insensitive to class skew."""
    n, p = X.shape
    b = bins or max(3, int(np.sqrt(n)))
    # equal-frequency bins on average ranks: tied values land in one bin
    q = np.minimum(((_ranks(X) - 1.0) * b / n).astype(int), b - 1)   # n x p bin index
    off = (np.arange(p) * b)[None, :]
    out = []
    for c in (0, 1):
        idx = (q[y == c] + off).ravel()
        h = np.bincount(idx, minlength=p * b).reshape(p, b).astype(float)
        out.append(h / h.sum(1, keepdims=True))
    return np.sqrt(((np.sqrt(out[1]) - np.sqrt(out[0])) ** 2).sum(1))


def relieff(X, y, k=5):
    """Relief-F for two classes (Kononenko 1994), features scaled to [0,1]."""
    lo, hi = X.min(0), X.max(0)
    Z = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    D = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(D, np.inf)
    n = len(y)
    w = np.zeros(X.shape[1])
    for i in range(n):
        same = np.where(y == y[i])[0]
        diff = np.where(y != y[i])[0]
        h = same[np.argsort(D[i, same])[:k]]
        m = diff[np.argsort(D[i, diff])[:k]]
        w -= np.abs(Z[h] - Z[i]).mean(0)
        w += np.abs(Z[m] - Z[i]).mean(0)
    return w / n


MINORITY_AWARE = {"hellinger": hellinger, "auc": auc_score, "bfisher": balanced_fisher}
