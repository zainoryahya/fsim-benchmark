"""Additional feature-ranking baselines (added after internal review), all fitted on the training fold only.

Every function returns a full ranking (best first) of the columns of X; the evaluation takes the top k.
y is 0/1 with 1 = minority. X is standardised with training statistics.

  l1_rank    L1-penalised logistic regression (class-balanced). C is raised along a grid until at least
             KMAX coefficients are non-zero; features are ranked by |coefficient| at that C. Ties at zero
             (if fewer than KMAX are non-zero at the largest C) are broken by the Hellinger score.
  en_rank    same with an elastic-net penalty (l1_ratio = 0.5, saga solver)
  mrmr_rank  minimum-redundancy maximum-relevance (MID form): mutual information on equal-frequency bins
             (5 bins), greedy selection over the 500 most relevant features, rest by relevance
  rfimp_rank random forest (500 trees, balanced subsamples) impurity importance
  rfperm_rank permutation importance (drop in AUC-ROC) of a random forest refitted on the 200 features with
             the largest impurity importance; two-fold inner split (fit on one half, permute on the other,
             then swap); remaining features follow by impurity importance
"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from . import indicators as ind

KMAX = 40
C_GRID = np.logspace(-3, 2, 16)


def _fill(order_main, tiebreak):
    """order_main: indices with a strictly informative score; append the rest by tiebreak order."""
    seen = np.zeros(len(tiebreak), bool); seen[order_main] = True
    return np.concatenate([order_main, tiebreak[~seen[tiebreak]]])


def _nnz_coef(make, C, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.abs(make(C).fit(X, y).coef_.ravel())
    return c


def _l1(C):
    return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, class_weight="balanced", max_iter=2000)


def _en(C):
    return LogisticRegression(l1_ratio=0.5, solver="saga", C=C, class_weight="balanced", max_iter=1000, tol=1e-3)


def _smallest_C(make, X, y, start=0):
    """Smallest grid C whose model has >= KMAX non-zero coefficients (nnz grows with C), searched from `start`."""
    i = start
    c = _nnz_coef(make, C_GRID[i], X, y)
    if (c > 0).sum() >= KMAX:
        while i > 0:
            c2 = _nnz_coef(make, C_GRID[i - 1], X, y)
            if (c2 > 0).sum() < KMAX:
                break
            i, c = i - 1, c2
    else:
        while i < len(C_GRID) - 1:
            i += 1
            c = _nnz_coef(make, C_GRID[i], X, y)
            if (c > 0).sum() >= KMAX:
                break
    return i, c


def _rank_from_coef(c, X, y):
    hell = np.argsort(-ind.hellinger(X, y))
    nz = np.flatnonzero(c > 0)
    return _fill(nz[np.argsort(-c[nz], kind="stable")], hell), int(len(nz))


def l1_rank(X, y):
    i, c = _smallest_C(_l1, X, y, 0)
    r, nnz = _rank_from_coef(c, X, y)
    return r, nnz, i


def en_rank(X, y, start=None):
    """start: grid index to begin the search (the L1 solution's index is a good start; the result is the same
    smallest-C rule)."""
    i, c = _smallest_C(_en, X, y, 0 if start is None else start)
    r, nnz = _rank_from_coef(c, X, y)
    return r, nnz, i


def _disc(X, b=5):
    r = ind._ranks(X)
    return np.minimum(((r - 1.0) * b / X.shape[0]).astype(int), b - 1)


def _mi_cols(A, v, b):
    """MI between each column of A (n x m, ints in [0,b)) and vector v (ints in [0,bv))."""
    n, m = A.shape
    bv = int(v.max()) + 1
    idx = A * bv + v[:, None]
    out = np.empty(m)
    for j in range(m):
        joint = np.bincount(idx[:, j], minlength=b * bv).reshape(b, bv) / n
        pa, pv = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        nzm = joint > 0
        out[j] = (joint[nzm] * np.log(joint[nzm] / (pa @ pv)[nzm])).sum()
    return out


def mrmr_rank(X, y, pool=500, b=5):
    Q = _disc(X, b)
    rel = _mi_cols(Q, y.astype(int), b)
    by_rel = np.argsort(-rel, kind="stable")
    cand = list(by_rel[: min(pool, X.shape[1])])
    sel = [cand.pop(0)]
    red_sum = np.zeros(len(cand))
    while len(sel) < min(KMAX, X.shape[1]) and cand:
        red_sum += _mi_cols(Q[:, cand], Q[:, sel[-1]], b)
        score = rel[cand] - red_sum / len(sel)
        j = int(np.argmax(score))
        sel.append(cand.pop(j)); red_sum = np.delete(red_sum, j)
    return _fill(np.array(sel), by_rel)


def rfimp_rank(X, y, seed=0):
    rf = RandomForestClassifier(500, class_weight="balanced_subsample", max_features="sqrt",
                                random_state=seed, n_jobs=1).fit(X, y)
    return np.argsort(-rf.feature_importances_, kind="stable"), rf.feature_importances_


def rfperm_rank(X, y, seed=0, pool=200):
    imp_order, _ = rfimp_rank(X, y, seed)
    top = imp_order[:pool]
    drop = np.zeros(len(top))
    rng = np.random.default_rng(seed)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
        rf = RandomForestClassifier(200, class_weight="balanced_subsample", random_state=seed, n_jobs=1).fit(X[a][:, top], y[a])
        Xb = X[b][:, top].copy()
        base = roc_auc_score(y[b], rf.predict_proba(Xb)[:, 1])
        for j in range(len(top)):
            keep = Xb[:, j].copy()
            Xb[:, j] = rng.permutation(keep)
            drop[j] += base - roc_auc_score(y[b], rf.predict_proba(Xb)[:, 1])
            Xb[:, j] = keep
    main = top[np.argsort(-drop, kind="stable")]
    return _fill(main, imp_order)
