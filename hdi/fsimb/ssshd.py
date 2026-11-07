"""sssHD (Fu, Wu, Zong & Pan 2020, BMC Bioinformatics 21:121), re-implemented from the paper and the authors'
R code (Additional file 3: functions sssHD, cvHellinger, HellingerDist, cvimbalance).

As in the R code: subsamples with r1 = r0 = 0.5 (class ratio kept); on each subsample a LASSO linear
SVM (sparseSVM: huberized hinge, gamma = 0.1, alpha = 1) is fitted along a lambda path; the lambda that maximises the
binormal Hellinger distance between the cross-validated decision scores of the two classes is chosen; the
non-zero coefficients of the full-subsample fit at that lambda are counted. Features are ranked by inclusion
frequency.

Differences from the R code (documented in the manuscript): K = 50 subsamples were used in the experiments
(run_ssshd.py, env K_SUB) instead of the R code's ITER = 100 (the function default here is 100); the scores are cross-validated with the code's
stratified 5-fold option (cvHellinger, LOO = FALSE) instead of its default leave-one-out, for run time; the path has
NLAM lambda values from lambda_max to 0.05 lambda_max (sparseSVM's lower limit when n < p) instead of 100; ties in
frequency are broken by the mean |coefficient| over subsamples, and features never selected follow in Hellinger
order (the R code returns only the frequencies); the data are standardised once per outer training fold, and the
subsamples are not re-standardised (sparseSVM standardises the full-subsample fit by default); the intercept-only
start used to set lambda_max is found by a grid search (the KKT loop then uses the fitted intercept).

Model (per fit):  min_b,b0  (1/n) sum_i h(y_i (x_i b + b0)) + lam [ alpha ||b||_1 + (1-alpha)/2 ||b||_2^2 ],
y in {-1, +1}, h(t) = 0 (t >= 1), (1-t)^2/(2 delta) (1-delta < t < 1), 1 - t - delta/2 (t <= 1-delta).
Solved by FISTA on a working set chosen by the sequential strong rule, with a KKT check over all features.
"""
import numpy as np
from . import indicators as ind

DELTA = 0.1
NLAM = 30


def _hgrad(m):
    """derivative of the huberized hinge with respect to the margin m"""
    g = np.zeros_like(m)
    mid = (m > 1 - DELTA) & (m < 1)
    g[mid] = -(1 - m[mid]) / DELTA
    g[m <= 1 - DELTA] = -1.0
    return g


def _fit(X, s, lam, alpha, b, b0, L, iters=5000, tol=1e-6):
    """FISTA with gradient-based adaptive restart on columns X (n x q); s = labels in {-1,+1}; warm start b, b0."""
    n = len(s)
    step = 1.0 / (L + lam * (1 - alpha))
    zb, zb0, t = b.copy(), b0, 1.0
    for _ in range(iters):
        m = s * (X @ zb + zb0)
        g = _hgrad(m) * s / n
        gb = X.T @ g + lam * (1 - alpha) * zb
        nb = zb - step * gb
        nb = np.sign(nb) * np.maximum(np.abs(nb) - step * lam * alpha, 0.0)
        nb0 = zb0 - step * g.sum()
        diff = max(np.max(np.abs(nb - b)) if len(b) else 0.0, abs(nb0 - b0))
        if (zb - nb) @ (nb - b) + (zb0 - nb0) * (nb0 - b0) > 0:     # restart momentum
            t = 1.0
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        zb = nb + (t - 1) / tn * (nb - b)
        zb0 = nb0 + (t - 1) / tn * (nb0 - b0)
        b, b0, t = nb, nb0, tn
        if diff < tol * max(1.0, np.max(np.abs(b)) if len(b) else 1.0):
            break
    return b, b0


def binormal_hd(t0, t1):
    """Hellinger distance between normal fits of two score samples (HellingerDist in the R code; sd with n-1)."""
    if len(t0) < 2 or len(t1) < 2:
        return 0.0
    s0, s1 = np.std(t0, ddof=1), np.std(t1, ddof=1)
    if s0 < 1e-12 or s1 < 1e-12:
        return 0.0
    v = s0 * s0 + s1 * s1
    d2 = 2 - 2 * np.sqrt(2 * s0 * s1 / v) * np.exp(-(np.mean(t1) - np.mean(t0)) ** 2 / (4 * v))
    return float(np.sqrt(max(d2, 0.0)))


def _hloss(m):
    return np.where(m >= 1, 0, np.where(m > 1 - DELTA, (1 - m) ** 2 / (2 * DELTA), 1 - m - DELTA / 2))


def _start(X, s):
    """intercept-only solution and the gradient at b = 0"""
    grid = np.linspace(-1.5, 1.5, 301)
    b0 = float(grid[int(np.argmin([np.mean(_hloss(s * c)) for c in grid]))])
    return b0, X.T @ (_hgrad(s * b0) * s / len(s))


def lam_grid(X, y, alpha=1.0, nlam=NLAM, ratio=0.05):
    s = np.where(y == 1, 1.0, -1.0)
    _, g0 = _start(X, s)
    lmax = np.max(np.abs(g0)) / alpha
    return lmax * np.logspace(0, np.log10(ratio), nlam)


def path(X, y, lams, alpha=1.0):
    """coefficients (nlam x p) and intercepts along the given lambda path (warm starts, strong rule)"""
    n, p = X.shape
    s = np.where(y == 1, 1.0, -1.0)
    b0, grad = _start(X, s)
    b = np.zeros(p)
    B, B0 = np.zeros((len(lams), p)), np.zeros(len(lams))
    prev = None
    for i, lam in enumerate(lams):
        thr = alpha * (2 * lam - prev) if prev is not None else alpha * lam
        strong = (np.abs(grad) >= thr) | (b != 0)
        while True:
            A = np.flatnonzero(strong)
            if len(A):
                XA = X[:, A]
                L = (np.linalg.norm(XA, 2) ** 2 + n) / (n * DELTA)
                bA, b0 = _fit(XA, s, lam, alpha, b[A], b0, L)
                b = np.zeros(p); b[A] = bA
                m = s * (XA @ bA + b0)
            else:
                m = s * b0
            grad = X.T @ (_hgrad(m) * s / n)
            viol = (~strong) & (np.abs(grad) > alpha * lam * (1 + 1e-6))
            if not viol.any():
                break
            strong |= viol
        B[i], B0[i] = b, b0
        prev = lam
    return B, B0


def cv_hellinger(X, y, rng, alpha=1.0, folds=5):
    """cvHellinger (LOO = FALSE): lambda path from the full data, stratified 5-fold scores, argmax of HD."""
    lams = lam_grid(X, y, alpha)
    fitted = np.zeros((len(y), len(lams)))
    fid = np.zeros(len(y), int)
    for c in (0, 1):                                   # cvimbalance: split each class into folds at random
        idx = rng.permutation(np.flatnonzero(y == c))
        fid[idx] = np.arange(len(idx)) % folds
    for k in range(folds):
        te = fid == k
        Bk, B0k = path(X[~te], y[~te], lams, alpha)
        fitted[te] = X[te] @ Bk.T + B0k
    hd = np.array([binormal_hd(fitted[y == 0, i], fitted[y == 1, i]) for i in range(len(lams))])
    best = int(np.argmax(hd))
    B, _ = path(X, y, lams[: best + 1], alpha)
    return B[-1], hd[best], best


def ssshd_rank(X, y, rng, K=100, frac=0.5, alpha=1.0, return_info=False):
    """Rank all columns of X by sssHD inclusion frequency (best first)."""
    n, p = X.shape
    freq = np.zeros(p)
    mag = np.zeros(p)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    sizes, hds, ilam = [], [], []
    for _ in range(K):
        a = np.concatenate([rng.choice(idx1, max(2, int(round(frac * len(idx1)))), replace=False),
                            rng.choice(idx0, max(2, int(round(frac * len(idx0)))), replace=False)])
        b, hd, bi = cv_hellinger(X[a], y[a], rng, alpha)
        nz = b != 0
        freq += nz
        mag += np.abs(b)
        sizes.append(int(nz.sum())); hds.append(hd); ilam.append(bi)
    hel = np.argsort(-ind.hellinger(X, y), kind="stable")
    sel = np.flatnonzero(freq > 0)
    main = sel[np.lexsort((-mag[sel], -freq[sel]))]
    seen = np.zeros(p, bool); seen[main] = True
    order = np.concatenate([main, hel[~seen[hel]]])
    if return_info:
        return order, dict(freq=freq / K, sizes=sizes, hds=hds, ilam=ilam, n_selected=len(main))
    return order
