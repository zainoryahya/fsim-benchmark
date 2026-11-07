"""Component 4 (stability) and Component 5 (statistical comparison)."""
import numpy as np
from scipy import stats


def nogueira_stability(subsets, d):
    """Nogueira, Sechidis & Brown (JMLR 2018) stability estimator.
    Works with subsets of different sizes. 1 = identical subsets,
    ~0 = what random selection of the same sizes would give."""
    M = len(subsets)
    Z = np.zeros((M, d), bool)
    for i, s in enumerate(subsets):
        Z[i, s] = True
    pf = Z.mean(0)
    kbar = Z.sum(1).mean()
    s2 = M / (M - 1) * pf * (1 - pf)
    denom = (kbar / d) * (1 - kbar / d)
    return float(1 - s2.mean() / denom) if denom > 0 else float("nan")


def friedman(table):
    """table: datasets x methods (higher = better). Returns stat, p, mean ranks."""
    ranks = np.array([stats.rankdata(-row) for row in table])
    st, p = stats.friedmanchisquare(*table.T)
    return float(st), float(p), ranks.mean(0)


# Studentised range / sqrt(2) at alpha = 0.05 (Demsar 2006, Table 5a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def nemenyi_cd(k, N):
    return Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))


def holm_wilcoxon(table, names, control):
    """Wilcoxon signed-rank of control vs each other method, Holm-adjusted."""
    c = names.index(control)
    rows = []
    for j, n in enumerate(names):
        if j == c:
            continue
        d = table[:, c] - table[:, j]
        p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue
        rows.append([n, float(np.median(d)), float(p)])
    rows.sort(key=lambda r: r[2])
    m = len(rows)
    adj, run = [], 0.0
    for i, r in enumerate(rows):
        run = max(run, min(1.0, (m - i) * r[2]))
        adj.append(run)
    return [r + [a] for r, a in zip(rows, adj)]


def bayes_signed_rank(x, y, rope=0.01, s=0.5, n_samples=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al., JMLR 2017).
    Returns P(x < y), P(practically equal), P(x > y) with a ROPE on the
    difference x - y."""
    z = np.concatenate([[0.0], np.asarray(x) - np.asarray(y)])
    n = len(z)
    S = (z[:, None] + z[None, :]) / 2.0
    left, right = (S < -rope).astype(float), (S > rope).astype(float)
    rng = np.random.default_rng(seed)
    alpha = np.ones(n)
    alpha[0] = s
    W = rng.dirichlet(alpha, size=n_samples)
    pl = np.einsum("ki,ij,kj->k", W, left, W)
    pr = np.einsum("ki,ij,kj->k", W, right, W)
    pe = 1 - pl - pr
    win = np.argmax(np.stack([pl, pe, pr], 1), 1)
    return [float((win == i).mean()) for i in range(3)]
