"""Re-implementation of MF-CSO (Li, Xuan, Lin, Jiang, Ming & Tan, IEEE TEVC 27(4):802-816, 2023):
evolutionary multitasking with multiple filtering for high-dimensional feature selection.

Written from scratch in Python from the paper (arXiv:2212.08854). Settings the paper does not state were
taken from the authors' public MATLAB materials (github.com/gelin123/MF_CSO_Materials, files multitask.m,
divide_four_task.m, fitness2.m), which were read but not reused:
  * tasks: Relief-F (1 neighbour), term variance and |Pearson correlation|; for each, features whose weight
    exceeds the knee point of the sorted weights form an auxiliary task; a fourth task uses all features;
  * 70 particles per task (280 in all), 70 iterations; positions in [0, 1], feature selected if > 0.5;
  * within each task, random pairs give a winner and a loser; a loser learns from its winner, or (with
    probability 0.5) from a weighted combination of winners of the other tasks (weights 0.45 for the other
    filter tasks, 0.1 for the all-feature task; 0.45 each for the three filter tasks if the loser is in the
    all-feature task), plus its own winner; velocities are reset each iteration as in the code;
  * winners undergo polynomial mutation (distribution index 20, rate 1/D) with greedy replacement;
  * fitness = alpha^2 * balanced error of 1-NN under stratified 10-fold CV on the training fold
    + (1 - alpha) * (|S| / D) + (1 - alpha) * sigmoid(5 (d_w - d_b)), alpha = 0.999999; data min-max scaled;
  * the final subset is the best thresholded particle over all tasks, re-scored with the same fitness.
Deviation (for run time): the tiny-weight distance term uses Euclidean instead of city-block distances.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from . import indicators as ind

ALPHA = 0.999999


def knee_threshold(w):
    s = np.sort(w)
    n = len(s)
    x = np.arange(1, n + 1, dtype=float)
    if n < 3 or s[-1] == s[0]:
        return s[-1]
    k = (s[0] - s[-1]) / (1 - n)
    b = s[0] - k
    d = np.abs(k * x + b - s) / np.sqrt(k * k + 1)
    return s[int(np.argmax(d))]


class Fitness:
    def __init__(self, X, y, rng):
        lo, hi = X.min(0), X.max(0)
        self.Z = ((X - lo) / np.where(hi > lo, hi - lo, 1.0)).astype(np.float32)
        self.y = y.astype(int)
        self.D = X.shape[1]
        self.rng = rng
        self.evals = 0
        self.same = self.y[:, None] == self.y[None, :]
        np.fill_diagonal(self.same, False)
        self.diff = ~(self.y[:, None] == self.y[None, :])

    def __call__(self, mask):
        self.evals += 1
        s = int(mask.sum())
        if s == 0:
            return 1.0
        Zs = self.Z[:, mask]
        q = (Zs * Zs).sum(1)
        D2 = np.maximum(q[:, None] + q[None, :] - 2.0 * (Zs @ Zs.T), 0.0)
        dist = np.sqrt(D2) / np.sqrt(s)
        y = self.y
        nmin = np.bincount(y).min()
        kf = StratifiedKFold(min(10, max(2, nmin)), shuffle=True, random_state=int(self.rng.integers(1 << 30)))
        pred = np.empty_like(y)
        for tr, te in kf.split(Zs, y):
            j = np.argmin(dist[np.ix_(te, tr)], axis=1)
            pred[te] = y[tr][j]
        err = np.mean([np.mean(pred[y == c] != c) for c in (0, 1)])
        db = np.where(self.diff, dist, np.inf).min(1).mean()
        dw = np.where(self.same, dist, -np.inf).max(1)
        dw = np.where(np.isfinite(dw), dw, 0.0).mean()
        return ALPHA * ALPHA * err + (1 - ALPHA) * s / self.D + (1 - ALPHA) / (1 + np.exp(-5 * (dw - db)))


def pm(x, rng, dis=20):
    D = len(x)
    site = rng.random(D) < 1.0 / D
    mu = rng.random(D)
    off = np.clip(x.copy(), 0, 1)
    t = site & (mu <= 0.5)
    off[t] = off[t] + ((2 * mu[t] + (1 - 2 * mu[t]) * (1 - off[t]) ** (dis + 1)) ** (1 / (dis + 1)) - 1)
    t = site & (mu > 0.5)
    off[t] = off[t] + (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * off[t] ** (dis + 1)) ** (1 / (dis + 1)))
    return np.clip(off, 0, 1)


def mfcso(X, y, rng, n_per_task=70, n_iter=70, return_info=False):
    D = X.shape[1]
    lo, hi = X.min(0), X.max(0)
    Zs = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    w_rel = ind.relieff(X, y, k=1)
    w_tv = Zs.var(0)
    yc = y - y.mean()
    Xc = Zs - Zs.mean(0)
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    w_pcc = np.abs((Xc * yc[:, None]).sum(0) / np.where(den > 0, den, 1.0))
    tasks = np.vstack([w_rel > knee_threshold(w_rel), w_tv > knee_threshold(w_tv), w_pcc > knee_threshold(w_pcc),
                       np.ones(D, bool)])
    T = 4
    fit = Fitness(X, y, rng)
    N = n_per_task * T
    task_of = np.array([i % T for i in range(N)])            # interleaved assignment as in the code
    P = rng.random((N, D)).astype(np.float32)
    for i in range(N):
        P[i, ~tasks[task_of[i]]] = 0.0                         # init positions masked to the task
    F = np.array([fit((P[i] > 0.5) & tasks[task_of[i]]) for i in range(N)])
    members = [np.flatnonzero(task_of == t) for t in range(T)]
    for _ in range(n_iter):
        V = np.zeros((N, D), np.float32)
        winners, losers = [], []
        for t in range(T):
            perm = rng.permutation(members[t])
            for a, b in zip(perm[0::2], perm[1::2]):
                w, l = (a, b) if F[a] < F[b] else (b, a)
                winners.append(w); losers.append(l)
        winners = np.array(winners); losers = np.array(losers)
        wtask = task_of[winners]
        for w, l in zip(winners, losers):
            lt = task_of[l]
            if rng.random() > 0.5:                              # transfer branch (transp < rand)
                pick = {}
                for t in range(T):
                    if t == lt:
                        pick[t] = w
                    else:
                        cand = winners[wtask == t]
                        pick[t] = cand[rng.integers(len(cand))]
                if lt == 3:
                    comb = 0.45 * (P[pick[0]] + P[pick[1]] + P[pick[2]])
                else:
                    comb = 0.45 * (P[pick[0]] + P[pick[1]] + P[pick[2]]) + 0.1 * P[pick[3]] - 0.45 * P[pick[lt]]
                nv = rng.random() * V[w] + rng.random() * (comb - P[l]) + rng.random() * (P[w] - P[l])
            else:
                nv = rng.random() * V[w] + rng.random() * (P[w] - P[l])
            P[l] = np.clip(P[l] + nv + rng.random() * (nv - V[l]), 0, 1)
            V[l] = nv
            F[l] = fit((P[l] > 0.5) & tasks[lt])
        for w in winners:
            child = pm(P[w], rng)
            fc = fit((child > 0.5) & tasks[task_of[w]])
            if not fc > F[w]:
                P[w], F[w] = child, fc
    best, bf = None, np.inf
    for i in range(N):
        m = P[i] > 0.5
        if not m.any():
            m = m.copy(); m[0] = True
        f = fit(m)
        if f < bf:
            best, bf = m, f
    if return_info:
        return best, dict(evals=fit.evals, task_sizes=tasks.sum(1).tolist(), best_fit=float(bf))
    return best
