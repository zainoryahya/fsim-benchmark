"""Binary Grey Wolf Optimizer, single-task and multi-task (Component 2).

Multi-task design (multi-population EMT):
  * task 0 = main task, searching the whole candidate pool (K features);
  * tasks 1..T = auxiliary tasks, each restricted to its own feature group
    (a boolean mask over the pool). All tasks share the main task's
    encoding, so an auxiliary solution is directly a main-task solution.
  * knowledge transfer: each iteration, with probability rmp per main-task
    wolf, the delta leader is replaced by the alpha of a source auxiliary
    task.
  * source selection: uniform at random, or by a UCB1 bandit (the simplest
    reinforcement-learning agent) whose reward is the fraction of
    transferred wolves that improved their own fitness.

Equal evaluation budget: every method gets pop_total * iters evaluations.
"""
import numpy as np


def _binarize(P, rng, task_mask=None):
    prob = 1.0 / (1.0 + np.exp(-10.0 * (P - 0.5)))
    M = prob > rng.random(P.shape)
    if task_mask is not None:
        M &= task_mask[None, :]
    return M


def _init(n, K, rng, task_mask=None):
    rho = rng.uniform(0.05, 0.4, size=(n, 1))
    on = rng.random((n, K)) < rho
    P = np.where(on, rng.uniform(0.6, 1.0, (n, K)), rng.uniform(0.0, 0.4, (n, K)))
    if task_mask is not None:
        P[:, ~task_mask] = 0.0
    return P


class _Pack:
    """One sub-population (one task)."""

    def __init__(self, n, K, rng, fit, task_mask=None):
        self.tm = task_mask
        self.P = _init(n, K, rng, task_mask)
        self.M = _binarize(self.P, rng, task_mask)
        self.F = np.array([fit(m) for m in self.M])
        self._leaders()

    def _leaders(self):
        o = np.argsort(self.F)
        self.L = [self.P[o[i]].copy() for i in range(3)]
        self.LF = [self.F[o[i]] for i in range(3)]
        self.best_mask = self.M[o[0]].copy()
        self.best_fit = self.F[o[0]]

    def step(self, a, rng, fit, delta_override=None, override_rows=None):
        n, K = self.P.shape
        leaders = [self.L[0], self.L[1], self.L[2]]
        newP = np.zeros_like(self.P)
        for li, L in enumerate(leaders):
            Lm = np.broadcast_to(L, (n, K)).copy()
            if li == 2 and delta_override is not None and override_rows is not None:
                Lm[override_rows] = delta_override
            A = 2 * a * rng.random((n, K)) - a
            C = 2 * rng.random((n, K))
            newP += Lm - A * np.abs(C * Lm - self.P)
        newP = np.clip(newP / 3.0, 0.0, 1.0)
        if self.tm is not None:
            newP[:, ~self.tm] = 0.0
        newM = _binarize(newP, rng, self.tm)
        newF = np.array([fit(m) for m in newM])
        improved = newF < self.F
        self.P, self.M, self.F = newP, newM, newF
        # keep elite memory: leaders only replaced by better solutions
        cand = list(zip(self.LF, self.L)) + list(zip(self.F, self.P))
        cand.sort(key=lambda t: t[0])
        self.LF = [c[0] for c in cand[:3]]
        self.L = [c[1].copy() for c in cand[:3]]
        i = int(np.argmin(self.F))
        if self.F[i] < self.best_fit:
            self.best_fit, self.best_mask = self.F[i], self.M[i].copy()
        return improved


def bgwo(fit, K, rng, pop=50, iters=40):
    """Single-task binary GWO."""
    pack = _Pack(pop, K, rng, fit)
    for t in range(iters - 1):
        pack.step(2 - 2 * t / iters, rng, fit)
    return pack.best_mask, pack.best_fit, {}


class UCB1:
    def __init__(self, n_arms, c=1.0):
        self.n = np.zeros(n_arms)
        self.q = np.zeros(n_arms)
        self.c = c
        self.t = 0

    def select(self, rng):
        self.t += 1
        if (self.n == 0).any():
            return int(rng.choice(np.where(self.n == 0)[0]))
        ucb = self.q + self.c * np.sqrt(np.log(self.t) / self.n)
        return int(np.argmax(ucb))

    def update(self, arm, r):
        self.n[arm] += 1
        self.q[arm] += (r - self.q[arm]) / self.n[arm]


def mt_bgwo(fit, K, task_masks, rng, pop_main=20, pop_aux_total=30, iters=40,
            rmp=0.3, selector="uniform"):
    """Multi-task binary GWO. task_masks: list of boolean arrays (len K)."""
    T = len(task_masks)
    pa = max(4, pop_aux_total // T)
    main = _Pack(pop_main, K, rng, fit)
    aux = [_Pack(pa, K, rng, fit, tm) for tm in task_masks]
    bandit = UCB1(T) if selector == "ucb" else None
    picks = np.zeros(T, dtype=int)
    for t in range(iters - 1):
        a = 2 - 2 * t / iters
        for p in aux:
            p.step(a, rng, fit)
        rows = np.where(rng.random(pop_main) < rmp)[0]
        src = bandit.select(rng) if bandit else int(rng.integers(T))
        picks[src] += 1
        improved = main.step(a, rng, fit, delta_override=aux[src].L[0], override_rows=rows)
        if bandit is not None:
            r = improved[rows].mean() if len(rows) else 0.0
            bandit.update(src, float(r))
    packs = [main] + aux
    j = int(np.argmin([p.best_fit for p in packs]))
    info = {"picks": picks.tolist(), "winner_task": j}
    if bandit is not None:
        info["q"] = bandit.q.round(3).tolist()
    return packs[j].best_mask, packs[j].best_fit, info
