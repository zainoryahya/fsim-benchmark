"""Nested evaluation pipeline (Components 3 and 5).

Outer loop: repeated stratified K-fold. EVERYTHING that looks at labels —
standardisation, the candidate pool filter, the indicators that form the
tasks, the wrapper search, any resampling — is fitted on the outer training
fold only. The outer test fold is touched once, for the final prediction.

Metrics are computed on the pooled out-of-fold predictions of each repeat
(Forman & Scholz 2010), which is more stable than averaging per-fold
G-means when a test fold holds only 3-6 minority samples.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from . import indicators as ind
from .fitness import LOOKNNFitness, gmean, mcc
from .optimizer import bgwo, mt_bgwo

POOL = 200        # candidate pool size after the neutral prefilter
GROUP = 50        # size of each auxiliary task's feature group
FILTER_K = 20     # subset size for the filter-only baselines


def knn_predict(Xtr, ytr, Xte, k=3):
    s_tr, s_te = (Xtr ** 2).sum(1), (Xte ** 2).sum(1)
    D = s_te[:, None] + s_tr[None, :] - 2 * Xte @ Xtr.T
    nn = np.argpartition(D, k, axis=1)[:, :k]
    prob = ytr[nn].mean(1)
    return (prob > 0.5).astype(int), prob


def select(method, Xtr, ytr, rng):
    """Return selected column indices (into Xtr) and an info dict."""
    p = Xtr.shape[1]
    if method == "F-top20":
        return np.argsort(-ind.anova_f(Xtr, ytr))[:FILTER_K], {}
    if method == "H-top20":
        return np.argsort(-ind.hellinger(Xtr, ytr))[:FILTER_K], {}

    pool = np.argsort(-ind.anova_f(Xtr, ytr))[: min(POOL, p)]
    Xp = Xtr[:, pool]
    K = Xp.shape[1]

    def group(score):
        m = np.zeros(K, bool)
        m[np.argsort(-score)[:GROUP]] = True
        return m

    if method == "GWO-Acc":
        fit = LOOKNNFitness(Xp, ytr, metric="acc")
        mask, f, info = bgwo(fit, K, rng)
    elif method == "GWO-Gm":
        fit = LOOKNNFitness(Xp, ytr, metric="gmean")
        mask, f, info = bgwo(fit, K, rng)
    elif method == "EMT-Relief":
        fit = LOOKNNFitness(Xp, ytr, metric="gmean")
        tasks = [group(ind.relieff(Xp, ytr))]
        mask, f, info = mt_bgwo(fit, K, tasks, rng, selector="uniform")
    elif method in ("EMT-MinAware", "EMT-MinAware-RL"):
        fit = LOOKNNFitness(Xp, ytr, metric="gmean")
        tasks = [group(fn(Xp, ytr)) for fn in ind.MINORITY_AWARE.values()]
        sel = "ucb" if method.endswith("RL") else "uniform"
        mask, f, info = mt_bgwo(fit, K, tasks, rng, selector=sel)
    else:
        raise ValueError(method)
    info["evals"] = fit.evals
    return pool[mask], info


def run_dataset(name, X, y, methods, n_splits=5, n_repeats=3, seed=0):
    """Returns per-repeat metrics and the selected subsets (original indices)."""
    out = {m: {"gmean": [], "mcc": [], "aucpr": [], "recall_min": [],
               "nfeat": [], "subsets": [], "info": []} for m in methods}
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        oof = {m: (np.zeros(len(y), int), np.zeros(len(y))) for m in methods}
        nf = {m: [] for m in methods}
        for fi, (tr, te) in enumerate(skf.split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            for mi, m in enumerate(methods):
                rng = np.random.default_rng([seed, r, fi, mi])
                cols, info = select(m, Xtr, y[tr], rng)
                pred, prob = knn_predict(Xtr[:, cols], y[tr], Xte[:, cols])
                oof[m][0][te], oof[m][1][te] = pred, prob
                nf[m].append(len(cols))
                out[m]["subsets"].append(np.asarray(cols))
                out[m]["info"].append(info)
        for m in methods:
            pred, prob = oof[m]
            out[m]["gmean"].append(gmean(y, pred))
            out[m]["mcc"].append(mcc(y, pred))
            out[m]["aucpr"].append(average_precision_score(y, prob))
            out[m]["recall_min"].append(float(pred[y == 1].mean()))
            out[m]["nfeat"].append(float(np.mean(nf[m])))
    return out
