"""Wrapper checks requested in review, on the same outer folds (stratified 5-fold, seeds 2027 + r).

All searches use the default fitness (balanced Brier of LOO 5-NN, alpha = 0.99, K = 200) and a candidate pool of
the 200 best features of the training fold.
  bgwo_cap20, bgwo_nocap   binary grey wolf optimiser, 50 wolves x 40 iterations (2000 evaluations), as before
  ga_cap20, ga_nocap       binary genetic algorithm with the same budget (population 50, 40 generations; binary
                           tournament selection, uniform crossover with probability 0.9, bit-flip mutation 1/K,
                           one elite), same initialisation as BGWO
  bgwoF_cap20              BGWO on an ANOVA-F pool (top 200 by F, repair in F order) instead of the Hellinger pool
  F20                      ANOVA-F top-20 filter (comparator for bgwoF_cap20)
  H20                      Hellinger top-20 filter of the same pool order (comparator for the Hellinger-pool runs)
  bgwo_cap20_x4, bgwo_nocap_x4   BGWO with four times the budget (50 wolves x 160 iterations), repeats 0-1 only
Search-seed stability: on the first fold of repeat 0, bgwo_cap20 is re-run with five further seeds.
Pool order: Hellinger and F scores sorted with a stable sort (ties in column order).
Subsets are scored with 5-NN, logistic regression, linear SVM and random forest (as in run_base.py).
Usage: python run_wrap2.py <data_dir> <out_dir> [n_jobs]   (env NREP, default 5; NREP_X4, default 2)
"""
import sys, os, glob, time, pickle, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.optimizer import bgwo, _init, _binarize
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from run_wrapper import MeasureFitness, CEILING
from run_cap import Capped, repair
from run_xclf import score

warnings.filterwarnings("ignore")
NREP = int(os.environ.get("NREP", "5"))
NREP_X4 = int(os.environ.get("NREP_X4", "2"))
POOL = 200
CLFS = ["knn", "lr", "svm", "rf"]
MAIN = ["bgwo_cap20", "bgwo_nocap", "ga_cap20", "ga_nocap", "bgwoF_cap20", "F20", "H20"]
X4 = ["bgwo_cap20_x4", "bgwo_nocap_x4"]


def bga(fit, K, rng, pop=50, gens=40, pc=0.9):
    """Binary GA, elitist, same evaluation count as bgwo (pop * gens)."""
    M = _binarize(_init(pop, K, rng), rng)
    F = np.array([fit(m) for m in M])
    best_i = int(np.argmin(F)); best_m, best_f = M[best_i].copy(), F[best_i]
    for _ in range(gens - 1):
        a, b = rng.integers(0, pop, (2, pop))
        parents = M[np.where(F[a] <= F[b], a, b)]
        mates = parents[rng.permutation(pop)]
        cross = rng.random(pop) < pc
        swap = (rng.random((pop, K)) < 0.5) & cross[:, None]
        child = np.where(swap, mates, parents)
        child ^= rng.random((pop, K)) < 1.0 / K
        child[0] = best_m                                  # elitism (counted as an evaluation, as in bgwo)
        M = child
        F = np.array([fit(m) for m in M])
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_m = F[i], M[i].copy()
    return best_m, best_f, {}


def run_one(path, out_dir, n_splits=5, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {(m, c): [np.zeros(len(y)) for _ in range(NREP)] for m in MAIN for c in CLFS}
    prob.update({(m, c): [np.zeros(len(y)) for _ in range(NREP_X4)] for m in X4 for c in CLFS})
    subsets = {m: [] for m in MAIN + X4}
    times = {m: [] for m in MAIN + X4}
    seed_subsets = []
    for r in range(NREP):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
            oh = np.argsort(-ind.hellinger(Xtr, ytr), kind="stable")
            of = np.argsort(-ind.anova_f(Xtr, ytr), kind="stable")
            ph, pf = oh[: min(POOL, X.shape[1])], of[: min(POOL, X.shape[1])]
            fh = MeasureFitness(Xtr[:, ph], ytr, "bbrier", 0.99)
            ff = MeasureFitness(Xtr[:, pf], ytr, "bbrier", 0.99)
            K = len(ph)
            cols = {}
            runs = [("bgwo_cap20", bgwo, Capped(fh, 20), ph, 20, {}), ("bgwo_nocap", bgwo, fh, ph, None, {}),
                    ("ga_cap20", bga, Capped(fh, 20), ph, 20, {}), ("ga_nocap", bga, fh, ph, None, {}),
                    ("bgwoF_cap20", bgwo, Capped(ff, 20), pf, 20, {})]
            if r < NREP_X4:
                runs += [("bgwo_cap20_x4", bgwo, Capped(fh, 20), ph, 20, dict(iters=160)),
                         ("bgwo_nocap_x4", bgwo, fh, ph, None, dict(iters=160))]
            for m, alg, fit, pool, cap, kw in runs:
                s = time.perf_counter()
                mask, _, _ = alg(fit, K, np.random.default_rng([seed, r, fi, 999]), **kw)
                times[m].append(time.perf_counter() - s)
                cols[m] = pool[repair(mask, cap)] if cap else pool[mask]
            cols["H20"], cols["F20"] = oh[:20], of[:20]
            if r == 0 and fi == 0:
                for extra in range(5):
                    mask, _, _ = bgwo(Capped(fh, 20), K, np.random.default_rng([seed, r, fi, 1000 + extra]))
                    seed_subsets.append(ph[repair(mask, 20)].astype(np.int32))
                seed_subsets.append(cols["bgwo_cap20"].astype(np.int32))
            for m, c_ in cols.items():
                subsets[m].append(np.asarray(c_, dtype=np.int32))
                if len(c_) == 0:
                    continue
                prob[(m, "knn")][r][te] = knn_prob(sqdist(Xte[:, c_], Xtr[:, c_]), ytr)
                for c in ("lr", "svm", "rf"):
                    prob[(m, c)][r][te] = np.clip(score(c, Xtr[:, c_], ytr, Xte[:, c_], seed), 0.02, 0.98)
    res, res_rep = {}, {}
    for key, ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[key] = {q: float(np.mean([x[q] for x in rows])) for q in ("gmean", "mcc", "aucpr", "auc")}
        res_rep[key] = [{q: float(x[q]) for q in ("gmean", "mcc", "aucpr", "auc")} for x in rows]
    pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), nrep=NREP, nrep_x4=NREP_X4, res=res, res_rep=res_rep,
                     subsets=subsets, times=times, seed_subsets=seed_subsets), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: os.path.getsize(p))
    Parallel(n_jobs=nj)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
