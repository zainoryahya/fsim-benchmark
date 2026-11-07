"""sssHD (Fu et al. 2020, re-implemented) on the same folds as run_base.py; compared with the H ranking of run_base.

Same outer folds as the wrapper experiments (stratified 5-fold, seeds 2027 + r, r = 0..NREP-1; standardisation
with training statistics). Each method ranks the features of the training fold; the top k (10, 20, 30, 40) are
scored with 5-NN (inverse distance, as in the wrapper experiments), logistic regression, linear SVM and random
forest (as in run_xclf). Records the ranking time of every method in every fold.
Fold rankings are checkpointed in <out_dir>/_cache (keyed by dataset, repeat and fold only: clear the cache if
K_SUB or fsimb/ssshd.py is changed).
Usage: python run_ssshd.py <data_dir> <out_dir> [n_jobs]   (env NREP, default 2; R0, default 0; K_SUB, default 50)
"""
import sys, os, glob, time, pickle, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.ssshd import ssshd_rank
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from run_xclf import score
from run_wrapper import CEILING

warnings.filterwarnings("ignore")
NREP = int(os.environ.get("NREP", "2"))
R0 = int(os.environ.get("R0", "0"))
KS = (10, 20, 30, 40)
METHODS = ["sssHD", "H"]
K_SUB = int(os.environ.get("K_SUB", "50"))
CLFS = ["knn", "lr", "svm", "rf"]


def rankings(Xtr, ytr, seed):
    s = time.perf_counter()
    order, info = ssshd_rank(Xtr, ytr, np.random.default_rng(seed), K=K_SUB, return_info=True)
    t = time.perf_counter() - s
    info = dict(n_selected=info["n_selected"], sizes=info["sizes"], hds=info["hds"], ilam=info["ilam"])
    s = time.perf_counter(); h = np.argsort(-ind.hellinger(Xtr, ytr), kind="stable"); th = time.perf_counter() - s
    return {"sssHD": order, "H": h}, {"sssHD": t, "H": th}, info


def run_one(path, out_dir, n_splits=5, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {(m, k, c): [np.zeros(len(y)) for _ in range(NREP)] for m in METHODS for k in KS for c in CLFS}
    subsets = {(m, k): [] for m in METHODS for k in KS}
    times = {m: [] for m in METHODS}
    info = []
    for r in range(NREP):
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r + R0).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
            ck = f"{out_dir}/_cache/{name}_{r + R0}_{fi}.pkl"      # fold checkpoint (container restarts)
            if os.path.exists(ck):
                rk, tm, inf = pickle.load(open(ck, "rb"))
            else:
                rk, tm, inf = rankings(Xtr, ytr, seed + 100 * (r + R0) + fi)
                pickle.dump((rk, tm, inf), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            info.append(inf)
            for m in METHODS:
                times[m].append(tm[m])
                for k in KS:
                    cols = rk[m][:k]
                    subsets[(m, k)].append(cols.astype(np.int32))
                    prob[(m, k, "knn")][r][te] = knn_prob(sqdist(Xte[:, cols], Xtr[:, cols]), ytr)
                    for c in ("lr", "svm", "rf"):
                        prob[(m, k, c)][r][te] = np.clip(score(c, Xtr[:, cols], ytr, Xte[:, cols], seed), 0.02, 0.98)
    res = {}
    for key, ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[key] = {q: float(np.mean([x[q] for x in rows])) for q in ("gmean", "mcc", "aucpr", "auc")}
    pickle.dump(dict(r0=R0, k_sub=K_SUB, name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), nrep=NREP, res=res,
                     subsets=subsets, times=times, info=info), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(f"{out_dir}/_cache", exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: os.path.getsize(p))
    Parallel(n_jobs=nj)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
