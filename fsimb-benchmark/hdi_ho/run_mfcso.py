"""MF-CSO (Li et al. 2023, re-implemented) inside our nested protocol, plus the Hellinger filter at the same size.

Same outer folds as the other experiments (stratified 5-fold, seeds 2027 + r, r = R0..R0+NREP-1; standardisation
with training statistics). MF-CSO runs with its published settings on the training fold. Its subset, and the top
|S| Hellinger features of the same fold, are scored with 5-NN, 1-NN, logistic regression, linear SVM and RF.
Usage: python run_mfcso.py <data_dir> <out_dir> [n_jobs]   (env R0 default 0, NREP default 2)
"""
import sys, os, glob, time, pickle, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.mfcso import mfcso
from fsimb.innerest import sqdist, knn_prob
from fsimb.measures import all_measures
from run_xclf import score
from run_wrapper import CEILING

warnings.filterwarnings("ignore")
R0 = int(os.environ.get("R0", "0")); NREP = int(os.environ.get("NREP", "2"))
CLFS = ["knn", "1nn", "lr", "svm", "rf"]
METHODS = ["mfcso", "H_same_k"]


def one_nn_prob(Xte, Xtr, ytr):
    D = sqdist(Xte, Xtr)
    return np.clip(ytr[np.argmin(D, axis=1)].astype(float), 0.02, 0.98)


def run_one(path, out_dir, n_splits=5, seed=2027):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    t0 = time.time()
    prob = {(m, c): [np.zeros(len(y)) for _ in range(NREP)] for m in METHODS for c in CLFS}
    subsets = {m: [] for m in METHODS}
    times, infos = [], []
    for r in range(NREP):
        rr = r + R0
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + rr).split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
            Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
            s = time.perf_counter()
            mask, info = mfcso(Xtr, ytr, np.random.default_rng([seed, rr, fi, 31]), return_info=True)
            times.append(time.perf_counter() - s); infos.append(info)
            cols = {"mfcso": np.flatnonzero(mask)}
            cols["H_same_k"] = np.argsort(-ind.hellinger(Xtr, ytr))[: len(cols["mfcso"])]
            for m in METHODS:
                c_ = cols[m]
                subsets[m].append(c_.astype(np.int32))
                prob[(m, "knn")][r][te] = knn_prob(sqdist(Xte[:, c_], Xtr[:, c_]), ytr)
                prob[(m, "1nn")][r][te] = one_nn_prob(Xte[:, c_], Xtr[:, c_], ytr)
                for c in ("lr", "svm", "rf"):
                    prob[(m, c)][r][te] = np.clip(score(c, Xtr[:, c_], ytr, Xte[:, c_], seed), 0.02, 0.98)
    res = {}
    for key, ps in prob.items():
        rows = [all_measures(y, p) for p in ps]
        res[key] = {q: float(np.mean([x[q] for x in rows])) for q in ("gmean", "mcc", "aucpr", "auc")}
    for m in METHODS:
        res[(m, "size")] = float(np.mean([len(s_) for s_ in subsets[m]]))
    pickle.dump(dict(name=name, n=len(y), p=X.shape[1], minority=int(y.sum()), r0=R0, nrep=NREP, res=res,
                     subsets=subsets, times=times, infos=infos), open(f"{out_dir}/{name}.pkl", "wb"))
    print(f"{name:22s} {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in sorted(glob.glob(f"{data_dir}/*.npz")) if os.path.basename(p)[:-4] not in CEILING
             and not os.path.exists(f"{out_dir}/{os.path.basename(p)[:-4]}.pkl")]
    paths.sort(key=lambda p: os.path.getsize(p))       # small first, so partial runs cover many datasets
    Parallel(n_jobs=nj)(delayed(run_one)(p, out_dir) for p in paths)
    print("done")
