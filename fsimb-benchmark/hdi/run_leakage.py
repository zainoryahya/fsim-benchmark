"""Component 3: where should resampling sit, and how big is the leakage bias?

FS = top-20 ANOVA F filter, classifier = 3-NN, 5-fold x 4 repeats,
G-mean on pooled out-of-fold predictions. Run on real data and on
label-permuted (null) copies, where any G-mean well above chance is
pure leakage.
"""
import pickle, numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from fsimb import indicators as ind
from fsimb.fitness import gmean
from fsimb.pipeline import knn_predict

K = 20
PIPES = ["Betul: FS sahaja", "Betul: SMOTE → FS", "Betul: FS → SMOTE",
         "Bocor: SMOTE sebelum CV", "Bocor: FS sebelum CV"]


def smote(X, y, seed):
    m = int(y.sum())
    return SMOTE(k_neighbors=min(5, m - 1), random_state=seed).fit_resample(X, y)


def topk(X, y):
    return np.argsort(-ind.anova_f(X, y))[:K]


def evaluate(X, y, pipe, seed, repeats=4):
    if pipe == "Bocor: SMOTE sebelum CV":
        X, y = smote(X, y, seed)
    glob = topk(X, y) if pipe == "Bocor: FS sebelum CV" else None
    g = []
    for r in range(repeats):
        pred = np.zeros(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            Xtr, Xte, ytr = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr]
            if pipe == "Betul: SMOTE → FS":
                Xtr, ytr = smote(Xtr, ytr, seed + r)
                cols = topk(Xtr, ytr)
            elif pipe == "Betul: FS → SMOTE":
                cols = topk(Xtr, ytr)
                Xs, ytr = smote(Xtr[:, cols], ytr, seed + r)
                Xtr = np.zeros((len(ytr), X.shape[1]))
                Xtr[:, cols] = Xs
            elif glob is not None:
                cols = glob
            else:
                cols = topk(Xtr, ytr)
            pred[te] = knn_predict(Xtr[:, cols], ytr, Xte[:, cols])[0]
        g.append(gmean(y, pred))
    return float(np.mean(g))


def job(name, perm):
    d = np.load(f"data/{name}.npz")
    X, y = d["X"], d["y"]
    if perm >= 0:
        y = np.random.default_rng(100 + perm).permutation(y)
    return name, perm, {p: evaluate(X, y, p, seed=7) for p in PIPES}


if __name__ == "__main__":
    real = ["gordon", "tian", "shipp", "gravier", "chowdary_ir5", "singh_ir5"]
    jobs = [(n, -1) for n in real] + [(n, k) for n in ["gordon", "tian", "gravier"] for k in range(5)]
    out = Parallel(n_jobs=2)(delayed(job)(n, k) for n, k in jobs)
    with open("results_leakage.pkl", "wb") as f:
        pickle.dump({"pipes": PIPES, "rows": out}, f)
    for n, k, r in out:
        print(n, k, {p: round(v, 3) for p, v in r.items()})
