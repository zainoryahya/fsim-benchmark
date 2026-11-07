"""Phase 5c: where should resampling sit, and how large is the leakage bias? (redo, v2 data)

Differences from the prototype (run_leakage.py): 34 v2 datasets (4 ceiling sets excluded), two filters
(ANOVA F and Hellinger, tie bug fixed), two classifiers (5-NN, logistic regression), and chance-referenced
metrics: raw MCC (chance 0) and AUC-ROC (chance 0.5) as well as G-mean. The prototype's G-mean has a
chance level that rises with the share of minority predictions, so a SMOTE pipeline looked "leaky" on
permuted labels when it was not.

Pipelines (top-20 filter, 5-fold x 2 repeats, metrics on pooled out-of-fold predictions):
  FS only            filter inside the training fold
  SMOTE->FS          SMOTE on the training fold, then filter           (correct, label-aware inside)
  FS->SMOTE          filter, then SMOTE on the selected columns         (correct)
  leak SMOTE-first   SMOTE on all data before CV                        (leaky)
  leak FS-first      filter on all data before CV                       (leaky)
Real labels (perm = -1) and label-permuted copies (perm >= 0) share seeds.
"""
import numpy as np, warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from fsimb import indicators as ind
from fsimb.innerest import sqdist, knn_prob

warnings.filterwarnings("ignore")
K = 20
PIPES = ["FS only", "SMOTE->FS", "FS->SMOTE", "leak SMOTE-first", "leak FS-first"]
FS = {"F": ind.anova_f, "H": ind.hellinger}
CLF = ["knn", "lr"]


def smote(X, y, seed):
    m = int(y.sum())
    return SMOTE(k_neighbors=min(5, m - 1), random_state=seed).fit_resample(X, y)


def topk(fs, X, y):
    return np.argsort(-FS[fs](X, y))[:K]


def predict(clf, Xtr, ytr, Xte):
    if clf == "knn":
        return knn_prob(sqdist(Xte, Xtr), ytr)
    return LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def metrics(y, p):
    return dict(mcc=float(matthews_corrcoef(y, p > 0.5)), auc=float(roc_auc_score(y, p)),
                gmean=float(np.sqrt(((p > 0.5)[y == 1].mean()) * ((p <= 0.5)[y == 0].mean()))))


def evaluate(X, y, seed=7, repeats=2):
    out = {}
    for fs in FS:
        for pipe in PIPES:
            Xe, ye = X, y
            if pipe == "leak SMOTE-first":
                Xe, ye = smote(X, y, seed)
            glob = topk(fs, Xe, ye) if pipe == "leak FS-first" else None
            acc = {c: [] for c in CLF}
            for r in range(repeats):
                prob = {c: np.zeros(len(ye)) for c in CLF}
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(Xe, ye):
                    mu, sd = Xe[tr].mean(0), Xe[tr].std(0)
                    sd[sd == 0] = 1.0
                    Xtr, Xte, ytr = (Xe[tr] - mu) / sd, (Xe[te] - mu) / sd, ye[tr]
                    if pipe == "SMOTE->FS":
                        Xtr, ytr = smote(Xtr, ytr, seed + r)
                        cols = topk(fs, Xtr, ytr)
                        Xa, ya = Xtr[:, cols], ytr
                    elif pipe == "FS->SMOTE":
                        cols = topk(fs, Xtr, ytr)
                        Xa, ya = smote(Xtr[:, cols], ytr, seed + r)
                    else:
                        cols = glob if glob is not None else topk(fs, Xtr, ytr)
                        Xa, ya = Xtr[:, cols], ytr
                    for c in CLF:
                        prob[c][te] = predict(c, Xa, ya, Xte[:, cols])
                for c in CLF:
                    acc[c].append(metrics(ye, prob[c]))
            for c in CLF:
                out[(fs, pipe, c)] = {k: float(np.mean([m[k] for m in acc[c]])) for k in ("mcc", "auc", "gmean")}
    return out


def job(path, perm):
    d = np.load(path)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    if perm >= 0:
        y = np.random.default_rng(100 + perm).permutation(y)
    return path.split("/")[-1][:-4], perm, evaluate(X, y)
