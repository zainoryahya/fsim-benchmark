"""Phase 2: the 12 candidate fitness measures.

Every measure maps (y, prob) -> a score in [0, 1], higher = better, so that
the wrapper fitness is always  f(S) = alpha * (1 - Q) + (1 - alpha) * |S| / K.
y is 0/1 with 1 = minority; prob = P(minority) in (0, 1); the class label is
prob > 0.5.

Label based (use the thresholded prediction only):
  acc     accuracy
  ba      balanced accuracy = (TPR + TNR) / 2
          NB: Youden's J = TPR + TNR - 1 = 2 * BA - 1. After the [0, 1]
          normalisation J and BA are the same number, so J is not listed
          separately.
  gmean   sqrt(TPR * TNR)
  mcc     (MCC + 1) / 2
  f1      F1 of the minority class
  f2      F2 of the minority class (recall weighted 4x precision)
  kappa   (Cohen's kappa + 1) / 2
  fm      Fowlkes-Mallows = sqrt(precision * recall), minority class

Probability based (use the whole score):
  auc     ROC AUC (average ranks for ties)
  aucpr   average precision (area under the precision-recall curve)
  bbrier  1 - class-balanced Brier score
  bll     exp(-class-balanced log loss): geometric-mean probability given to
          the true class, each class counting equally
"""
import numpy as np
from scipy.stats import rankdata

MEASURES = ["acc", "ba", "gmean", "mcc", "f1", "f2", "kappa", "fm",
            "auc", "aucpr", "bbrier", "bll"]
LABEL_BASED = MEASURES[:8]
PROB_BASED = MEASURES[8:]
GROUPS = {"label": LABEL_BASED, "prob": PROB_BASED}


def _conf(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def _div(a, b):
    return a / b if b > 0 else 0.0


def from_confusion(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    tpr, tnr = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "acc": po,
        "ba": (tpr + tnr) / 2,
        "gmean": float(np.sqrt(tpr * tnr)),
        "mcc": (mcc + 1) / 2,
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "f2": _div(5 * tp, 5 * tp + 4 * fn + fp),
        "kappa": (min(max(kappa, -1.0), 1.0) + 1) / 2,
        "fm": float(np.sqrt(prec * tpr)),
    }


def auc_roc(y, p):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, p):
    n1 = int(y.sum())
    if n1 == 0:
        return 0.0
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    ps = p[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(y) + 1)
    # ties in score: evaluate precision at the end of each tie block
    last = np.r_[ps[1:] != ps[:-1], True]
    rec = tp[last] / n1
    pr = prec[last]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * pr))


def all_measures(y, prob):
    """dict of the 12 measures for one set of out-of-sample predictions."""
    y = np.asarray(y)
    pred = (prob > 0.5).astype(int)
    out = from_confusion(*_conf(y, pred))
    pos, neg = prob[y == 1], prob[y == 0]
    out["auc"] = auc_roc(y, prob)
    out["aucpr"] = average_precision(y, prob)
    out["bbrier"] = 1.0 - 0.5 * (np.mean((1 - pos) ** 2) + np.mean(neg ** 2))
    out["bll"] = float(np.exp(-0.5 * (np.mean(-np.log(pos)) + np.mean(-np.log(1 - neg)))))
    return out


def measure_vector(y, prob):
    d = all_measures(y, prob)
    return np.array([d[m] for m in MEASURES])
