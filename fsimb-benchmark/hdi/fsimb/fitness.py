"""Component 1: imbalance-aware wrapper fitness.

f(S) = alpha * (1 - Q(S)) + (1 - alpha) * |S| / K

Q is a classification quality measure estimated by leave-one-out k-NN on
the (outer) training fold only. Q = accuracy is the usual choice in the
literature; Q = G-mean or MCC is the imbalance-aware choice.
"""
import numpy as np


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def gmean(y, pred):
    tp, tn, fp, fn = confusion(y, pred)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return float(np.sqrt(tpr * tnr))


def mcc(y, pred):
    tp, tn, fp, fn = confusion(y, pred)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else 0.0


def accuracy(y, pred):
    return float((y == pred).mean())


QUALITY = {"acc": accuracy, "gmean": gmean, "mcc": lambda y, p: (mcc(y, p) + 1) / 2}


class LOOKNNFitness:
    """Fast leave-one-out k-NN fitness on a fixed training matrix.

    Squared distances are built from per-feature contributions, so a
    subset's distance matrix is one matrix product: D(S) = sum_{f in S} D_f.
    """

    def __init__(self, X, y, metric="gmean", k=3, alpha=0.99):
        self.y = y
        self.k = k
        self.alpha = alpha
        self.q = QUALITY[metric]
        self.K = X.shape[1]
        # n x n x K would be large; use Gram trick per call instead
        self.X = X
        self.sq = X ** 2
        self.evals = 0

    def predict_loo(self, mask):
        Xs = self.X[:, mask]
        s = self.sq[:, mask].sum(1)
        D = s[:, None] + s[None, :] - 2.0 * Xs @ Xs.T
        np.fill_diagonal(D, np.inf)
        nn = np.argpartition(D, self.k, axis=1)[:, : self.k]
        return (self.y[nn].mean(1) > 0.5).astype(int)

    def __call__(self, mask):
        self.evals += 1
        m = int(mask.sum())
        if m == 0:
            return 1.0
        q = self.q(self.y, self.predict_loo(mask))
        return self.alpha * (1.0 - q) + (1.0 - self.alpha) * m / self.K
