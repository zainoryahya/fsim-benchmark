"""Class-balanced kernel ridge (regularised least-squares) classifier with a closed-form LOO.

Linear kernel plus a large constant (C0) so the intercept is almost unpenalised. Sample weights
make the two classes count equally. With s = sqrt(w), K~ = (s s^T) * K, A = K~ + lam I:
    alpha~ = A^-1 (s*y)           test score f(x) = sum_i s_i alpha~_i k(x, x_i)
    LOO prediction for i = y_i - [A^-1 (s*y)]_i / (A^-1)_ii / s_i      (exact for this smoother)
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

C0 = 1e4
CLIP = (0.02, 0.98)


def _weights(y):
    n = len(y)
    m = y.sum()
    return np.where(y == 1, n / (2.0 * m), n / (2.0 * (n - m)))


def _A(Z, y, lam):
    s = np.sqrt(_weights(y))
    K = Z @ Z.T + C0
    A = K * np.outer(s, s)
    A[np.diag_indices_from(A)] += lam
    return A, s


def loo_prob(Z, y, lam):
    A, s = _A(Z, y, lam)
    cf = cho_factor(A, lower=True, check_finite=False)
    Ainv = cho_solve(cf, np.eye(len(y)), check_finite=False)
    ya = Ainv @ (s * y)
    p = y - ya / np.diag(Ainv) / s
    return np.clip(p, *CLIP)


def fit_predict(Ztr, y, Zte, lam):
    A, s = _A(Ztr, y, lam)
    a = np.linalg.solve(A, s * y)
    return np.clip((Zte @ Ztr.T + C0) @ (s * a), *CLIP)
