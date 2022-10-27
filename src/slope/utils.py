from timeit import default_timer as timer

import numpy as np
import scipy.sparse as sparse
from numba import njit, float64
from numpy.linalg import norm
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


def dual_norm_slope(X, theta, alphas):
    """Dual slope norm of X.T @ theta"""
    Xtheta = np.sort(np.abs(X.T @ theta))[::-1]
    taus = 1 / np.cumsum(alphas)
    return np.max(np.cumsum(Xtheta) * taus)


def lambda_sequence(X, y, fit_intercept, reg=0.1, q=0.1):
    """Generates the BH-type lambda sequence"""
    n, p = X.shape

    randnorm = stats.norm(loc=0, scale=1)
    lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
    lambda_max = dual_norm_slope(X, (y - np.mean(y) * fit_intercept) / n, lambdas)

    return lambda_max * lambdas * reg


@njit
def sl1_norm(beta, lambdas):
    return np.sum(lambdas * np.sort(np.abs(beta))[::-1])


@njit
def primal(beta, X, y, lambdas):
    return (0.5 / len(y)) * norm(y - X @ beta) ** 2 + sl1_norm(beta, lambdas)


@njit(float64[:](float64[:], float64[:]))
def prox_slope(beta, lambdas):
    """Compute the sorted L1 proximal operator.

    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    beta_sign = np.sign(beta)
    beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1.0)

        k = k + 1

    for j in range(k):
        d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    beta *= beta_sign

    return beta


def preprocess(X):
    # remove zero variance predictors
    X = VarianceThreshold().fit_transform(X)

    # standardize
    if sparse.issparse(X):
        X = MaxAbsScaler().fit_transform(X)
        return X.tocsc()
    else:
        X = StandardScaler().fit_transform(X)
        return X
