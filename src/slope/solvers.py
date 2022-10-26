import numpy as np
from numpy.linalg import norm
from scipy import sparse

from slope.utils import dual_norm_slope, prox_slope, sl1_norm


def pgd_slope(
    X,
    y,
    lambdas,
    fit_intercept=True,
    gap_tol=1e-6,
    max_it=10_000,
    verbose=False,
):
    n, p = X.shape

    residual = y.copy()
    beta = np.zeros(p)
    intercept = 0.0

    primals, duals, gaps = [], [], []

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n
    else:
        L = norm(X, ord=2) ** 2 / n

    for it in range(max_it):
        beta = prox_slope(beta + (X.T @ residual) / (L * n), lambdas / L)

        residual = y - X @ beta

        if fit_intercept:
            intercept = np.mean(residual)

        residual -= intercept

        theta = residual / n
        theta /= max(1, dual_norm_slope(X, theta, lambdas))

        primal = 0.5 * norm(residual) ** 2 / n + sl1_norm(beta, lambdas)
        dual = 0.5 * (norm(y) ** 2 - norm(y - theta * n) ** 2) / n
        gap = primal - dual

        primals.append(primal)
        duals.append(primal)
        gaps.append(gap)

        if verbose:
            print(f"epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")

        if gap < gap_tol:
            break

    return dict(beta=beta, intercept=intercept, primals=primals, gaps=gaps)
