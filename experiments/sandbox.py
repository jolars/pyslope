import matplotlib.pyplot as plt
from benchopt.datasets import make_correlated_data

from slope.solvers import pgd_slope
from slope.utils import lambda_sequence

rho = 0.9

X, y, _ = make_correlated_data(n_samples=100, n_features=10, rho=rho, random_state=0)

n, p = X.shape
q = 0.1
fit_intercept = True
reg = 0.1

lambdas = lambda_sequence(X, y, fit_intercept, reg, q)

max_it = 10000
verbose = True
gap_tol = 1e-4

out = pgd_slope(
    X,
    y,
    lambdas,
    fit_intercept=fit_intercept,
    max_it=max_it,
    verbose=verbose,
    gap_tol=gap_tol,
)

plt.close("all")

plt.title(f"n: {n}, p: {p}, rho: {rho}, reg: {reg}, q: {q}")

plt.ylabel("duality gap")
plt.xlabel("Time (s)")

plt.semilogy(out["gaps"], label="cd")

out["gaps"]
plt.show(block=False)
