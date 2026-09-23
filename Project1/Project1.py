import matplotlib.pyplot as plt
import numpy as np

def runge_data(n=100, noise=0.1, xmin=(-1), xmax=1, seed=2026):
    """Runge function 1/(1+25x^2) on [-1,1], standardised polynomial features, centred y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, n)
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n)
    X = np.column_stack([x**k for k in range(1, degree + 1)])
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_norm, y - y.mean()


def closed_form(X, y, lam=0.0):
    """Lukket OLS/Ridge-løsning."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)

n=100
sigma=0.1
max_degrees=15
x, y=runge_data(n=n, noise=sigma, xmin=-1, xmax=1)
y_n=y-y.mean()
for degree in range(0, max_degrees+1):
    X = np.column_stack([x**k for k in range(1, degree + 1)])
    X_n = (X - X.mean(axis=0)) / X.std(axis=0)
    