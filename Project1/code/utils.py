"""
Shared functions for FYS-STK3155/4155 Project 1.

Contents
--------
Data:        runge, runge_data, design_matrix
Scaling:     Scaler (standardise X, centre y, training statistics only)
Metrics:     MSE, R2
Regression:  ols_fit (pseudoinverse), ridge_fit (closed form)
Output:      media_path  -> results/media/<part>/   (figures)
             output_path -> results/output/<part>/  (tables, summaries, parameters)
             save_table, save_text

Conventions
-----------
Ridge cost:  C(theta) = (1/n) ||y - X theta||^2 + lam ||theta||^2
             => theta = (X^T X + n*lam*I)^{-1} X^T y
             Equivalent to sklearn Ridge(alpha = n*lam, fit_intercept=False).
Intercept:   not included in X; handled by centring y and X, so it is never penalised.
"""
from pathlib import Path

import numpy as np

# Everything the code produces goes under results/ next to this file.
# Change here if results/ lives elsewhere in the repository.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def runge(x):
    """Runge's function f(x) = 1/(1+25x^2)."""
    return 1.0 / (1.0 + 25.0 * x**2)


def runge_data(n=100, noise=0.1, xmin=-1.0, xmax=1.0, seed=2026):
    """Uniformly sampled x on [xmin, xmax] and noisy Runge data y = f(x) + N(0, noise^2)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, n)
    y = runge(x) + noise * rng.standard_normal(n)
    return x, y


def design_matrix(x, degree):
    """Vandermonde matrix without the constant column: columns x^1, ..., x^degree."""
    return x[:, None] ** np.arange(1, degree + 1)


# ----------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------
class Scaler:
    """Standardise the columns of X and centre y, using training statistics only.

    Usage:
        sc = Scaler().fit(X_train, y_train)
        X_train_s, X_test_s = sc.transform_X(X_train), sc.transform_X(X_test)
        theta = fit(X_train_s, y_train - sc.y_mean)
        y_pred = X_test_s @ theta + sc.y_mean
    """

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.y_mean = y.mean()
        return self

    def transform_X(self, X):
        return (X - self.mu) / self.sd


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def MSE(y_data, y_model):
    return np.mean((y_data - y_model) ** 2)


def R2(y_data, y_model):
    return 1.0 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


# ----------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------
def ols_fit(X, y):
    """OLS parameters via the pseudoinverse (SVD), more stable than the normal equations."""
    return np.linalg.pinv(X) @ y


def ridge_fit(X, y, lam):
    """Closed-form Ridge (see module docstring for the convention). lam = 0 -> OLS via pinv."""
    if lam == 0:
        return ols_fit(X, y)
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)

# ----------------------------------------------------------------------------
# Resampling: bootstrap and bias-variance decomposition
# ----------------------------------------------------------------------------
def bootstrap_predictions(x_tr, y_tr, x_te, degree, n_boot, rng, fit=ols_fit):
    """Predictions on a FIXED test set from models fitted on bootstrap resamples.
 
    For each of the n_boot rounds the training set is resampled with replacement,
    the Scaler is fitted on the resample (no information from the original training
    set as a whole or from the test set), and the model is fitted and evaluated on x_te.
 
    fit: function fit(X, y) -> theta, e.g. ols_fit or lambda X, y: ridge_fit(X, y, lam).
    Returns an array of shape (len(x_te), n_boot).
    """
    n_tr = len(x_tr)
    X_te = design_matrix(x_te, degree)
    preds = np.empty((len(x_te), n_boot))
    for b in range(n_boot):
        idx = rng.integers(0, n_tr, n_tr)                  # draw with replacement
        X_b, y_b = design_matrix(x_tr[idx], degree), y_tr[idx]
        sc = Scaler().fit(X_b, y_b)                        # scaling learnt from the resample
        theta = fit(sc.transform_X(X_b), y_b - sc.y_mean)
        preds[:, b] = sc.transform_X(X_te) @ theta + sc.y_mean
    return preds
 
 
def bias_variance(y_ref, preds):
    """Sample bias-variance decomposition over bootstrap predictions.
 
    y_ref: reference values at the test points (the noisy y_test, or the true f(x_test)).
    preds: array (n_test, n_boot) from bootstrap_predictions.
 
    error = mean_i mean_b (y_i - yhat_ib)^2
    bias2 = mean_i (y_i - mean_b yhat_ib)^2
    var   = mean_i var_b (yhat_ib)
    With these definitions error = bias2 + var holds exactly.
    """
    mean_pred = preds.mean(axis=1)
    error = np.mean((y_ref[:, None] - preds) ** 2)
    bias2 = np.mean((y_ref - mean_pred) ** 2)
    var = np.mean(preds.var(axis=1))
    return error, bias2, var



# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _results_path(kind, part, filename):
    folder = RESULTS_DIR / kind / part
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def media_path(part, filename):
    """Path results/media/<part>/<filename> for figures (folder created if needed)."""
    return _results_path("media", part, filename)


def output_path(part, filename):
    """Path results/output/<part>/<filename> for tables and text (folder created if needed)."""
    return _results_path("output", part, filename)


def save_table(part, filename, columns):
    """Save a CSV file in results/output/<part>/.

    columns: dict {column name: 1D array}, all of equal length.
    """
    names = list(columns)
    data = np.column_stack([np.asarray(columns[k], dtype=float) for k in names])
    np.savetxt(output_path(part, filename), data, delimiter=",",
               header=",".join(names), comments="", fmt="%.10e")


def save_text(part, filename, lines):
    """Save a list of lines as a text file in results/output/<part>/."""
    output_path(part, filename).write_text("\n".join(lines) + "\n")