"""
Shared functions for FYS-STK3155/4155 Project 1.

Contents
--------
Data:        runge, runge_data, design_matrix
Scaling:     Scaler (standardise X, centre y, training statistics only)
Metrics:     MSE, R2
Regression:  ols_fit (pseudoinverse), ridge_fit (closed form)
Resampling:  bootstrap_predictions, bias_variance
Gradients:   ridge_cost, ridge_grad (analytical), ridge_hessian, autodiff_backend (JAX/Autograd)
Optimisers:  make_update (update rules: GD, momentum, AdaGrad, RMSprop, Adam),
             optimise (full batch; optional proximal step for ISTA),
             sgd (mini-batches, epochs, learning-rate schedule),
             gradient_descent (plain GD, fixed learning rate; wrapper used in part e)
Output:     media_path  -> results/media/<part>/   (figures)
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
    """Runge's function f(x) = 1/(1+25x^2).
    
    LLM was used to generate code."""
    return 1.0 / (1.0 + 25.0 * x**2)


def runge_data(n=100, noise=0.1, xmin=-1.0, xmax=1.0, seed=2026):
    """Uniformly sampled x on [xmin, xmax] and noisy Runge data y = f(x) + N(0, noise^2).
    
    LLM was used to generate code."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, n)
    y = runge(x) + noise * rng.standard_normal(n)
    return x, y


def design_matrix(x, degree):
    """Vandermonde matrix without the constant column: columns x^1, ..., x^degree.
    
    LLM was used to generate code."""
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

    LLM was used to generate code.
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
    """OLS parameters via the pseudoinverse (SVD), more stable than the normal equations.
    
    LLM was used to generate code."""
    return np.linalg.pinv(X) @ y


def ridge_fit(X, y, lam):
    """Closed-form Ridge (see module docstring for the convention). lam = 0 -> OLS via pinv.
    
    LLM was used to generate code."""
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

    LLM was used to generate code.
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

    LLM was used to generate code.
    """
    mean_pred = preds.mean(axis=1)
    error = np.mean((y_ref[:, None] - preds) ** 2)
    bias2 = np.mean((y_ref - mean_pred) ** 2)
    var = np.mean(preds.var(axis=1))
    return error, bias2, var


# ----------------------------------------------------------------------------
# Cost functions, gradients and Hessian (OLS = Ridge with lam = 0)
# ----------------------------------------------------------------------------
def ridge_cost(theta, X, y, lam=0.0):
    """C(theta) = (1/n)||y - X theta||^2 + lam ||theta||^2.
    
    LLM was used to generate code."""
    return np.mean((y - X @ theta) ** 2) + lam * np.sum(theta ** 2)


def ridge_grad(theta, X, y, lam=0.0):
    """Analytical gradient: (2/n) X^T (X theta - y) + 2 lam theta.
    
    LLM was used to generate code."""
    n = X.shape[0]
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta


def ridge_hessian(X, lam=0.0):
    """Hessian (constant): (2/n) X^T X + 2 lam I.
    
    LLM was used to generate code."""
    n, p = X.shape
    return (2.0 / n) * X.T @ X + 2.0 * lam * np.eye(p)


def autodiff_backend():
    """Return (numpy-like module, grad transform, name) for automatic differentiation.

    Prefers JAX (in double precision, with jit), falls back to Autograd.
    The cost function to differentiate must be written with the returned numpy-like module.

    LLM was used to generate code.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)     # float64: needed for machine-precision checks
        import jax.numpy as jnp
        return jnp, (lambda f: jax.jit(jax.grad(f))), "JAX"
    except ImportError:
        pass
    try:
        import autograd.numpy as anp
        from autograd import grad
        return anp, grad, "Autograd"
    except ImportError:
        raise ImportError("Automatic differentiation needs JAX or Autograd: "
                          "pip install jax   (or: pip install autograd)")


# ----------------------------------------------------------------------------
# Optimisers
# ----------------------------------------------------------------------------
OPTIMISERS = ("gd", "momentum", "adagrad", "rmsprop", "adam")


def make_update(method, shape, gamma=0.9, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8):
    """Update rule with its own state, shared by optimise (full batch) and sgd (mini-batches).

    Returns update(theta, g, eta) -> new theta. Rules (t = number of updates so far, from 1):
        "gd"        theta <- theta - eta g
        "momentum"  v <- gamma v + eta g,                  theta <- theta - v
        "adagrad"   G <- G + g^2,                          theta <- theta - eta g / (sqrt(G) + eps)
        "rmsprop"   s <- rho s + (1 - rho) g^2,            theta <- theta - eta g / (sqrt(s) + eps)
        "adam"      m <- beta1 m + (1 - beta1) g,  s <- beta2 s + (1 - beta2) g^2,
                    theta <- theta - eta m_hat / (sqrt(s_hat) + eps), with bias-corrected m_hat, s_hat
    """
    if method not in OPTIMISERS:
        raise ValueError(f"method must be one of {OPTIMISERS}")
    state = {"v": np.zeros(shape), "s": np.zeros(shape), "t": 0}

    def update(theta, g, eta):
        state["t"] += 1
        t = state["t"]
        if method == "gd":
            return theta - eta * g
        if method == "momentum":
            state["v"] = gamma * state["v"] + eta * g
            return theta - state["v"]
        if method == "adagrad":
            state["s"] = state["s"] + g * g
            return theta - eta * g / (np.sqrt(state["s"]) + eps)
        if method == "rmsprop":
            state["s"] = rho * state["s"] + (1 - rho) * g * g
            return theta - eta * g / (np.sqrt(state["s"]) + eps)
        state["v"] = beta1 * state["v"] + (1 - beta1) * g                       # adam
        state["s"] = beta2 * state["s"] + (1 - beta2) * g * g
        return theta - eta * (state["v"] / (1 - beta1 ** t)) / (np.sqrt(state["s"] / (1 - beta2 ** t)) + eps)

    return update


def optimise(grad, theta0, eta, max_iter, method="gd", theta_ref=None, tol=None, blowup=1e10,
             monitor=None, prox=None, **params):
    """Full-batch gradient descent with a fixed or adaptive learning rate (parts e, f, g).

    method:    one of OPTIMISERS; update rules in make_update. params: gamma, rho, beta1, beta2, eps.
    grad:      function theta -> gradient (or subgradient, e.g. for Lasso)
    theta_ref: optional reference solution. If given, the relative error
               ||theta_k - theta_ref|| / ||theta_ref|| is recorded every iteration, and the
               iteration stops when it drops below tol (if tol is given) or diverges.
    monitor:   optional function of theta recorded every iteration instead of the relative
               error (e.g. the cost). Stopping on tol/divergence still uses theta_ref.
    prox:      optional proximal step (z, eta) -> theta applied after each "gd" step. With
               the gradient of the smooth part and soft thresholding this is ISTA (part g).
    Returns (theta, history, n_iter): history has length n_iter + 1 (entry 0 = starting point)
    and is empty if neither theta_ref nor monitor is given.

    LLM was used to generate code.
    """
    if prox is not None and method != "gd":
        raise ValueError("prox is only defined for plain gradient descent (ISTA)")
    theta = np.array(theta0, dtype=float)
    update = make_update(method, theta.shape, **params)
    ref_norm = None if theta_ref is None else np.linalg.norm(theta_ref)
    history = []
    k = 0
    for k in range(max_iter + 1):
        if theta_ref is not None:
            err = np.linalg.norm(theta - theta_ref) / ref_norm
            history.append(err if monitor is None else monitor(theta))
            if tol is not None and err < tol:
                break
            if not np.isfinite(err) or err > blowup:
                break
        elif monitor is not None:
            history.append(monitor(theta))
        if k == max_iter:
            break
        theta = update(theta, grad(theta), eta)
        if prox is not None:
            theta = prox(theta, eta)
    return theta, np.array(history), k


def sgd(grad_batch, theta0, n, eta, n_epochs, batch_size, method="gd", rng=None, schedule=None,
        monitor=None, blowup=1e10, **params):
    """Stochastic (mini-batch) gradient descent with the same update rules as optimise (part h).

    grad_batch: function (theta, idx) -> gradient computed on the data points idx only,
                e.g. lambda t, idx: ridge_grad(t, X[idx], y[idx], lam).
    n:          number of training points. Every epoch the indices are shuffled and split into
                ceil(n / batch_size) mini-batches; batch_size = n is full-batch GD.
    schedule:   optional function t -> learning rate, t = number of updates so far (from 0);
                default constant eta.
    monitor:    optional function of theta recorded once per epoch (entry 0 = starting point).
    Returns (theta, history, n_updates). Stops early if monitor returns a non-finite value
    or one above blowup (divergence).

    LLM was used to generate code.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.array(theta0, dtype=float)
    update = make_update(method, theta.shape, **params)
    lr = (lambda t: eta) if schedule is None else schedule
    history = [] if monitor is None else [monitor(theta)]
    t = 0
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            theta = update(theta, grad_batch(theta, idx), lr(t))
            t += 1
        if monitor is not None:
            history.append(monitor(theta))
            value = np.max(history[-1])
            if not np.isfinite(value) or value > blowup:
                break
    return theta, np.array(history), t


def gradient_descent(grad, theta0, eta, max_iter, theta_ref=None, tol=None, blowup=1e10):
    """Plain gradient descent with fixed learning rate (part e): optimise(..., method="gd").

    Returns (theta, errors, n_iter) exactly as before; see optimise.

    LLM was used to generate code.
    """
    return optimise(grad, theta0, eta, max_iter, "gd", theta_ref, tol, blowup)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _results_path(kind, part, filename):
    folder = RESULTS_DIR / kind / part
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def media_path(part, filename):
    """Path results/media/<part>/<filename> for figures (folder created if needed).
    
    LLM was used to generate code."""
    return _results_path("media", part, filename)


def output_path(part, filename):
    """Path results/output/<part>/<filename> for tables and text (folder created if needed).
    
    LLM was used to generate code."""
    return _results_path("output", part, filename)


def save_table(part, filename, columns):
    """Save a CSV file in results/output/<part>/.

    columns: dict {column name: 1D array}, all of equal length.

    LLM was used to generate code.
    """
    names = list(columns)
    data = np.column_stack([np.asarray(columns[k], dtype=float) for k in names])
    np.savetxt(output_path(part, filename), data, delimiter=",",
               header=",".join(names), comments="", fmt="%.6g")


def save_text(part, filename, lines):
    """Save a list of lines as a text file in results/output/<part>/.
    
    LLM was used to generate code."""
    output_path(part, filename).write_text("\n".join(lines) + "\n")