"""
FYS-STK3155/4155, Project 1, part g): own code for Lasso regression with gradient methods.

All optimisation uses utils.optimise, the same code as in parts e) and f):
plain GD, momentum, AdaGrad, RMSprop and Adam, with the Lasso subgradient.

Cost:  C(theta) = (1/n)||y - X theta||^2 + lam ||theta||_1
       Same normalisation as OLS/Ridge in parts b) and e): Scikit-Learn's Lasso minimises
       (1/(2n))||y - Xw||^2 + alpha ||w||_1, which is our cost divided by 2, so alpha = lam/2.

1) The kink at theta_j = 0: what automatic differentiation returns for d|theta|/dtheta at 0,
   whether that is a valid subgradient, and a gradient check (analytical subgradient vs AD).
2) Convergence (degree DEG_CONV, lambda LAM_CONV, better conditioned than DEG_MAIN so that
   the budget stays small): the methods of parts e) and f) with the subgradient (analytical
   and AD), compared with ISTA = plain GD followed by soft thresholding (the proximal step of
   the l1 penalty), which handles the kink exactly. Subgradient methods oscillate around the
   kink; they are plotted as an upper envelope (max over the last 20 % of iterations).
3) Coefficient paths theta(lambda): own subgradient GD vs Scikit-Learn (alpha = lam/2), and
   the Ridge path for contrast (shrinkage without sparsity).
4) Test MSE vs lambda for OLS, Ridge and Lasso, all fitted with the same plain GD from
   part e) (Lasso with the subgradient), for two polynomial degrees.

Features standardised and y centred with training statistics, as in the previous parts.
Requires JAX (pip install jax) or Autograd (pip install autograd).
Figures: results/media/g/   Tables and summary: results/output/g/
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import (runge_data, design_matrix, Scaler, MSE, ridge_fit, ridge_grad,
                   ridge_hessian, autodiff_backend, optimise, media_path, save_table, save_text)

PART = "g"
SEED = 2026
N, SIGMA, TEST_SIZE = 100, 0.1, 0.3
DEG_MAIN, LAM_MAIN = 8, 1e-3        # sparse solution: 2 of 8 coefficients exactly zero
DEG_CONV, LAM_CONV = 5, 1e-2        # convergence study: 2 of 5 coefficients exactly zero, kappa ~ 500
N_ITER = 10_000                     # iteration budget in the convergence study
ENVELOPE = 0.2                      # oscillating methods are plotted as max over the last 20 % of iterations
N_ITER_PATH = 20_000                # per lambda in the path studies (warm started)
LAMBDAS = np.logspace(-5, 0, 21)
ZERO_FACTOR = 2.0                   # subgradient GD: |theta_j| < ZERO_FACTOR * eta * lam counts as zero


# ----------------------------------------------------------------------------
# Lasso cost, analytical subgradient and the proximal step of the l1 penalty
# ----------------------------------------------------------------------------
def lasso_cost(theta, X, y, lam):
    return np.mean((y - X @ theta) ** 2) + lam * np.sum(np.abs(theta))


def lasso_subgrad(theta, X, y, lam):
    """(2/n) X^T (X theta - y) + lam sign(theta); sign(0) = 0 is a valid subgradient of |.| at 0."""
    return ridge_grad(theta, X, y, 0.0) + lam * np.sign(theta)


def soft_threshold(z, t):
    """Proximal step of t ||.||_1: shrinks every component by t and sets |z_j| <= t to exactly 0."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


# ----------------------------------------------------------------------------
# Automatic differentiation of the Lasso cost
# ----------------------------------------------------------------------------
anp, ad_grad_transform, AD_NAME = autodiff_backend()


def lasso_cost_ad(theta, X, y, lam):
    return anp.mean((y - X @ theta) ** 2) + lam * anp.sum(anp.abs(theta))


_lasso_grad_ad = ad_grad_transform(lasso_cost_ad)
_abs_grad_ad = ad_grad_transform(lambda t: anp.sum(anp.abs(t)))


def lasso_grad_ad(theta, X, y, lam):
    return np.asarray(_lasso_grad_ad(theta, X, y, lam))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fit_gd(X, y, lam, kind, n_iter, theta0=None):
    """Plain GD from part e) with eta = 1/h_max for OLS, Ridge and Lasso (subgradient)."""
    lam_smooth = lam if kind == "ridge" else 0.0
    eta = 1.0 / np.linalg.eigvalsh(ridge_hessian(X, lam_smooth)).max()
    if kind == "lasso":
        grad = lambda t: lasso_subgrad(t, X, y, lam)
    else:
        grad = lambda t: ridge_grad(t, X, y, lam_smooth)
    theta0 = np.zeros(X.shape[1]) if theta0 is None else theta0
    return optimise(grad, theta0, eta, n_iter, "gd")[0], eta


def lasso_sklearn(X, y, lam):
    """Scikit-Learn reference with alpha = lam/2 (its cost is ours divided by 2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=lam / 2, fit_intercept=False, tol=1e-12, max_iter=1_000_000)
        return model.fit(X, y).coef_


def prepare(x_tr, x_te, y_tr, degree):
    sc = Scaler().fit(design_matrix(x_tr, degree), y_tr)
    return (sc.transform_X(design_matrix(x_tr, degree)),
            sc.transform_X(design_matrix(x_te, degree)), y_tr - sc.y_mean, sc.y_mean)


def upper_envelope(a, frac=ENVELOPE):
    """Max over the last frac * k entries at iteration k: smooth upper envelope on a log axis."""
    return np.array([a[i - max(1, int(frac * i)) + 1:i + 1].max() for i in range(len(a))])


def plain(label):
    """Matplotlib label -> plain text for the summary file."""
    return label.replace("$", "").replace("\\", "")


if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    rng = np.random.default_rng(SEED)
    summary = ["Part g) Lasso regression with gradient methods (utils.optimise, as in parts e and f)",
               f"Data: n = {N}, sigma = {SIGMA}, seed = {SEED}, test_size = {TEST_SIZE} "
               f"(n_train = {len(x_tr)})",
               "Cost: (1/n)||y - X theta||^2 + lambda||theta||_1; Scikit-Learn alpha = lambda/2",
               f"Automatic differentiation: {AD_NAME}", ""]

    # ------------------------------------------------------------------------
    # 1) The kink at zero and the gradient check
    # ------------------------------------------------------------------------
    d_abs = np.asarray(_abs_grad_ad(np.array([0.0, 2.0, -2.0])))
    summary += ["1) Derivative of |theta| from automatic differentiation:",
                f"   at theta = 0: {d_abs[0]:g}  (theta = 2: {d_abs[1]:g}, theta = -2: {d_abs[2]:g})",
                "   Subdifferential of |theta| at 0 is [-1, 1]; the returned value is "
                + ("a valid subgradient." if abs(d_abs[0]) <= 1 else "NOT a valid subgradient."),
                "   Analytical subgradient uses np.sign(0) = 0 (the minimum-norm subgradient)."]

    X_tr, X_te, yc, y_mean = prepare(x_tr, x_te, y_tr, DEG_MAIN)
    theta = rng.standard_normal(DEG_MAIN)                  # no exact zeros: differentiable point
    g_an = lasso_subgrad(theta, X_tr, yc, LAM_MAIN)
    d_rel = np.linalg.norm(lasso_grad_ad(theta, X_tr, yc, LAM_MAIN) - g_an) / np.linalg.norm(g_an)
    theta_kink = theta.copy()
    theta_kink[[1, 4]] = 0.0                               # two components exactly on the kink
    diff_kink = (lasso_grad_ad(theta_kink, X_tr, yc, LAM_MAIN)
                 - lasso_subgrad(theta_kink, X_tr, yc, LAM_MAIN))
    summary += [f"   Gradient check at random theta, degree {DEG_MAIN}, lambda {LAM_MAIN:.0e}: "
                f"relative difference AD vs analytical = {d_rel:.1e}",
                "   With theta_2 = theta_5 = 0: AD - analytical per component = "
                + np.array2string(diff_kink, precision=2, max_line_width=200)
                + f"  (lambda = {LAM_MAIN:g})", ""]

    # ------------------------------------------------------------------------
    # 2) Convergence: methods of parts e) and f) with the subgradient, vs ISTA
    # ------------------------------------------------------------------------
    Xc, _, ycc, _ = prepare(x_tr, x_te, y_tr, DEG_CONV)
    h_max = np.linalg.eigvalsh(ridge_hessian(Xc)).max()
    eta = 1.0 / h_max
    cost = lambda t: lasso_cost(t, Xc, ycc, LAM_CONV)
    sub = lambda t: lasso_subgrad(t, Xc, ycc, LAM_CONV)
    sub_ad = lambda t: lasso_grad_ad(t, Xc, ycc, LAM_CONV)
    smooth = lambda t: ridge_grad(t, Xc, ycc, 0.0)
    prox = lambda z, e: soft_threshold(z, e * LAM_CONV)

    theta_skl = lasso_sklearn(Xc, ycc, LAM_CONV)
    zeros_ref = theta_skl == 0

    # (label, gradient, learning rate, method, prox, color)
    runs = [("GD (subgradient)", sub, eta, "gd", None, "C0"),
            (f"GD, {AD_NAME} gradient", sub_ad, eta, "gd", None, "C0"),
            ("Momentum", sub, 0.1 * eta, "momentum", None, "C1"),
            ("AdaGrad", sub, 0.1, "adagrad", None, "C2"),
            ("RMSprop", sub, 1e-4, "rmsprop", None, "C4"),
            ("Adam", sub, 1e-3, "adam", None, "C5"),
            ("ISTA (GD + soft thresholding)", smooth, eta, "gd", prox, "k")]
    zero_set = lambda t: np.max(np.abs(t[zeros_ref]))       # largest coefficient that should be 0
    results, zero_traces = {}, {}
    for name, grad, step, method, px, _ in runs:
        th, hist, it = optimise(grad, np.zeros(DEG_CONV), step, N_ITER, method, prox=px,
                                monitor=lambda t: (cost(t), zero_set(t)))
        results[name] = (th, hist[:, 0], it)
        zero_traces[name] = hist[:, 1]
    C_star = min(cost(theta_skl), min(hist.min() for _, hist, _ in results.values()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    k = np.arange(1, N_ITER + 2)
    for name, _, _, method, px, c in runs:
        if name.startswith(f"GD, {AD_NAME}"):              # AD run: identical to the analytical one
            continue
        gap = np.maximum(results[name][1] - C_star, 1e-16)
        zs = np.maximum(zero_traces[name], 1e-17)
        if px is None:                                      # subgradient methods oscillate
            gap, zs = upper_envelope(gap), upper_envelope(zs)
        axes[0].loglog(k, gap, c=c, lw=2 if px else 1.4, label=name)
        axes[1].loglog(k, zs, c=c, lw=2 if px else 1.4, label=name)
    axes[0].set(xlabel="Iteration k + 1", ylabel=r"$C(\theta_k) - C^*$",
                title=f"Lasso, degree {DEG_CONV}, $\\lambda$ = {LAM_CONV:.0e}: cost gap")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].axhline(eta * LAM_CONV, c="gray", ls=":", label=r"$\eta\lambda$ (GD step at the kink)")
    axes[1].set(xlabel="Iteration k + 1",
                ylabel=r"$\max_j |\theta_j|$ over the zero set (1e-17 = exactly 0)",
                title=f"Coefficients that are exactly zero in the solution ({zeros_ref.sum()} of {DEG_CONV})")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.text(0.5, 0.005, "Subgradient methods: upper envelope (max over the last 20 % of iterations)",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(media_path(PART, "convergence.png"), dpi=150)

    rows = {"method": [], "learning_rate": [], "final_cost_gap": [], "median_gap_last_1000": [],
            "rel_param_error": [], "exact_zeros": [], "max_abs_theta_on_zero_set": [],
            "iterations_to_gap_1e-8": []}
    summary.append(f"2) Convergence, degree {DEG_CONV}, lambda {LAM_CONV:.0e}, {N_ITER:,} iterations, "
                   f"1/h_max = {eta:.4f}; reference has {zeros_ref.sum()} zeros:")
    for i, ((name, _, step, _, _, _), (th, hist, _)) in enumerate(zip(runs, results.values())):
        gap = hist - C_star
        k8 = np.flatnonzero(gap < 1e-8)
        vals = (i, step, gap[-1], np.median(gap[-1000:]),
                np.linalg.norm(th - theta_skl) / np.linalg.norm(theta_skl),
                np.sum(th == 0), np.abs(th[zeros_ref]).max(), k8[0] if k8.size else np.nan)
        for key, v in zip(rows, vals):
            rows[key].append(v)
        summary.append(f"   {name:30s} eta {step:.1e}: gap {vals[2]:.1e} (median last 1000 {vals[3]:.1e}), "
                       f"param error {vals[4]:.1e}, exact zeros {vals[5]}, "
                       f"max |theta_j| on zero set {vals[6]:.1e}, gap < 1e-8 after "
                       f"{'-' if np.isnan(vals[7]) else int(vals[7])} it.")
    save_table(PART, "convergence.csv", rows)
    summary += ["   (method index in convergence.csv follows the order above)",
                f"   ISTA vs Scikit-Learn: max |diff| = "
                f"{np.max(np.abs(results[runs[6][0]][0] - theta_skl)):.1e}", ""]

    # ------------------------------------------------------------------------
    # 3) Coefficient paths: own subgradient GD vs Scikit-Learn, and Ridge for contrast
    # ------------------------------------------------------------------------
    own, skl, ridge = [], [], []
    th = np.zeros(DEG_MAIN)
    for lam in LAMBDAS[::-1]:                              # large -> small lambda, warm start
        th, eta_path = fit_gd(X_tr, yc, lam, "lasso", N_ITER_PATH, theta0=th)
        own.append(th)
        skl.append(lasso_sklearn(X_tr, yc, lam))
        ridge.append(ridge_fit(X_tr, yc, lam))
    own, skl, ridge = (np.array(a)[::-1] for a in (own, skl, ridge))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, DEG_MAIN))
    for j in range(DEG_MAIN):
        axes[0].semilogx(LAMBDAS, own[:, j], "-", c=colors[j], label=rf"$\theta_{{{j + 1}}}$")
        axes[0].semilogx(LAMBDAS, skl[:, j], "o", c=colors[j], ms=4, mfc="none")
        axes[1].semilogx(LAMBDAS, ridge[:, j], "-", c=colors[j])
    axes[0].set(xlabel=r"$\lambda$", ylabel="Coefficient",
                title=f"Lasso, degree {DEG_MAIN}: own GD (lines), Scikit-Learn (circles)")
    axes[1].set(xlabel=r"$\lambda$", title=f"Ridge, degree {DEG_MAIN} (closed form)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(media_path(PART, "coefficient_paths.png"), dpi=150)

    near_zero = (np.abs(own) < ZERO_FACTOR * eta_path * LAMBDAS[:, None]).sum(axis=1)
    diff = np.abs(own - skl).max(axis=1)
    cost_gap = np.array([lasso_cost(a, X_tr, yc, l) - lasso_cost(b, X_tr, yc, l)
                         for a, b, l in zip(own, skl, LAMBDAS)])
    cols = {"lambda": LAMBDAS, "near_zero_own": near_zero, "exact_zero_own": (own == 0).sum(axis=1),
            "exact_zero_sklearn": (skl == 0).sum(axis=1), "max_abs_diff_own_sklearn": diff,
            "cost_own_minus_sklearn": cost_gap}
    cols.update({f"theta_{j + 1}": own[:, j] for j in range(DEG_MAIN)})
    save_table(PART, "coefficient_paths.csv", cols)
    i_max = np.argmax(diff)
    summary += [f"3) Coefficient paths, degree {DEG_MAIN}, {N_ITER_PATH:,} GD iterations per lambda "
                f"(warm start), eta = 1/h_max:",
                f"   max |own - Scikit-Learn| over the path: {diff.max():.1e} (at lambda "
                f"{LAMBDAS[i_max]:.0e}, where the cost differs by {cost_gap[i_max]:.1e})",
                f"   max cost own - cost Scikit-Learn over the path: {cost_gap.max():.1e}",
                f"   zero coefficients per lambda, own (|theta_j| < {ZERO_FACTOR:g} eta lambda) / "
                "Scikit-Learn (exact): "
                + ", ".join(f"{l:.0e}: {a}/{b}" for l, a, b in
                            zip(LAMBDAS, near_zero, cols["exact_zero_sklearn"])),
                f"   exact zeros from own subgradient GD over the whole path: {int((own == 0).sum())}",
                "   Ridge: no coefficient is exactly zero for any lambda: "
                + str(bool(np.all(ridge != 0))), ""]

    # ------------------------------------------------------------------------
    # 4) Test MSE vs lambda, OLS, Ridge and Lasso with the same plain GD
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    summary.append(f"4) Test MSE with plain GD for all three ({N_ITER_PATH:,} iterations, eta = 1/h_max; "
                   "dashed = closed form / Scikit-Learn):")
    for ax, p in zip(axes, (DEG_MAIN, 15)):
        Xp, Xp_te, ycp, ymp = prepare(x_tr, x_te, y_tr, p)
        mse = lambda th: MSE(y_te, Xp_te @ th + ymp)
        mse_ols = mse(fit_gd(Xp, ycp, 0.0, "ols", N_ITER_PATH)[0])
        mse_ols_cf = mse(ridge_fit(Xp, ycp, 0.0))
        mse_r, mse_r_cf, mse_l, mse_l_skl = [], [], [], []
        th = np.zeros(p)
        for lam in LAMBDAS[::-1]:
            th = fit_gd(Xp, ycp, lam, "lasso", N_ITER_PATH, theta0=th)[0]
            mse_l.append(mse(th))
            mse_l_skl.append(mse(lasso_sklearn(Xp, ycp, lam)))
            mse_r.append(mse(fit_gd(Xp, ycp, lam, "ridge", N_ITER_PATH)[0]))
            mse_r_cf.append(mse(ridge_fit(Xp, ycp, lam)))
        mse_r, mse_r_cf, mse_l, mse_l_skl = (np.array(a)[::-1] for a in (mse_r, mse_r_cf, mse_l, mse_l_skl))

        ax.axhline(mse_ols, c="k", lw=1.2, label="OLS (GD)")
        ax.axhline(mse_ols_cf, c="k", ls="--", lw=0.8, label="OLS (closed form)")
        ax.semilogx(LAMBDAS, mse_r, "-s", ms=3, c="C0", label="Ridge (GD)")
        ax.semilogx(LAMBDAS, mse_r_cf, "--", c="C9", lw=1.2, label="Ridge (closed form)")
        ax.semilogx(LAMBDAS, mse_l, "-o", ms=3, c="C3", label="Lasso (GD)")
        ax.semilogx(LAMBDAS, mse_l_skl, "--", c="C1", lw=1.2, label="Lasso (Scikit-Learn)")
        ax.set(xlabel=r"$\lambda$", title=f"Degree {p}", yscale="log")
        ax.legend(fontsize=8)
        save_table(PART, f"test_mse_degree{p}.csv",
                   {"lambda": LAMBDAS, "ridge_gd": mse_r, "ridge_closed_form": mse_r_cf,
                    "lasso_gd": mse_l, "lasso_sklearn": mse_l_skl})
        i_r, i_l, i_ls = np.argmin(mse_r), np.argmin(mse_l), np.argmin(mse_l_skl)
        summary += [f"   degree {p:2d}: OLS GD {mse_ols:.5f} (closed form {mse_ols_cf:.5f}); "
                    f"best Ridge GD {mse_r[i_r]:.5f} at lambda {LAMBDAS[i_r]:.0e}; "
                    f"best Lasso GD {mse_l[i_l]:.5f} at lambda {LAMBDAS[i_l]:.0e} "
                    f"(Scikit-Learn {mse_l_skl[i_ls]:.5f} at {LAMBDAS[i_ls]:.0e})"]
    axes[0].set_ylabel("Test MSE")
    fig.suptitle("Test MSE vs penalty (single train/test split; model selection with CV in part i)")
    fig.tight_layout()
    fig.savefig(media_path(PART, "test_mse.png"), dpi=150)

    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))
    plt.show()