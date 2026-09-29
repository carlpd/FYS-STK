"""
FYS-STK3155/4155, Project 1, part d): k-fold cross-validation.

1) OLS: CV MSE vs polynomial degree with Scikit-Learn (KFold + cross_val_score), k = 5 and 10.
2) Own k-fold code with the SAME fold assignment, checked against Scikit-Learn.
3) Ridge: CV MSE as a function of both polynomial degree and lambda, k = 5 and 10.
4) Comparison with the bootstrap estimate from part c) on the same data set.
5) Check of scaling inside vs outside the cross-validation loop.

Preprocessing (standardisation of the polynomial features) sits inside a Pipeline, so in
every fold it is fitted on the training folds only and applied unchanged to the test fold.

Lambda convention as in part b): cost (1/n)||y - X theta||^2 + lambda ||theta||^2, i.e.
Scikit-Learn Ridge(alpha = n_train * lambda), where n_train is the size of the training folds.

Data: the same data set as in part c) (n = 100, sigma = 0.1, seed = 2026).
Figures: results/media/d/   Tables and summary: results/output/d/
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from utils import (runge_data, design_matrix, Scaler, ols_fit, ridge_fit,
                   bootstrap_predictions, bias_variance,
                   media_path, save_table, save_text)

PART = "d"
SEED = 2026
N, SIGMA = 100, 0.1
DEGREES = np.arange(1, 21)
K_VALUES = (5, 10)
LAMS = np.logspace(-8, 1, 10)
N_BOOT, TEST_SIZE = 300, 0.3          # bootstrap settings from part c)


# ----------------------------------------------------------------------------
# Models and cross-validation
# ----------------------------------------------------------------------------
def sk_model(degree, lam=None, n_train=None):
    """Scikit-Learn pipeline: polynomial features x..x^p -> standardisation -> OLS/Ridge.
    lam = None gives OLS; otherwise Ridge with alpha = n_train * lam (our convention)."""
    reg = LinearRegression() if lam is None else Ridge(alpha=n_train * lam)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def sk_cv_mse(x, y, degree, kfold, lam=None):
    """Fold MSEs from Scikit-Learn cross_val_score."""
    n_train = len(x) - len(x) // kfold.get_n_splits()
    scores = cross_val_score(sk_model(degree, lam, n_train), x[:, None], y, cv=kfold,
                             scoring="neg_mean_squared_error")
    return -scores


def own_cv(x, y, degree, kfold, lam=0.0):
    """Own k-fold CV with utils (Scaler + ols_fit/ridge_fit), same folds as `kfold`.
    Returns (fold MSEs, squared error of every data point when it is in the test fold)."""
    fold_mse, sq_err = [], np.empty(len(x))
    for tr, te in kfold.split(x):
        X_tr, X_te = design_matrix(x[tr], degree), design_matrix(x[te], degree)
        sc = Scaler().fit(X_tr, y[tr])                      # training folds only
        theta = ridge_fit(sc.transform_X(X_tr), y[tr] - sc.y_mean, lam)
        err = (y[te] - (sc.transform_X(X_te) @ theta + sc.y_mean)) ** 2
        sq_err[te] = err
        fold_mse.append(err.mean())
    return np.array(fold_mse), sq_err


def leaky_cv_ridge(x, y, degree, kfold, lam):
    """For comparison only: standardisation fitted on ALL data before the CV loop (leakage)."""
    X_all = design_matrix(x, degree)
    X_s = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    fold_mse = []
    for tr, te in kfold.split(x):
        y_mean = y[tr].mean()
        theta = ridge_fit(X_s[tr], y[tr] - y_mean, lam)
        fold_mse.append(np.mean((y[te] - (X_s[te] @ theta + y_mean)) ** 2))
    return np.array(fold_mse)


def int_ticks(fig):
    for a in fig.axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    edges = np.isin(np.arange(N), [np.argmin(x), np.argmax(x)])   # the two extreme x values
    kfolds = {k: KFold(n_splits=k, shuffle=True, random_state=SEED) for k in K_VALUES}
    summary = ["Part d) k-fold cross-validation",
               f"Data: n = {N}, sigma = {SIGMA}, seed = {SEED} (same data set as part c)",
               f"k = {K_VALUES}, KFold(shuffle=True, random_state={SEED}); degrees "
               f"{DEGREES[0]}-{DEGREES[-1]}; lambda in [{LAMS[0]:.0e}, {LAMS[-1]:.0e}] (log grid)",
               "Scaling inside the CV loop (Pipeline / own Scaler fitted on the training folds)",
               "Ridge: Scikit-Learn alpha = n_train * lambda", ""]

    # ------------------------------------------------------------------------
    # 1) + 2) OLS: Scikit-Learn CV and own CV with the same folds
    # ------------------------------------------------------------------------
    cv = {k: np.array([sk_cv_mse(x, y, p, kfolds[k]).mean() for p in DEGREES]) for k in K_VALUES}
    cv_std = {k: np.array([sk_cv_mse(x, y, p, kfolds[k]).std() for p in DEGREES]) for k in K_VALUES}
    own, own_noedge = {}, {}
    for k in K_VALUES:
        res = [own_cv(x, y, p, kfolds[k]) for p in DEGREES]
        own[k] = np.array([r[0].mean() for r in res])
        own_noedge[k] = np.array([r[1][~edges].mean() for r in res])
    rel_diff = {k: np.abs(own[k] - cv[k]) / cv[k] for k in K_VALUES}

    # ------------------------------------------------------------------------
    # 4) Bootstrap from part c) on the same data (same split and seed as in part c)
    # ------------------------------------------------------------------------
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    inside = (x_te >= x_tr.min()) & (x_te <= x_tr.max())
    rng = np.random.default_rng(SEED)
    boot_all, boot_in = np.zeros(len(DEGREES)), np.zeros(len(DEGREES))
    for j, p in enumerate(DEGREES):
        preds = bootstrap_predictions(x_tr, y_tr, x_te, p, N_BOOT, rng)
        boot_all[j] = bias_variance(y_te, preds)[0]
        boot_in[j] = bias_variance(y_te[inside], preds[inside])[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for k, m in zip(K_VALUES, ("o", "s")):
        ax.plot(DEGREES, cv[k], f"-{m}", ms=4, label=f"CV, k = {k}")
    ax.plot(DEGREES, boot_all, "k--^", ms=4, label="Bootstrap (part c), all test points")
    ax.set(title="All data points")
    ax = axes[1]
    for k, m in zip(K_VALUES, ("o", "s")):
        ax.plot(DEGREES, own_noedge[k], f"-{m}", ms=4, label=f"CV, k = {k}, without the 2 extreme x")
    ax.plot(DEGREES, boot_in, "k--^", ms=4, label="Bootstrap, test points inside training range")
    ax.set(title="Without extrapolation to the interval ends")
    for ax in axes:
        ax.axhline(SIGMA**2, ls=":", c="gray", label=r"$\sigma^2$")
        ax.set(xlabel="Polynomial degree", ylabel="Estimated test MSE", yscale="log")
        ax.legend(fontsize=8)
    fig.suptitle(f"OLS: cross-validation vs bootstrap (n = {N}, $\\sigma$ = {SIGMA})")
    int_ticks(fig)
    fig.tight_layout()
    fig.savefig(media_path(PART, "cv_vs_bootstrap_ols.png"), dpi=150)

    # Spread between folds (k = 10): individual fold MSEs
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in DEGREES:
        f = sk_cv_mse(x, y, p, kfolds[10])
        ax.plot(np.full(len(f), p), f, "o", c="tab:blue", alpha=0.35, ms=4)
    ax.plot(DEGREES, cv[10], "k-", lw=2, label="Mean over folds")
    ax.axhline(SIGMA**2, ls=":", c="gray", label=r"$\sigma^2$")
    ax.set(xlabel="Polynomial degree", ylabel="Fold MSE", yscale="log",
           title="OLS, k = 10: MSE of each test fold")
    ax.legend()
    int_ticks(fig)
    fig.tight_layout()
    fig.savefig(media_path(PART, "cv_folds_ols.png"), dpi=150)

    save_table(PART, "cv_ols.csv",
               {"degree": DEGREES, "cv5_mse": cv[5], "cv5_fold_std": cv_std[5],
                "cv10_mse": cv[10], "cv10_fold_std": cv_std[10],
                "cv5_without_extreme_x": own_noedge[5], "cv10_without_extreme_x": own_noedge[10],
                "bootstrap_all": boot_all, "bootstrap_inside": boot_in})

    opt = lambda arr: DEGREES[np.argmin(arr)]
    summary += ["OLS, optimal degree (minimum estimated test MSE):"]
    for k in K_VALUES:
        summary.append(f"  CV k = {k:2d}: degree {opt(cv[k]):2d} (MSE {cv[k].min():.4g});  "
                       f"without the 2 extreme x: degree {opt(own_noedge[k]):2d} "
                       f"(MSE {own_noedge[k].min():.4g})")
    summary += [f"  Bootstrap, all test points:    degree {opt(boot_all):2d} (MSE {boot_all.min():.4g})",
                f"  Bootstrap, inside train range: degree {opt(boot_in):2d} (MSE {boot_in.min():.4g})",
                f"  Extreme x values: {x[edges].round(4)}", ""]
    for k in K_VALUES:
        ok = DEGREES <= 12
        summary.append(f"  Own k-fold vs Scikit-Learn (k = {k}), relative difference in CV MSE: "
                       f"max {rel_diff[k][ok].max():.1e} for degree <= 12, "
                       f"max {rel_diff[k].max():.1e} for all degrees")
    summary.append("")

    # ------------------------------------------------------------------------
    # 3) Ridge: CV MSE over degree x lambda, k = 5 and 10
    # ------------------------------------------------------------------------
    grid = {k: np.array([[sk_cv_mse(x, y, p, kfolds[k], lam).mean() for p in DEGREES]
                         for lam in LAMS]) for k in K_VALUES}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    vmin = np.log10(min(g.min() for g in grid.values()))
    vmax = np.log10(max(np.percentile(g, 95) for g in grid.values()))
    for ax, k in zip(axes, K_VALUES):
        im = ax.imshow(np.log10(grid[k]), origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[DEGREES[0] - 0.5, DEGREES[-1] + 0.5,
                               np.log10(LAMS[0]) - 0.5, np.log10(LAMS[-1]) + 0.5])
        i, j = np.unravel_index(np.argmin(grid[k]), grid[k].shape)
        ax.plot(DEGREES[j], np.log10(LAMS[i]), "r*", ms=14,
                label=f"min: degree {DEGREES[j]}, $\\lambda$ = {LAMS[i]:.0e}")
        ax.set(xlabel="Polynomial degree", title=f"Ridge, k = {k}")
        ax.legend(loc="upper right", fontsize=8)
        save_table(PART, f"cv_ridge_k{k}_grid.csv",
                   {"log10_lambda": np.log10(LAMS),
                    **{f"deg_{p}": grid[k][:, c] for c, p in enumerate(DEGREES)}})
        summary.append(f"Ridge, k = {k:2d}: minimum CV MSE {grid[k][i, j]:.4g} at degree {DEGREES[j]}, "
                       f"lambda = {LAMS[i]:.0e}")
    axes[0].set_ylabel(r"$\log_{10}\lambda$")
    int_ticks(fig)                               # before the colour bar is added
    fig.colorbar(im, ax=axes, label=r"$\log_{10}$ CV MSE (colour scale capped at the 95th percentile)")
    fig.savefig(media_path(PART, "cv_ridge_heatmap.png"), dpi=150, bbox_inches="tight")

    # CV MSE vs degree for selected lambdas (k = 10), with OLS
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(DEGREES, cv[10], "k--o", ms=3, label="OLS")
    for lam in (1e-6, 1e-4, 1e-2, 1e0):
        i = np.argmin(np.abs(LAMS - lam))
        ax.plot(DEGREES, grid[10][i], "-o", ms=3, label=rf"Ridge, $\lambda$ = {LAMS[i]:.0e}")
    ax.axhline(SIGMA**2, ls=":", c="gray", label=r"$\sigma^2$")
    ax.set(xlabel="Polynomial degree", ylabel="CV MSE", yscale="log",
           title="10-fold CV: Ridge vs OLS")
    ax.legend(fontsize=8)
    int_ticks(fig)
    fig.tight_layout()
    fig.savefig(media_path(PART, "cv_ridge_vs_degree.png"), dpi=150)

    # Own Ridge CV vs Scikit-Learn for a few settings
    checks = []
    for k in K_VALUES:
        for p, lam in ((5, 1e-4), (10, 1e-3), (15, 1e-5)):
            own_r = own_cv(x, y, p, kfolds[k], lam)[0].mean()
            sk_r = sk_cv_mse(x, y, p, kfolds[k], lam).mean()
            checks.append(abs(own_r - sk_r) / sk_r)
    summary += [f"Own Ridge k-fold vs Scikit-Learn (alpha = n_train*lambda), 6 settings: "
                f"max relative difference {max(checks):.1e}", ""]

    # ------------------------------------------------------------------------
    # 5) Scaling inside vs outside the CV loop (Ridge, k = 5)
    # ------------------------------------------------------------------------
    summary.append("Scaling inside vs outside the CV loop (Ridge, k = 5):")
    for p, lam in ((5, 1e-4), (10, 1e-3), (15, 1e-5), (15, 1e-2)):
        inside_cv = own_cv(x, y, p, kfolds[5], lam)[0].mean()
        leaky = leaky_cv_ridge(x, y, p, kfolds[5], lam).mean()
        summary.append(f"  degree {p:2d}, lambda {lam:.0e}: inside {inside_cv:.5g}, outside {leaky:.5g}, "
                       f"relative difference {abs(leaky - inside_cv) / inside_cv:.1e}")

    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))

    plt.show()