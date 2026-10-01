"""
FYS-STK3155/4155, Project 1, part i): final model selection for the Runge function with OLS,
Ridge and Lasso, using cross-validation, and a bias-variance reading of the three methods.

1) Cross-validation (setup of part d): repeated K-fold on the TRAINING set, with the Scaler
   fitted on the training folds only (inside the loop). OLS as a function of the polynomial
   degree, Ridge and Lasso as functions of degree and lambda. Reported: CV MSE and its
   standard error over the folds. Best model per method = lowest CV MSE; in addition the
   one-standard-error rule (simplest model within one SE of the minimum: lowest degree,
   then largest lambda).
2) Final check: each selected model refitted on the whole training set and evaluated ONCE on
   the held-out test set (never used for any choice).
3) Bias-variance (part c): bootstrap over training sets with the TRUE f(x_test) as reference,
   so that MSE_f = bias^2 + variance exactly (the noise sigma^2 is not part of it):
   a) vs lambda at a high degree DEG_BV for Ridge and Lasso (OLS as reference),
   b) vs degree for OLS, and for Ridge and Lasso at their CV-optimal lambda per degree.
   The SAME bootstrap resamples are used for every lambda, degree and method (common random
   numbers), so differences along the curves come from the model, not from the resampling.
   The variance is dominated by resamples that miss points near x = +-1, where high-degree
   polynomials extrapolate wildly (Runge's phenomenon); this is part of what is measured.

Conventions as in the previous parts: Ridge cost (1/n)||y - X theta||^2 + lam ||theta||^2
(closed form, utils.ridge_fit), Lasso cost (1/n)||y - X theta||^2 + lam ||theta||_1, solved with
Scikit-Learn's coordinate descent (alpha = lam/2), which part g) showed agrees with our own
proximal gradient code to ~1e-11; coordinate descent is used here only for speed over the
large (degree, lambda, fold) grid.

Lasso fits that do not converge within LASSO_MAX_ITER (tiny lambda at high degree, where the
problem is as ill-conditioned as OLS) are EXCLUDED from model selection (CV MSE set to NaN)
and counted in the summary: an unconverged iterate depends on the starting point and acts as
an uncontrolled regulariser (early stopping), so its CV error does not describe the Lasso.
For such lambda the Lasso is close to OLS anyway, which is covered by the OLS curve.
Figures: results/media/i/   Tables and summary: results/output/i/
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold, train_test_split

from utils import (runge, runge_data, design_matrix, Scaler, MSE, R2, ols_fit, ridge_fit,
                   bootstrap_predictions, bias_variance, media_path, save_table, save_text)

PART = "i"
SEED = 2026
N, SIGMA, TEST_SIZE = 100, 0.1, 0.3
K_FOLDS, N_REPEATS = 5, 3            # repeated K-fold: CV estimate less dependent on one split
DEGREES = np.arange(1, 16)
LAMBDAS = np.logspace(-7, 0, 15)
DEG_BV = 12                          # high degree for the bias-variance study vs lambda
N_BOOT = 200
METHODS = ("OLS", "Ridge", "Lasso")
COLORS = {"OLS": "k", "Ridge": "C0", "Lasso": "C3"}

LASSO_TOL, LASSO_MAX_ITER = 1e-8, 100_000

warnings.simplefilter("ignore", ConvergenceWarning)    # handled explicitly: see lasso_fit


def new_lasso(warm_start=False):
    return Lasso(fit_intercept=False, tol=LASSO_TOL, max_iter=LASSO_MAX_ITER, warm_start=warm_start)


def lasso_fit(X, y, lam, model=None):
    """Lasso with our convention (alpha = lam/2). Pass a model to warm start along a lambda path.
    Returns (coefficients, converged)."""
    model = new_lasso() if model is None else model
    model.set_params(alpha=lam / 2).fit(X, y)
    return model.coef_.copy(), model.n_iter_ < LASSO_MAX_ITER


def fold_data(x, y, tr, va, degree):
    """Scaler fitted on the training folds only; returns scaled train/validation data."""
    sc = Scaler().fit(design_matrix(x[tr], degree), y[tr])
    return (sc.transform_X(design_matrix(x[tr], degree)), y[tr] - sc.y_mean,
            sc.transform_X(design_matrix(x[va], degree)), y[va], sc.y_mean)


def cross_validate(x, y, folds):
    """CV MSE per fold. Returns dict method -> array (n_degrees, n_lambdas, n_folds);
    OLS has n_lambdas = 1. Unconverged Lasso fits give NaN for the whole (degree, lambda) cell."""
    out = {"OLS": np.empty((len(DEGREES), 1, len(folds))),
           "Ridge": np.empty((len(DEGREES), len(LAMBDAS), len(folds))),
           "Lasso": np.empty((len(DEGREES), len(LAMBDAS), len(folds)))}
    for i, p in enumerate(DEGREES):
        for f, (tr, va) in enumerate(folds):
            X, yc, X_va, y_va, y_mean = fold_data(x, y, tr, va, p)
            out["OLS"][i, 0, f] = MSE(y_va, X_va @ ols_fit(X, yc) + y_mean)
            model = new_lasso(warm_start=True)
            for j in range(len(LAMBDAS) - 1, -1, -1):           # large -> small lambda, warm start
                lam = LAMBDAS[j]
                out["Ridge"][i, j, f] = MSE(y_va, X_va @ ridge_fit(X, yc, lam) + y_mean)
                coef, ok = lasso_fit(X, yc, lam, model)
                out["Lasso"][i, j, f] = MSE(y_va, X_va @ coef + y_mean) if ok else np.nan
    bad = np.isnan(out["Lasso"]).any(axis=2)
    out["Lasso"][bad] = np.nan
    return out


def select(cv, rule="min"):
    """Index (degree, lambda) of the best model: lowest mean CV MSE, or the one-SE rule."""
    mean, se = cv.mean(axis=2), cv.std(axis=2, ddof=1) / np.sqrt(cv.shape[2])
    i, j = np.unravel_index(np.nanargmin(mean), mean.shape)
    if rule == "min":
        return i, j
    ok = mean <= mean[i, j] + se[i, j]
    i1 = np.flatnonzero(ok.any(axis=1))[0]                  # lowest degree within one SE
    j1 = np.flatnonzero(ok[i1])[-1]                         # then the largest lambda
    return i1, j1


def fitter(method, lam):
    if method == "OLS":
        return ols_fit
    if method == "Ridge":
        return lambda X, y: ridge_fit(X, y, lam)
    return lambda X, y: lasso_fit(X, y, lam)[0]


def describe(method, i, j):
    return f"degree {DEGREES[i]}" + ("" if method == "OLS" else f", lambda {LAMBDAS[j]:.1e}")


if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    folds = list(RepeatedKFold(n_splits=K_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(x_tr))
    summary = ["Part i) Final model selection: OLS, Ridge and Lasso with cross-validation",
               f"Data: n = {N}, sigma = {SIGMA} (sigma^2 = {SIGMA ** 2:.3f}), seed = {SEED}, test_size = "
               f"{TEST_SIZE} (n_train = {len(x_tr)}, n_test = {len(x_te)})",
               f"CV: {K_FOLDS}-fold repeated {N_REPEATS} times on the training set ({len(folds)} folds), "
               "Scaler fitted inside each fold",
               f"Degrees {DEGREES[0]}-{DEGREES[-1]}, lambda {LAMBDAS[0]:.0e}-{LAMBDAS[-1]:.0e} "
               f"({len(LAMBDAS)} values); Lasso alpha = lambda/2", ""]

    # ------------------------------------------------------------------------
    # 1) Cross-validation
    # ------------------------------------------------------------------------
    cv = cross_validate(x_tr, y_tr, folds)
    mean = {m: cv[m].mean(axis=2) for m in METHODS}
    se = {m: cv[m].std(axis=2, ddof=1) / np.sqrt(len(folds)) for m in METHODS}
    best_lam_per_degree = {m: np.nanargmin(mean[m], axis=1) for m in METHODS}
    n_bad = int(np.isnan(mean["Lasso"]).sum())
    bad_lams = [f"degree {p}: lambda <= {LAMBDAS[np.flatnonzero(np.isnan(mean['Lasso'][i]))].max():.0e}"
                for i, p in enumerate(DEGREES) if np.isnan(mean["Lasso"][i]).any()]
    summary.append(f"Lasso cells excluded (not converged in {LASSO_MAX_ITER:,} CD iterations, tol {LASSO_TOL:.0e}): "
                   f"{n_bad} of {mean['Lasso'].size}" + (" (" + "; ".join(bad_lams) + ")" if bad_lams else ""))
    summary.append("")

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1)
    for m in METHODS:
        j = best_lam_per_degree[m]
        mu = mean[m][np.arange(len(DEGREES)), j]
        s = se[m][np.arange(len(DEGREES)), j]
        ax.semilogy(DEGREES, mu, "-o", ms=4, c=COLORS[m],
                    label=m if m == "OLS" else f"{m} (best $\\lambda$ per degree)")
        ax.fill_between(DEGREES, mu - s, mu + s, color=COLORS[m], alpha=0.15)
    ax.axhline(SIGMA ** 2, c="gray", ls=":", label=r"Noise floor $\sigma^2$")
    ax.set(xlabel="Polynomial degree", ylabel="CV MSE (band: $\\pm$1 SE)",
           title=f"Cross-validated MSE ({K_FOLDS}-fold x {N_REPEATS})")
    ax.legend(fontsize=8)
    both = np.concatenate([mean["Ridge"].ravel(), mean["Lasso"].ravel()])
    vmin, vmax = np.nanmin(both), np.nanquantile(both, 0.95)
    for k, m in enumerate(("Ridge", "Lasso")):
        ax = fig.add_subplot(1, 3, k + 2)
        im = ax.pcolormesh(np.log10(LAMBDAS), DEGREES, mean[m], shading="nearest",
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        i, j = select(cv[m])
        i1, j1 = select(cv[m], "1se")
        ax.plot(np.log10(LAMBDAS[j]), DEGREES[i], "r*", ms=14, label="Lowest CV MSE")
        ax.plot(np.log10(LAMBDAS[j1]), DEGREES[i1], "wo", ms=8, mfc="none", mew=2, label="One-SE rule")
        ax.set(xlabel=r"$\log_{10}\lambda$", ylabel="Polynomial degree",
               title=f"{m}: CV MSE" + (" (white = not converged, excluded)" if m == "Lasso" else ""))
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(media_path(PART, "cv_mse.png"), dpi=150)

    for m in METHODS:
        cols = {"degree": np.repeat(DEGREES, mean[m].shape[1]),
                "lambda": np.tile(LAMBDAS if m != "OLS" else [0.0], len(DEGREES)),
                "cv_mse": mean[m].ravel(), "cv_se": se[m].ravel()}
        save_table(PART, f"cv_{m.lower()}.csv", cols)

    # ------------------------------------------------------------------------
    # 2) Selected models: refit on the training set, evaluate once on the test set
    # ------------------------------------------------------------------------
    rows = {"method": [], "rule": [], "degree": [], "lambda": [], "cv_mse": [], "cv_se": [],
            "test_mse": [], "test_r2": [], "nonzero_coefficients": []}
    summary.append("1-2) Selected models (CV on training set; test set used only here):")
    selected = {}
    for k, m in enumerate(METHODS):
        for r, rule in enumerate(("min", "1se")):
            i, j = select(cv[m], rule)
            p, lam = DEGREES[i], (0.0 if m == "OLS" else LAMBDAS[j])
            sc = Scaler().fit(design_matrix(x_tr, p), y_tr)
            theta = fitter(m, lam)(sc.transform_X(design_matrix(x_tr, p)), y_tr - sc.y_mean)
            y_pred = sc.transform_X(design_matrix(x_te, p)) @ theta + sc.y_mean
            vals = (k, r, p, lam, mean[m][i, j], se[m][i, j], MSE(y_te, y_pred), R2(y_te, y_pred),
                    int(np.sum(theta != 0)))
            for key, v in zip(rows, vals):
                rows[key].append(v)
            if rule == "min":
                selected[m] = (i, j)
            summary.append(f"   {m:5s} {'lowest CV' if rule == 'min' else 'one-SE   '}: {describe(m, i, j):26s} "
                           f"CV MSE {vals[4]:.5f} +- {vals[5]:.5f}, test MSE {vals[6]:.5f}, R2 {vals[7]:.3f}, "
                           f"nonzero coefficients {vals[8]}/{p}")
    save_table(PART, "selected_models.csv", rows)
    best_m = min(METHODS, key=lambda m: mean[m][selected[m]])
    i, j = selected[best_m]
    within = [m for m in METHODS if mean[m][selected[m]] <= mean[best_m][i, j] + se[best_m][i, j]]
    summary += [f"   Lowest CV MSE overall: {best_m}, {describe(best_m, i, j)}; methods within one SE of it: "
                + ", ".join(within),
                "   (method index in selected_models.csv: 0 OLS, 1 Ridge, 2 Lasso; rule 0 = lowest CV, 1 = one-SE)"]

    # robustness of the choice: plain 5-fold and 10-fold without repetition
    for k_alt in (5, 10):
        folds_alt = list(RepeatedKFold(n_splits=k_alt, n_repeats=1, random_state=SEED + 1).split(x_tr))
        cv_alt = cross_validate(x_tr, y_tr, folds_alt)
        summary.append(f"   Check with a single {k_alt}-fold split: "
                       + "; ".join(f"{m} {describe(m, *select(cv_alt[m]))} "
                                   f"(CV {cv_alt[m].mean(axis=2)[select(cv_alt[m])]:.5f})" for m in METHODS))
    summary.append("")

    # ------------------------------------------------------------------------
    # 3) Bias-variance with the true f as reference
    # ------------------------------------------------------------------------
    f_te = runge(x_te)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    boot_rng = lambda: np.random.default_rng(SEED)        # identical resamples for every call
    _, b_ols, v_ols = bias_variance(f_te, bootstrap_predictions(x_tr, y_tr, x_te, DEG_BV, N_BOOT, boot_rng()))
    cols = {"lambda": LAMBDAS}
    summary.append(f"3a) Bias-variance vs lambda, degree {DEG_BV}, {N_BOOT} bootstrap samples, reference f(x_test):")
    summary.append(f"   OLS: bias^2 {b_ols:.5f}, variance {v_ols:.5f}")
    for m, ls in (("Ridge", "-"), ("Lasso", "--")):
        # Lasso: only lambda values whose CV fits converged at this degree (see module docstring)
        use = np.isfinite(mean[m][DEG_BV - 1])
        res = np.full((len(LAMBDAS), 3), np.nan)
        for j in np.flatnonzero(use):
            res[j] = bias_variance(f_te, bootstrap_predictions(x_tr, y_tr, x_te, DEG_BV, N_BOOT, boot_rng(),
                                                               fit=fitter(m, LAMBDAS[j])))
        axes[0].loglog(LAMBDAS, res[:, 1], ls, c="C1", label=f"{m}: bias$^2$")
        axes[0].loglog(LAMBDAS, res[:, 2], ls, c="C2", label=f"{m}: variance")
        axes[0].loglog(LAMBDAS, res[:, 0], ls, c=COLORS[m], lw=2, label=f"{m}: MSE w.r.t. f")
        cols.update({f"{m.lower()}_mse_f": res[:, 0], f"{m.lower()}_bias2": res[:, 1],
                     f"{m.lower()}_variance": res[:, 2]})
        j0, jb = np.flatnonzero(use)[0], np.nanargmin(res[:, 0])
        summary.append(f"   {m}: bias^2 {res[j0, 1]:.5f} -> {res[-1, 1]:.5f}, variance {res[j0, 2]:.5f} -> "
                       f"{res[-1, 2]:.5f} from lambda {LAMBDAS[j0]:.0e} to {LAMBDAS[-1]:.0e}; lowest MSE w.r.t. f "
                       f"{res[jb, 0]:.5f} at lambda {LAMBDAS[jb]:.1e} (bias^2 {res[jb, 1]:.5f}, variance {res[jb, 2]:.5f})")
    axes[0].axhline(b_ols, c="C1", ls=":", lw=1, label="OLS: bias$^2$")
    axes[0].axhline(v_ols, c="C2", ls=":", lw=1, label="OLS: variance")
    axes[0].set(xlabel=r"$\lambda$", ylabel="Contribution to MSE w.r.t. f",
                title=f"Degree {DEG_BV}: what the penalty trades")
    axes[0].legend(fontsize=7, ncol=2)
    save_table(PART, f"bias_variance_vs_lambda_degree{DEG_BV}.csv", cols)

    cols = {"degree": DEGREES}
    summary.append(f"3b) Bias-variance vs degree (Ridge and Lasso at their CV-optimal lambda per degree):")
    for m in METHODS:
        res = np.array([bias_variance(f_te, bootstrap_predictions(
            x_tr, y_tr, x_te, p, N_BOOT, boot_rng(), fit=fitter(m, LAMBDAS[best_lam_per_degree[m][i]])))
            for i, p in enumerate(DEGREES)])
        axes[1].semilogy(DEGREES, res[:, 1], "-", c=COLORS[m], label=f"{m}: bias$^2$")
        axes[1].semilogy(DEGREES, res[:, 2], "--", c=COLORS[m], label=f"{m}: variance")
        cols.update({f"{m.lower()}_bias2": res[:, 1], f"{m.lower()}_variance": res[:, 2]})
        summary.append(f"   {m:5s}: variance at degree 5 / 10 / 15: {res[4, 2]:.5f} / {res[9, 2]:.5f} / "
                       f"{res[14, 2]:.5f}; bias^2: {res[4, 1]:.5f} / {res[9, 1]:.5f} / {res[14, 1]:.5f}")
    axes[1].set(xlabel="Polynomial degree", ylabel="Contribution to MSE w.r.t. f",
                title=r"Bias$^2$ (solid) and variance (dashed) vs degree")
    axes[1].legend(fontsize=7, ncol=3)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(media_path(PART, "bias_variance.png"), dpi=150)
    save_table(PART, "bias_variance_vs_degree.csv", cols)

    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))
    plt.show()