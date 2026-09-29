"""
FYS-STK3155/4155, Project 1, part c): the bias-variance tradeoff and the bootstrap (OLS only).

1) Training and test MSE vs polynomial degree for several n (cf. Hastie et al., Fig. 2.11).
2) Bootstrap bias-variance decomposition of the test error vs polynomial degree.
   - The test set is kept FIXED; only the training set is resampled.
   - The Scaler is refitted on every bootstrap resample (no data leakage).
   - Bias is computed both with the noisy y_test (as in the assignment) and with the
     true f(x_test), which is known here. The difference is approximately sigma^2.
3) The same decomposition for several numbers of data points n.

Expectation values are taken over training sets (approximated by bootstrap resamples),
with the test inputs x_test held fixed.

Figures: results/media/c/   Tables and summary: results/output/c/
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

from utils import (runge, runge_data, design_matrix, Scaler, MSE, ols_fit,
                   bootstrap_predictions, bias_variance,
                   media_path, save_table, save_text)

PART = "c"
SEED = 2026
SIGMA = 0.1
TEST_SIZE = 0.3
DEGREES = np.arange(1, 21)          # up to degree 20: beyond the degree 15 used in a) and b)
N_BOOT = 300
N_VALUES = (50, 100, 500, 1000)


def train_test_mse(x, y, degrees):
    """Training and test MSE vs degree for a single train/test split (OLS)."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    mse_tr, mse_te = [], []
    for p in degrees:
        X_tr, X_te = design_matrix(x_tr, p), design_matrix(x_te, p)
        sc = Scaler().fit(X_tr, y_tr)
        theta = ols_fit(sc.transform_X(X_tr), y_tr - sc.y_mean)
        mse_tr.append(MSE(y_tr, sc.transform_X(X_tr) @ theta + sc.y_mean))
        mse_te.append(MSE(y_te, sc.transform_X(X_te) @ theta + sc.y_mean))
    return np.array(mse_tr), np.array(mse_te)


def bias_variance_analysis(x, y, degrees, n_boot, seed=SEED):
    """Bootstrap bias-variance decomposition vs degree for one data set.

    Computed twice: over all test points, and over the test points that lie inside the
    range of the training inputs ("interior"). Test points outside that range require
    extrapolation of the polynomial and dominate the variance for high degrees.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=seed)
    f_te = runge(x_te)                                  # true function at the test points
    inside = (x_te >= x_tr.min()) & (x_te <= x_tr.max())
    rng = np.random.default_rng(seed)

    keys = ("error", "bias2", "var", "bias2_f", "error_in", "bias2_in", "var_in", "bias2_f_in")
    res = {k: np.zeros(len(degrees)) for k in keys}
    for j, p in enumerate(degrees):
        preds = bootstrap_predictions(x_tr, y_tr, x_te, p, n_boot, rng)
        res["error"][j], res["bias2"][j], res["var"][j] = bias_variance(y_te, preds)
        res["bias2_f"][j] = bias_variance(f_te, preds)[1]
        res["error_in"][j], res["bias2_in"][j], res["var_in"][j] = \
            bias_variance(y_te[inside], preds[inside])
        res["bias2_f_in"][j] = bias_variance(f_te[inside], preds[inside])[1]
    res["noise_test_in"] = np.mean((y_te[inside] - f_te[inside]) ** 2)   # realised noise variance
    res["n_test"], res["n_outside"] = len(x_te), int(np.sum(~inside))
    res["x_outside"] = x_te[~inside]
    return res


if __name__ == "__main__":
    summary = ["Part c) Bias-variance tradeoff with the bootstrap (OLS)",
               f"Parameters: sigma = {SIGMA}, degrees {DEGREES[0]}-{DEGREES[-1]}, "
               f"test_size = {TEST_SIZE}, seed = {SEED}, bootstrap rounds B = {N_BOOT}, "
               f"n in {N_VALUES}",
               "Scaling: refitted on every bootstrap resample; test set fixed", ""]

    # ------------------------------------------------------------------------
    # 1) Training and test MSE vs degree for several n (Hastie Fig. 2.11)
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, N_VALUES):
        x, y = runge_data(n=n, noise=SIGMA, seed=SEED)
        mse_tr, mse_te = train_test_mse(x, y, DEGREES)
        ax.plot(DEGREES, mse_tr, "o-", ms=3, label="Train")
        ax.plot(DEGREES, mse_te, "s-", ms=3, label="Test")
        ax.axhline(SIGMA**2, ls="--", c="gray", label=r"$\sigma^2$")
        ax.set(yscale="log", title=f"n = {n}")
        save_table(PART, f"train_test_mse_n{n}.csv",
                   {"degree": DEGREES, "mse_train": mse_tr, "mse_test": mse_te})
    for ax in axes[1]:
        ax.set_xlabel("Polynomial degree")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE")
    axes[0, 0].legend()
    fig.suptitle("OLS: training and test MSE vs model complexity")
    for a in fig.axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(media_path(PART, "train_test_mse_vs_n.png"), dpi=150)

    # ------------------------------------------------------------------------
    # 2) Bootstrap bias-variance decomposition, n = 100
    # ------------------------------------------------------------------------
    n_main = 100
    x, y = runge_data(n=n_main, noise=SIGMA, seed=SEED)
    res = bias_variance_analysis(x, y, DEGREES, N_BOOT)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sfx, title in ((axes[0], "", f"All {res['n_test']} test points"),
                           (axes[1], "_in", f"Test points inside the training range "
                                            f"({res['n_test'] - res['n_outside']})")):
        ax.plot(DEGREES, res["error" + sfx], "k-o", ms=4, label="Error (test MSE)")
        ax.plot(DEGREES, res["bias2" + sfx], "-s", ms=4, label=r"Bias$^2$ (with $y$)")
        ax.plot(DEGREES, res["var" + sfx], "-^", ms=4, label="Variance")
        ax.plot(DEGREES, res["bias2_f" + sfx], "--", alpha=0.7, label=r"Bias$^2$ (with true $f$)")
        ax.axhline(SIGMA**2, ls=":", c="gray", label=r"$\sigma^2$")
        ax.set(xlabel="Polynomial degree", ylabel="MSE contribution", yscale="log", title=title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Bootstrap bias-variance decomposition (OLS, n = {n_main}, B = {N_BOOT})")
    for a in fig.axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(media_path(PART, "bias_variance.png"), dpi=150)

    save_table(PART, f"bias_variance_n{n_main}.csv",
               {"degree": DEGREES, "error": res["error"], "bias2_y": res["bias2"],
                "variance": res["var"], "bias2_f": res["bias2_f"],
                "error_inside": res["error_in"], "bias2_y_inside": res["bias2_in"],
                "variance_inside": res["var_in"], "bias2_f_inside": res["bias2_f_in"]})

    j_opt, j_opt_in = np.argmin(res["error"]), np.argmin(res["error_in"])
    identity = max(np.max(np.abs(res["error"] - res["bias2"] - res["var"]) / res["error"]),
                   np.max(np.abs(res["error_in"] - res["bias2_in"] - res["var_in"]) / res["error_in"]))
    low = DEGREES <= DEGREES[j_opt_in]                  # well-behaved degrees for the noise check
    noise_gap = res["bias2_in"][low] - res["bias2_f_in"][low]
    summary += [f"n = {n_main}: {res['n_test']} test points, {res['n_outside']} outside the "
                f"training range (x = {np.array2string(res['x_outside'], precision=3)})",
                f"  All test points:    minimum error {res['error'][j_opt]:.4g} at degree {DEGREES[j_opt]}",
                f"  Interior points:    minimum error {res['error_in'][j_opt_in]:.4g} "
                f"at degree {DEGREES[j_opt_in]}",
                f"  Check error = bias2 + var: max relative deviation {identity:.1e}",
                f"  Bias2(y) - Bias2(f), interior points, degrees 1-{DEGREES[j_opt_in]}: "
                f"{noise_gap.min():.4g} to {noise_gap.max():.4g} "
                f"(realised noise mean((y-f)^2) = {res['noise_test_in']:.4g}, sigma^2 = {SIGMA**2:.4g})",
                ""]

    # ------------------------------------------------------------------------
    # 3) Bias-variance decomposition for several n
    # ------------------------------------------------------------------------
    # Decomposition over the interior test points; the error over ALL test points is shown
    # as a thin grey line to indicate the effect of extrapolation.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)   # own y-axis per panel
    summary.append("Dependence on n (optimal degree = minimum bootstrap test error):")
    for ax, n in zip(axes.flat, N_VALUES):
        x, y = runge_data(n=n, noise=SIGMA, seed=SEED)
        r = res if n == n_main else bias_variance_analysis(x, y, DEGREES, N_BOOT)
        ax.plot(DEGREES, r["error"], "-", c="gray", lw=1, label="Error, all test points")
        ax.plot(DEGREES, r["error_in"], "k-o", ms=3, label="Error (interior)")
        ax.plot(DEGREES, r["bias2_in"], "-s", ms=3, label=r"Bias$^2$ (interior)")
        ax.plot(DEGREES, r["var_in"], "-^", ms=3, label="Variance (interior)")
        ax.axhline(SIGMA**2, ls=":", c="gray", label=r"$\sigma^2$")
        ax.set(yscale="log",
               title=f"n = {n} ({r['n_outside']} of {r['n_test']} test points outside)")
        if n != n_main:
            save_table(PART, f"bias_variance_n{n}.csv",
                       {"degree": DEGREES, "error": r["error"], "bias2_y": r["bias2"],
                        "variance": r["var"], "bias2_f": r["bias2_f"],
                        "error_inside": r["error_in"], "bias2_y_inside": r["bias2_in"],
                        "variance_inside": r["var_in"], "bias2_f_inside": r["bias2_f_in"]})
        j, ji = np.argmin(r["error"]), np.argmin(r["error_in"])
        summary.append(f"  n = {n:5d}: all points: degree {DEGREES[j]:2d} (error {r['error'][j]:.4g}); "
                       f"interior: degree {DEGREES[ji]:2d} (error {r['error_in'][ji]:.4g}, "
                       f"bias2 {r['bias2_in'][ji]:.4g}, var {r['var_in'][ji]:.4g})")
    for ax in axes[1]:
        ax.set_xlabel("Polynomial degree")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE contribution")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Bias-variance decomposition vs n (OLS, B = {N_BOOT}, interior test points)")
    for a in fig.axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(media_path(PART, "bias_variance_vs_n.png"), dpi=150)

    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))

    plt.show()