"""
FYS-STK3155/4155, Project 1, part b): Ridge regression for the Runge function.

Shared functions (data, scaling, metrics, ridge_fit) are in utils.py.
Figures: results/media/b/   Tables and summary: results/output/b/

Cost function convention (same as in part e):
    C(theta) = (1/n) ||y - X theta||^2 + lam ||theta||^2
    => theta = (X^T X + n*lam*I)^{-1} X^T y
    Equivalent to sklearn Ridge(alpha = n*lam, fit_intercept=False).

The intercept is not penalised: it is handled by centring y and X
(using training statistics only), exactly as in part a).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from utils import (runge_data, design_matrix, Scaler, MSE, R2, ridge_fit,
                   media_path, save_table, save_text)

PART = "b"


# ----------------------------------------------------------------------------
# Ridge analysis
# ----------------------------------------------------------------------------
def ridge_analysis(x_tr, x_te, y_tr, y_te, lams, degrees):
    """Train/test MSE and R2 on a (lambda x degree) grid, plus parameters.
    
    LLM was used to generate code."""
    shape = (len(lams), len(degrees))
    res = {"mse_tr": np.zeros(shape), "mse_te": np.zeros(shape),
           "r2_tr": np.zeros(shape), "r2_te": np.zeros(shape),
           "theta": {}}
    for i, lam in enumerate(lams):
        for j, p in enumerate(degrees):
            X_tr, X_te = design_matrix(x_tr, p), design_matrix(x_te, p)
            sc = Scaler().fit(X_tr, y_tr)
            X_tr_s, X_te_s = sc.transform_X(X_tr), sc.transform_X(X_te)   # training statistics

            theta = ridge_fit(X_tr_s, y_tr - sc.y_mean, lam)
            y_pred_tr = X_tr_s @ theta + sc.y_mean
            y_pred_te = X_te_s @ theta + sc.y_mean

            res["mse_tr"][i, j] = MSE(y_tr, y_pred_tr)
            res["mse_te"][i, j] = MSE(y_te, y_pred_te)
            res["r2_tr"][i, j] = R2(y_tr, y_pred_tr)
            res["r2_te"][i, j] = R2(y_te, y_pred_te)
            res["theta"][(lam, p)] = theta
    return res


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    n, sigma = 100, 0.1
    degrees = np.arange(1, 16)
    lams = np.logspace(-8, 1, 10)

    x, y = runge_data(n=n, noise=sigma)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=2026)
    n_tr = len(y_tr)

    res = ridge_analysis(x_tr, x_te, y_tr, y_te, lams, degrees)
    ols = ridge_analysis(x_tr, x_te, y_tr, y_te, [0.0], degrees)   # OLS reference

    # 1) Heatmap of test MSE over (degree, lambda)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.log10(res["mse_te"]), origin="lower", aspect="auto", cmap="viridis",
                   extent=[degrees[0] - 0.5, degrees[-1] + 0.5,
                           np.log10(lams[0]) - 0.5, np.log10(lams[-1]) + 0.5])
    i_best, j_best = np.unravel_index(np.argmin(res["mse_te"]), res["mse_te"].shape)
    ax.plot(degrees[j_best], np.log10(lams[i_best]), "r*", ms=14,
            label=f"min: degree {degrees[j_best]}, $\\lambda$={lams[i_best]:.0e}")
    ax.set(xlabel="Polynomial degree", ylabel=r"$\log_{10}\lambda$",
           title=r"Ridge: $\log_{10}$(test MSE)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ MSE")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(media_path(PART, "heatmap_mse.png"), dpi=150)

    # 2) Test and train MSE vs degree for selected lambdas, with OLS
    sel = [1e-6, 1e-4, 1e-2, 1e0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for lam in sel:
        i = np.argmin(np.abs(lams - lam))
        ax[0].plot(degrees, res["mse_tr"][i], "o-", ms=3, label=rf"$\lambda$={lams[i]:.0e}")
        ax[1].plot(degrees, res["mse_te"][i], "o-", ms=3, label=rf"$\lambda$={lams[i]:.0e}")
    ax[0].plot(degrees, ols["mse_tr"][0], "k--o", ms=3, label="OLS")
    ax[1].plot(degrees, ols["mse_te"][0], "k--o", ms=3, label="OLS")
    for a, t in zip(ax, ("Training", "Test")):
        a.axhline(sigma**2, ls=":", c="gray", label=r"$\sigma^2$")
        a.set(xlabel="Polynomial degree", yscale="log", title=f"{t} MSE")
        a.legend(fontsize=8)
    ax[0].set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(media_path(PART, "mse_vs_degree.png"), dpi=150)

    # 3) R2 vs degree for selected lambdas
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(degrees, ols["r2_te"][0], "k--o", ms=3, label="OLS")
    for lam in sel:
        i = np.argmin(np.abs(lams - lam))
        ax.plot(degrees, res["r2_te"][i], "o-", ms=3, label=rf"$\lambda$={lams[i]:.0e}")
    ax.set(xlabel="Polynomial degree", ylabel=r"Test $R^2$", ylim=(-0.5, 1),
           title=r"Ridge: test $R^2$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(media_path(PART, "r2_vs_degree.png"), dpi=150)

    # 4) MSE vs lambda for a fixed high degree
    p_fix = 15
    j = p_fix - 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(lams, res["mse_tr"][:, j], "o-", label="Train")
    ax.semilogx(lams, res["mse_te"][:, j], "s-", label="Test")
    ax.axhline(ols["mse_te"][0, j], ls="--", c="k", label="OLS test")
    ax.axhline(sigma**2, ls=":", c="gray", label=r"$\sigma^2$")
    ax.set(xlabel=r"$\lambda$", ylabel="MSE", yscale="log",
           title=f"Ridge, degree {p_fix}: MSE vs $\\lambda$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(media_path(PART, "mse_vs_lambda.png"), dpi=150)

    # 5) Parameters theta vs lambda (shrinkage) for fixed degree
    lams_fine = np.logspace(-8, 1, 60)
    X_tr = design_matrix(x_tr, p_fix)
    sc = Scaler().fit(X_tr, y_tr)
    X_tr_s = sc.transform_X(X_tr)
    thetas = np.array([ridge_fit(X_tr_s, y_tr - sc.y_mean, lam) for lam in lams_fine])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(p_fix):
        ax.semilogx(lams_fine, thetas[:, k], label=rf"$\theta_{{{k+1}}}$" if k < 6 else None)
    ax.set(xlabel=r"$\lambda$", ylabel=r"$\theta_j$ (standardised features)", yscale="symlog",
           title=f"Ridge coefficients vs $\\lambda$ (degree {p_fix})")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(media_path(PART, "theta_vs_lambda.png"), dpi=150)

    # 6) SVD shrinkage factors d_j^2 / (d_j^2 + n*lam)
    d = np.linalg.svd(X_tr_s, compute_uv=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lam in (1e-6, 1e-4, 1e-2, 1e0):
        f = d**2 / (d**2 + n_tr * lam)
        ax.semilogx(d, f, "o-", ms=4, label=rf"$\lambda$={lam:.0e}, dof={f.sum():.1f}")
    ax.set(xlabel=r"Singular value $d_j$", ylabel=r"$d_j^2/(d_j^2+n\lambda)$",
           title=f"Ridge shrinkage of singular-value modes (degree {p_fix})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(media_path(PART, "shrinkage.png"), dpi=150)

    # 7) Printed summary
    print(f"Best test MSE: {res['mse_te'][i_best, j_best]:.4e} "
          f"at degree {degrees[j_best]}, lambda = {lams[i_best]:.0e}")
    print(f"Best OLS test MSE: {ols['mse_te'][0].min():.4e} "
          f"at degree {degrees[np.argmin(ols['mse_te'][0])]}")
    print(f"OLS test MSE at degree {p_fix}: {ols['mse_te'][0, j]:.4e}\n")

    print(f"Singular values of standardised X (degree {p_fix}):")
    print("  ", np.array2string(d, precision=2))
    print(f"\n{'lambda':>8}  {'eff. dof':>8}  {'cond(XtX + n lam I)':>20}")
    print(f"{'0':>8}  {p_fix:8.2f}  {np.linalg.cond(X_tr_s.T @ X_tr_s):20.3e}")
    for lam in (1e-6, 1e-4, 1e-2, 1e0):
        dof = np.sum(d**2 / (d**2 + n_tr * lam))
        c = np.linalg.cond(X_tr_s.T @ X_tr_s + n_tr * lam * np.eye(p_fix))
        print(f"{lam:8.0e}  {dof:8.2f}  {c:20.3e}")

    # 8) Check against Scikit-Learn (alpha = n * lambda)
    lam = 1e-3
    th_own = ridge_fit(X_tr_s, y_tr - sc.y_mean, lam)
    th_sk = Ridge(alpha=n_tr * lam, fit_intercept=False).fit(X_tr_s, y_tr - sc.y_mean).coef_
    sk_diff = np.max(np.abs(th_own - th_sk))
    print(f"\nmax |theta_own - theta_sklearn| (alpha = n*lambda): {sk_diff:.2e}")

    print("OLS", ols["mse_te"][0])
    # 9) Save tables and summary to results/output/b/
    grid_cols = lambda key: {"log10_lambda": np.log10(lams),
                             **{f"deg_{p}": res[key][:, k] for k, p in enumerate(degrees)}}
    save_table(PART, "mse_test_grid.csv", grid_cols("mse_te"))
    save_table(PART, "mse_train_grid.csv", grid_cols("mse_tr"))
    save_table(PART, "r2_test_grid.csv", grid_cols("r2_te"))
    save_table(PART, "ols_scores_vs_degree.csv",
               {"degree": degrees, "mse_train": ols["mse_tr"][0], "mse_test": ols["mse_te"][0],
                "r2_train": ols["r2_tr"][0], "r2_test": ols["r2_te"][0]})
    save_table(PART, f"theta_vs_lambda_deg{p_fix}.csv",
               {"lambda": lams_fine, **{f"theta_{k+1}": thetas[:, k] for k in range(p_fix)}})

    shrink_lams = (1e-6, 1e-4, 1e-2, 1e0)
    save_table(PART, f"shrinkage_deg{p_fix}.csv",
               {"singular_value": d,
                **{f"lambda_{lam:.0e}": d**2 / (d**2 + n_tr * lam) for lam in shrink_lams}})
    save_table(PART, f"dof_cond_deg{p_fix}.csv",
               {"lambda": [0.0, *shrink_lams],
                "eff_dof": [p_fix, *[np.sum(d**2 / (d**2 + n_tr * l)) for l in shrink_lams]],
                "cond_XtX_plus_nlamI": [np.linalg.cond(X_tr_s.T @ X_tr_s + n_tr * l * np.eye(p_fix))
                                        for l in (0.0, *shrink_lams)]})

    save_text(PART, "summary.txt", [
        "Part b) Ridge regression for the Runge function",
        f"Parameters: n = {n}, sigma = {sigma}, degrees 1-{degrees[-1]}, "
        f"lambda in [{lams[0]:.0e}, {lams[-1]:.0e}] (log grid, {len(lams)} values), "
        "test_size = 0.3, seed = 2026",
        "Cost: (1/n)||y - X theta||^2 + lambda ||theta||^2  (sklearn alpha = n*lambda)",
        "Scaling: X standardised and y centred with training statistics",
        f"Best Ridge test MSE: {res['mse_te'][i_best, j_best]:.4e} "
        f"at degree {degrees[j_best]}, lambda = {lams[i_best]:.0e}",
        f"Best OLS test MSE: {ols['mse_te'][0].min():.4e} "
        f"at degree {degrees[np.argmin(ols['mse_te'][0])]}",
        f"OLS test MSE at degree {p_fix}: {ols['mse_te'][0, j]:.4e}",
        f"Scikit-Learn check (lambda = 1e-3): max |theta_own - theta_sklearn| = {sk_diff:.2e}",
    ])

    plt.show()