"""
FYS-STK3155/4155, Project 1, part a): OLS for the Runge function.

- Data: f(x) = 1/(1+25x^2) on [-1,1] with Gaussian noise N(0, sigma^2)
- Features: x, x^2, ..., x^p (intercept handled by centring)
- Scaling: standardise X and centre y using TRAINING statistics only
- Solver: pseudoinverse (SVD-based)

Figures: results/media/a/   Tables and summary: results/output/a/
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (runge_data, design_matrix, Scaler, MSE, R2, ols_fit,
                   media_path, save_table, save_text)

PART = "a"


def ols_analysis(x, y, max_degree=15, test_size=0.3, seed=2026):
    """Fit OLS for degrees 1..max_degree; return scores, parameters and condition numbers.

    LLM was used to generate code."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)

    res = {"degrees": np.arange(1, max_degree + 1),
           "mse_tr": [], "mse_te": [], "r2_tr": [], "r2_te": [],
           "theta": [], "cond": []}

    for p in res["degrees"]:
        X_tr, X_te = design_matrix(x_tr, p), design_matrix(x_te, p)

        sc = Scaler().fit(X_tr, y_tr)
        X_tr_s, X_te_s = sc.transform_X(X_tr), sc.transform_X(X_te)   # training statistics

        theta = ols_fit(X_tr_s, y_tr - sc.y_mean)
        y_pred_tr = X_tr_s @ theta + sc.y_mean
        y_pred_te = X_te_s @ theta + sc.y_mean

        res["mse_tr"].append(MSE(y_tr, y_pred_tr))
        res["mse_te"].append(MSE(y_te, y_pred_te))
        res["r2_tr"].append(R2(y_tr, y_pred_tr))
        res["r2_te"].append(R2(y_te, y_pred_te))
        res["theta"].append(theta)
        res["cond"].append(np.linalg.cond(X_tr_s))

    for key in ("mse_tr", "mse_te", "r2_tr", "r2_te", "cond"):
        res[key] = np.array(res[key])
    return res


if __name__ == "__main__":
    n, sigma, max_degree = 100, 0.1, 15
    x, y = runge_data(n=n, noise=sigma)
    res = ols_analysis(x, y, max_degree=max_degree)
    deg = res["degrees"]

    # 1) MSE and R2 vs polynomial degree
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(deg, res["mse_tr"], "o-", label="Train")
    ax[0].plot(deg, res["mse_te"], "s-", label="Test")
    ax[0].axhline(sigma**2, ls="--", c="gray", label=r"$\sigma^2$")
    ax[0].set(xlabel="Polynomial degree", ylabel="MSE", yscale="log",
              title=f"OLS, n={n}, $\\sigma$={sigma}")
    ax[0].legend()
    ax[1].plot(deg, res["r2_tr"], "o-", label="Train")
    ax[1].plot(deg, res["r2_te"], "s-", label="Test")
    ax[1].set(xlabel="Polynomial degree", ylabel=r"$R^2$", title=r"$R^2$ score")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(media_path(PART, "mse_r2.png"), dpi=150)

    # 2) Parameters theta vs degree
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j in range(max_degree):
        ps = deg[deg > j]
        vals = [res["theta"][p - 1][j] for p in ps]
        ax.plot(ps, vals, "o-", ms=3, label=rf"$\theta_{{{j+1}}}$" if j < 8 else None)
    ax.set(xlabel="Polynomial degree", ylabel=r"$\theta_j$ (standardised features)",
           yscale="symlog", title="OLS parameters vs polynomial degree")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(media_path(PART, "theta.png"), dpi=150)

    # 3) Dependence on the number of data points
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n_i in (30, 50, 100, 500, 1000):
        x_i, y_i = runge_data(n=n_i, noise=sigma)
        r = ols_analysis(x_i, y_i, max_degree=max_degree)
        ax.plot(deg, r["mse_te"], "o-", ms=3, label=f"n={n_i}")
    ax.axhline(sigma**2, ls="--", c="gray", label=r"$\sigma^2$")
    ax.set(xlabel="Polynomial degree", ylabel="Test MSE", yscale="log",
           title="Test MSE vs degree for different n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(media_path(PART, "mse_vs_n.png"), dpi=150)

    # 4) Dependence on the noise level
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in (0.0, 0.05, 0.1, 0.2, 0.5):
        x_i, y_i = runge_data(n=n, noise=s)
        r = ols_analysis(x_i, y_i, max_degree=max_degree)
        ax.plot(deg, r["mse_te"], "o-", ms=3, label=rf"$\sigma$={s}")
    ax.set(xlabel="Polynomial degree", ylabel="Test MSE", yscale="log",
           title=f"Test MSE vs degree for different noise levels (n={n})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(media_path(PART, "mse_vs_sigma.png"), dpi=150)

    # 5) Table: condition number and scores (printed and saved)
    print(" deg   cond(X_s)    MSE_train   MSE_test    R2_test")
    for p, c, mtr, mte, r2 in zip(deg, res["cond"], res["mse_tr"], res["mse_te"], res["r2_te"]):
        print(f"{p:4d}  {c:10.3e}  {mtr:10.4e}  {mte:10.4e}  {r2:8.4f}")
    save_table(PART, "scores_vs_degree.csv",
               {"degree": deg, "cond_X": res["cond"], "mse_train": res["mse_tr"],
                "mse_test": res["mse_te"], "r2_train": res["r2_tr"], "r2_test": res["r2_te"]})

    # Parameters theta for every degree (row = degree, column j = theta_j, NaN if j > degree)
    theta_table = np.full((max_degree, max_degree), np.nan)
    for p in deg:
        theta_table[p - 1, :p] = res["theta"][p - 1]
    save_table(PART, "theta_vs_degree.csv",
               {"degree": deg, **{f"theta_{j+1}": theta_table[:, j] for j in range(max_degree)}})

    # 6) Check: OLS predictions are invariant to feature scaling
    p = 10
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=2026)
    X_tr, X_te = design_matrix(x_tr, p), design_matrix(x_te, p)
    sc = Scaler().fit(X_tr, y_tr)
    pred_scaled = sc.transform_X(X_te) @ ols_fit(sc.transform_X(X_tr), y_tr - sc.y_mean) + sc.y_mean
    X1_tr = np.column_stack([np.ones(len(x_tr)), X_tr])     # unscaled, explicit intercept
    X1_te = np.column_stack([np.ones(len(x_te)), X_te])
    pred_raw = X1_te @ ols_fit(X1_tr, y_tr)
    invariance = np.max(np.abs(pred_scaled - pred_raw))
    print(f"\nMax |pred_scaled - pred_unscaled| (degree {p}): {invariance:.2e}")

    save_text(PART, "summary.txt", [
        "Part a) OLS for the Runge function",
        f"Parameters: n = {n}, sigma = {sigma}, max degree = {max_degree}, "
        f"test_size = 0.3, seed = 2026, x ~ U[-1, 1]",
        "Scaling: X standardised and y centred with training statistics; solver: pinv",
        f"Best test MSE: {res['mse_te'].min():.4e} at degree {deg[np.argmin(res['mse_te'])]}",
        f"Scaling invariance check (degree {p}): max |pred_scaled - pred_unscaled| = {invariance:.2e}",
    ])

    plt.show()