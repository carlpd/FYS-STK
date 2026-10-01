"""
FYS-STK3155/4155, Project 1, part e): own gradient descent with analytical gradients and
automatic differentiation, for OLS and Ridge.

1) Gradient check: analytical gradient vs automatic differentiation (JAX or Autograd, float64
   asserted), and vs central finite differences (numerical differentiation) for contrast.
2) Plain GD vs the closed-form solutions of parts a) and b), degree DEG_MAIN, with both
   gradients and two learning rates (1/h_max and eta_opt = 2/(h_max + h_min)):
   parameter error, cost gap C(theta_k) - C* and test MSE as functions of the iteration,
   compared with the bound ||theta_k - theta*|| / ||theta*|| <= rho^k.
3) Learning rate: iterations to tolerance vs eta for OLS and Ridge, measured and predicted from
   rho(eta) = max_i |1 - eta h_i|; divergence for eta > 2/h_max
   (h_i = eigenvalues of the Hessian (2/n) X^T X + 2 lam I).
4) Condition number and iterations vs polynomial degree (with the unscaled design matrix for
   comparison), and at a high degree: parameter error vs test MSE along the GD path.

Cost: C(theta) = (1/n)||y - X theta||^2 + lam ||theta||^2 (lam = 0: OLS), as in part b).
Features standardised and y centred with training statistics, as before. GD starts at
theta_0 = 0, so the relative error ||theta_k - theta*|| / ||theta*|| equals ||e_k|| / ||e_0||.

On the cost of automatic differentiation (report, theory section): reverse-mode AD
(jax.grad) evaluates the cost once forwards, storing intermediates, and propagates
derivatives backwards once; the full gradient costs a small constant times one cost
evaluation, independent of the number of parameters p. The central finite differences in
grad_fd need 2p cost evaluations and carry truncation and cancellation errors.

Requires JAX (pip install jax) or Autograd (pip install autograd).
Figures: results/media/e/   Tables and summary: results/output/e/
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

from utils import (runge_data, design_matrix, Scaler, MSE, ridge_fit, ridge_cost, ridge_grad,
                   ridge_hessian, autodiff_backend, gradient_descent, optimise,
                   media_path, save_table, save_text)

PART = "e"
SEED = 2026
N, SIGMA, TEST_SIZE = 100, 0.1, 0.3
DEG_MAIN = 5                  # degree for the convergence and learning-rate studies
DEG_HIGH = 12                 # ill-conditioned degree for the parameter-vs-prediction study
LAM_MAIN = 1e-3               # Ridge penalty for the same studies
TOL = 1e-6                    # "converged": ||theta_k - theta*|| / ||theta*|| < TOL
MAX_ITER = 100_000            # iteration budget
N_HIGH = 100_000              # iterations in the high-degree study
LR_SCAN = np.concatenate([np.linspace(0.05, 0.95, 10), [0.98, 0.99, 0.995, 0.999, 1.001, 1.01, 1.05]])

# ----------------------------------------------------------------------------
# Automatic differentiation: the cost function written as an ordinary Python function
# ----------------------------------------------------------------------------
anp, ad_grad_transform, AD_NAME = autodiff_backend()


def cost_ad(theta, X, y, lam):
    """Same cost as utils.ridge_cost, written with the autodiff library's numpy."""
    return anp.mean((y - X @ theta) ** 2) + lam * anp.sum(theta ** 2)


_grad_ad = ad_grad_transform(cost_ad)              # gradient with respect to the first argument


def grad_ad(theta, X, y, lam):
    return np.asarray(_grad_ad(theta, X, y, lam))


def grad_fd(theta, X, y, lam, h=1e-6):
    """Central finite differences (numerical differentiation), for comparison only."""
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h
        g[j] = (ridge_cost(theta + e, X, y, lam) - ridge_cost(theta - e, X, y, lam)) / (2 * h)
    return g


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def prepare(x_tr, x_te, y_tr, degree):
    """Standardised design matrices (training statistics) and centred training targets."""
    sc = Scaler().fit(design_matrix(x_tr, degree), y_tr)
    return (sc.transform_X(design_matrix(x_tr, degree)),
            sc.transform_X(design_matrix(x_te, degree)), y_tr - sc.y_mean, sc.y_mean)


def label(lam):
    """Plot label for a penalty value."""
    return "OLS" if lam == 0 else rf"Ridge, $\lambda$ = {lam:.0e}"


def plain(text):
    """Matplotlib label -> plain text for the summary file."""
    for ch in ("$", "\\", "{", "}"):
        text = text.replace(ch, "")
    return text


def int_ticks(fig):
    for a in fig.axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))


def check_gradients(x_tr, x_te, y_tr, rng, summary):
    """Section 1: analytical vs AD vs finite differences."""
    g_test = grad_ad(np.zeros(3), np.ones((4, 3)), np.ones(4), 0.0)
    assert g_test.dtype == np.float64, f"AD gradient is {g_test.dtype}, not float64"
    summary += [f"1) Gradient check ({AD_NAME} gradient dtype: {g_test.dtype}); relative difference "
                "||g - g_analytical|| / ||g_analytical|| at random theta:"]
    rows = {"degree": [], "lambda": [], "rel_diff_autodiff": [], "rel_diff_finite_diff": []}
    for p in (5, 10, 15):
        X_tr, _, yc, _ = prepare(x_tr, x_te, y_tr, p)
        for lam in (0.0, LAM_MAIN):
            theta = rng.standard_normal(p)
            g_an = ridge_grad(theta, X_tr, yc, lam)
            d_ad = np.linalg.norm(grad_ad(theta, X_tr, yc, lam) - g_an) / np.linalg.norm(g_an)
            d_fd = np.linalg.norm(grad_fd(theta, X_tr, yc, lam) - g_an) / np.linalg.norm(g_an)
            rows["degree"].append(p)
            rows["lambda"].append(lam)
            rows["rel_diff_autodiff"].append(d_ad)
            rows["rel_diff_finite_diff"].append(d_fd)
            summary.append(f"   degree {p:2d}, lambda {lam:.0e}: {AD_NAME} {d_ad:.1e}, "
                           f"finite differences (h = 1e-6) {d_fd:.1e}")
    save_table(PART, "gradient_check.csv", rows)
    summary.append("")


def gd_vs_closed_form(x_tr, x_te, y_tr, y_te, summary):
    """Section 2: GD vs closed form; parameter error, cost gap and test MSE vs iteration."""
    X_tr, X_te, yc, y_mean = prepare(x_tr, x_te, y_tr, DEG_MAIN)
    summary.append(f"2) Gradient descent vs closed form, degree {DEG_MAIN}, theta_0 = 0, "
                   f"tolerance {TOL:.0e} on the relative parameter error:")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rows = {"lambda": [], "eta": [], "gradient": [], "iterations": [], "predicted_bound": [],
            "test_mse_gd": [], "test_mse_closed_form": []}
    for ax_row, lam in zip(axes, (0.0, LAM_MAIN)):
        h = np.linalg.eigvalsh(ridge_hessian(X_tr, lam))
        kappa = h.max() / h.min()
        theta_star = ridge_fit(X_tr, yc, lam)
        C_star = ridge_cost(theta_star, X_tr, yc, lam)
        mse_cf = MSE(y_te, X_te @ theta_star + y_mean)
        ref = np.linalg.norm(theta_star)
        monitor = lambda t: (np.linalg.norm(t - theta_star) / ref,
                             ridge_cost(t, X_tr, yc, lam) - C_star,
                             MSE(y_te, X_te @ t + y_mean))
        etas = {r"$\eta = 1/h_{max}$": (1.0 / h.max(), 1 - 1 / kappa),
                r"$\eta_{opt} = 2/(h_{max}+h_{min})$": (2.0 / (h.max() + h.min()), (kappa - 1) / (kappa + 1))}
        grads = {"analytical": lambda t: ridge_grad(t, X_tr, yc, lam),
                 AD_NAME: lambda t: grad_ad(t, X_tr, yc, lam)}
        thetas = {}
        for (eta_name, (eta, rho)), color in zip(etas.items(), ("C0", "C2")):
            for grad_name, grad in grads.items():
                th, hist, it = optimise(grad, np.zeros(DEG_MAIN), eta, MAX_ITER, "gd",
                                        theta_ref=theta_star, tol=TOL, monitor=monitor)
                thetas[(eta_name, grad_name)] = th
                k = np.arange(len(hist))
                ls, lw = ("-", 2.5) if grad_name == "analytical" else ("--", 1.2)
                name = f"{eta_name}, {grad_name}"
                ax_row[0].semilogy(k, hist[:, 0], ls, c=color, lw=lw, label=name)
                ax_row[1].semilogy(k, np.maximum(hist[:, 1], 1e-18), ls, c=color, lw=lw, label=name)
                if color == "C0":     # eta_opt is left out: its stiff mode oscillates (factor ~ -1)
                    ax_row[2].semilogx(k + 1, hist[:, 2], ls, c=color, lw=lw, label=name)
                bound = np.log(TOL) / np.log(rho)
                rows["lambda"].append(lam)
                rows["eta"].append(eta)
                rows["gradient"].append(0 if grad_name == "analytical" else 1)
                rows["iterations"].append(it)
                rows["predicted_bound"].append(bound)
                rows["test_mse_gd"].append(hist[-1, 2])
                rows["test_mse_closed_form"].append(mse_cf)
                summary.append(f"   {plain(label(lam)):18s} {plain(eta_name):26s} {grad_name:10s}: "
                               f"{it:6d} iterations (bound ln(tol)/ln(rho) = {bound:.0f}), "
                               f"test MSE {hist[-1, 2]:.6f}")
            k = np.arange(int(np.log(TOL) / np.log(rho)) + 1)
            ax_row[0].semilogy(k, rho ** k, ":", c=color, lw=1, label=rf"bound $\rho^k$, $\rho$ = {rho:.4f}")
        diff = max(np.max(np.abs(thetas[(e, "analytical")] - thetas[(e, AD_NAME)])) for e in etas)
        summary.append(f"   {plain(label(lam))}: kappa = {kappa:.1f}, closed-form test MSE {mse_cf:.6f}, "
                       f"max |theta_analytical - theta_{AD_NAME}| = {diff:.1e}")

        ax_row[0].axhline(TOL, ls="-.", c="k", lw=0.8, label=f"Tolerance {TOL:.0e}")
        ax_row[0].set(xlabel="Iteration k", ylabel=r"$\|\theta_k - \theta^*\| \,/\, \|\theta^*\|$",
                      title=f"{label(lam)}: parameter error, $\\kappa$ = {kappa:.0f}")
        ax_row[1].set(xlabel="Iteration k", ylabel=r"$C(\theta_k) - C^*$",
                      title=f"{label(lam)}: cost gap")
        ax_row[2].axhline(mse_cf, c="k", ls="--", lw=1, label="Closed form")
        ax_row[2].set(xlabel="Iteration k + 1", ylabel="Test MSE",
                      title=f"{label(lam)}: test MSE ($\\eta = 1/h_{{max}}$)")
        ax_row[0].legend(fontsize=7)
        ax_row[2].legend(fontsize=7)
    fig.suptitle(f"Gradient descent converging to the closed-form solution, degree {DEG_MAIN}")
    fig.tight_layout()
    fig.savefig(media_path(PART, "convergence.png"), dpi=150)
    save_table(PART, "convergence.csv", rows)
    summary += ["   (gradient column in convergence.csv: 0 = analytical, 1 = automatic differentiation)",
                "   eta_opt lies just below 2/h_max, so the stiffest mode is multiplied by about -1 per step:",
                "   the error norm decreases monotonically, but the test MSE oscillates for many iterations.", ""]
    return X_tr, yc


def learning_rate_study(X_tr, yc, summary):
    """Section 3: error curves, and iterations to tolerance vs eta, theory vs measurement."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    h = np.linalg.eigvalsh(ridge_hessian(X_tr, 0.0))
    eta_crit = 2.0 / h.max()
    theta_star = ridge_fit(X_tr, yc, 0.0)
    g = lambda t: ridge_grad(t, X_tr, yc, 0.0)
    for frac in (0.05, 0.25, 0.5, 0.9, 0.99, 1.01, 1.05):
        _, err, _ = gradient_descent(g, np.zeros(DEG_MAIN), frac * eta_crit, 3000, theta_star)
        axes[0].semilogy(err, label=rf"$\eta$ = {frac:.2f} $\cdot 2/h_{{max}}$")
    axes[0].set(xlabel="Iteration k", ylabel=r"$\|\theta_k - \theta^*\| \,/\, \|\theta^*\|$",
                title=f"OLS, degree {DEG_MAIN}: error vs iteration (stopped at 1e10 if diverging)")
    axes[0].legend(fontsize=8)

    cols = {"eta_over_2_div_hmax": LR_SCAN}
    summary.append(f"3) Learning rate, degree {DEG_MAIN}: iterations to tolerance {TOL:.0e} "
                   f"(budget {MAX_ITER:,}) vs eta; rho(eta) = max_i |1 - eta h_i|")
    for lam, c in ((0.0, "C0"), (LAM_MAIN, "C3")):
        h = np.linalg.eigvalsh(ridge_hessian(X_tr, lam))
        eta_crit, eta_opt = 2.0 / h.max(), 2.0 / (h.max() + h.min())
        theta_star = ridge_fit(X_tr, yc, lam)
        g = lambda t: ridge_grad(t, X_tr, yc, lam)
        rho = np.array([np.max(np.abs(1 - f * eta_crit * h)) for f in LR_SCAN])
        predicted = np.where(rho < 1, np.log(TOL) / np.log(np.minimum(rho, 1 - 1e-16)), np.nan)
        measured, diverged = [], []
        for f in LR_SCAN:
            _, err, it = gradient_descent(g, np.zeros(DEG_MAIN), f * eta_crit, MAX_ITER, theta_star, TOL)
            measured.append(it if err[-1] < TOL else np.nan)
            diverged.append(bool(err[-1] > 1))
        measured, diverged = np.array(measured), np.array(diverged)
        axes[1].semilogy(LR_SCAN, predicted, "-", c=c, lw=1.2, label=f"{label(lam)}: bound ln(tol)/ln $\\rho(\\eta)$")
        axes[1].semilogy(LR_SCAN, measured, "o", c=c, mfc="none", label=f"{label(lam)}: measured")
        tag = "ols" if lam == 0 else "ridge"
        cols[f"rho_{tag}"] = rho
        cols[f"predicted_{tag}"] = predicted
        cols[f"measured_{tag}"] = measured
        cols[f"diverged_{tag}"] = diverged
        summary += [f"   {plain(label(lam))}: h_max = {h.max():.4f}, h_min = {h.min():.4g}, "
                    f"2/h_max = {eta_crit:.4f}, eta_opt = {eta_opt:.4f} = {eta_opt / eta_crit:.4f} x 2/h_max",
                    f"      largest scanned eta that converged: {LR_SCAN[np.isfinite(measured)].max():.3f} x 2/h_max; "
                    f"smallest that diverged: {LR_SCAN[diverged].min():.3f} x 2/h_max",
                    f"      measured / bound iterations, min-max over converged eta: "
                    f"{np.nanmin(measured / predicted):.2f}-{np.nanmax(measured / predicted):.2f}"]
    axes[1].axvspan(1.0, LR_SCAN.max() + 0.01, color="gray", alpha=0.15, label=r"$\eta > 2/h_{max}$: diverges")
    axes[1].set(xlabel=r"$\eta \,/\, (2/h_{max})$  ($h_{max}$ of each method)",
                ylabel=f"Iterations to tolerance {TOL:.0e}",
                title="Iterations vs learning rate: measured and predicted")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(media_path(PART, "learning_rate.png"), dpi=150)
    save_table(PART, "learning_rate_scan.csv", cols)
    summary.append("")


def degree_study(x_tr, x_te, y_tr, y_te, summary):
    """Section 4: condition number and iterations vs degree; high-degree GD path."""
    degrees = np.arange(1, 16)
    lams = (0.0, 1e-4, 1e-3, 1e-2)
    kappas = {lam: [] for lam in lams}
    iters = {lam: [] for lam in lams}
    kappa_unscaled = []
    for p in degrees:
        Xp, _, ycp, _ = prepare(x_tr, x_te, y_tr, p)
        h_raw = np.linalg.eigvalsh(ridge_hessian(design_matrix(x_tr, p)))
        kappa_unscaled.append(h_raw.max() / h_raw.min())
        for lam in lams:
            h = np.linalg.eigvalsh(ridge_hessian(Xp, lam))
            kappas[lam].append(h.max() / h.min())
            _, err, it = gradient_descent(lambda t: ridge_grad(t, Xp, ycp, lam), np.zeros(p),
                                          1.0 / h.max(), MAX_ITER, ridge_fit(Xp, ycp, lam), TOL)
            iters[lam].append(it if err[-1] < TOL else np.nan)      # NaN: not converged

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for lam, m in zip(lams, ("o", "s", "^", "d")):
        axes[0].semilogy(degrees, kappas[lam], f"-{m}", ms=4, label=label(lam))
        axes[1].semilogy(degrees, iters[lam], f"-{m}", ms=4, label=label(lam))
    axes[0].semilogy(degrees, kappa_unscaled, "--", c="gray", label="OLS, unscaled design matrix")
    axes[0].set(xlabel="Polynomial degree", ylabel=r"$\kappa = h_{max}/h_{min}$",
                title="Condition number of the Hessian")
    axes[1].axhline(MAX_ITER, c="gray", ls="--", label=f"Iteration budget ({MAX_ITER:,})")
    axes[1].set(xlabel="Polynomial degree", ylabel="Iterations to tolerance",
                title=rf"GD iterations ($\eta = 1/h_{{max}}$, tolerance {TOL:.0e}); gaps = not converged")
    for ax in axes:
        ax.legend(fontsize=8)
    int_ticks(fig)
    fig.tight_layout()
    fig.savefig(media_path(PART, "iterations_vs_degree.png"), dpi=150)

    cols = {"degree": degrees, "kappa_ols_unscaled": kappa_unscaled}
    for lam in lams:
        tag = "ols" if lam == 0 else f"ridge_{lam:.0e}"
        cols[f"kappa_{tag}"] = kappas[lam]
        cols[f"iterations_{tag}"] = iters[lam]
    save_table(PART, "iterations_vs_degree.csv", cols)

    summary.append(f"4) Iterations to tolerance {TOL:.0e} with eta = 1/h_max (budget {MAX_ITER:,}):")
    for lam in lams:
        conv = [f"{p}: {int(i)}" for p, i in zip(degrees, iters[lam]) if np.isfinite(i)]
        not_conv = [str(p) for p, i in zip(degrees, iters[lam]) if not np.isfinite(i)]
        summary.append(f"   {plain(label(lam))}: " + ", ".join(conv)
                       + (f"; not converged for degree {', '.join(not_conv)}" if not_conv else ""))
    summary.append(f"   kappa OLS degree {DEG_MAIN}: standardised {kappas[0.0][DEG_MAIN - 1]:.1e}, "
                   f"unscaled {kappa_unscaled[DEG_MAIN - 1]:.1e}; degree 15: standardised "
                   f"{kappas[0.0][-1]:.1e}, unscaled {kappa_unscaled[-1]:.1e}")

    # High degree: the parameters converge slowly, the predictions do not need to
    Xh, Xh_te, ych, ymh = prepare(x_tr, x_te, y_tr, DEG_HIGH)
    h = np.linalg.eigvalsh(ridge_hessian(Xh))
    theta_star = ridge_fit(Xh, ych, 0.0)
    ref = np.linalg.norm(theta_star)
    mse_cf_te = MSE(y_te, Xh_te @ theta_star + ymh)
    mse_cf_tr = MSE(ych, Xh @ theta_star)
    monitor = lambda t: (np.linalg.norm(t - theta_star) / ref, MSE(y_te, Xh_te @ t + ymh), MSE(ych, Xh @ t))
    _, hist, _ = optimise(lambda t: ridge_grad(t, Xh, ych), np.zeros(DEG_HIGH), 1.0 / h.max(),
                          N_HIGH, "gd", monitor=monitor)
    k = np.arange(1, len(hist) + 1)
    k_best = int(np.argmin(hist[:, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(k, hist[:, 0], c="C0")
    axes[0].set(xlabel="Iteration k + 1", ylabel=r"$\|\theta_k - \theta^*\| \,/\, \|\theta^*\|$",
                title=f"OLS, degree {DEG_HIGH}, $\\kappa$ = {h.max() / h.min():.1e}: parameter error")
    axes[1].semilogx(k, hist[:, 1], c="C3", label="Test MSE (GD)")
    axes[1].semilogx(k, hist[:, 2], c="C0", label="Train MSE (GD)")
    axes[1].axhline(mse_cf_te, c="C3", ls="--", lw=1, label="Test MSE (closed form)")
    axes[1].axhline(mse_cf_tr, c="C0", ls="--", lw=1, label="Train MSE (closed form)")
    axes[1].axvline(k_best + 1, c="gray", ls=":", label=f"Lowest test MSE (k = {k_best})")
    axes[1].set(xlabel="Iteration k + 1", ylabel="MSE", yscale="log",
                title=f"OLS, degree {DEG_HIGH}: predictions along the GD path")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(media_path(PART, "high_degree.png"), dpi=150)
    save_table(PART, f"high_degree_{DEG_HIGH}.csv",
               {"iteration": k - 1, "rel_param_error": hist[:, 0], "test_mse": hist[:, 1],
                "train_mse": hist[:, 2]})
    summary += [f"   High degree {DEG_HIGH} (OLS, eta = 1/h_max, {N_HIGH:,} iterations): "
                f"kappa = {h.max() / h.min():.1e}, final parameter error {hist[-1, 0]:.1e}",
                f"      test MSE: closed form {mse_cf_te:.5f}, GD final {hist[-1, 1]:.5f}, "
                f"GD lowest {hist[k_best, 1]:.5f} at k = {k_best}",
                f"      train MSE: closed form {mse_cf_tr:.5f}, GD final {hist[-1, 2]:.5f}"]


if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    rng = np.random.default_rng(SEED)
    summary = ["Part e) Gradient descent with analytical gradients and automatic differentiation",
               f"Data: n = {N}, sigma = {SIGMA}, seed = {SEED}, test_size = {TEST_SIZE} "
               f"(n_train = {len(x_tr)})",
               "Cost: (1/n)||y - X theta||^2 + lambda||theta||^2; Hessian (2/n)X^T X + 2 lambda I",
               f"Automatic differentiation: {AD_NAME}", ""]

    check_gradients(x_tr, x_te, y_tr, rng, summary)
    X_main, yc_main = gd_vs_closed_form(x_tr, x_te, y_tr, y_te, summary)
    learning_rate_study(X_main, yc_main, summary)
    degree_study(x_tr, x_te, y_tr, y_te, summary)

    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))
    plt.show()