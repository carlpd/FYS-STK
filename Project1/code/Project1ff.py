"""
FYS-STK3155/4155, Project 1, part f): gradient descent with momentum and adaptive learning
rates (AdaGrad, RMSprop, Adam) for OLS and Ridge.

All methods use utils.optimise (the same code as parts e and g) with the analytical gradient;
the update rules do not care where the gradient comes from.

1) Learning-rate scan: for every method, iterations needed to reach a relative parameter error
   ||theta_k - theta*|| / ||theta*|| < TOL with respect to the closed-form solution, and the
   error left after the iteration budget, as functions of the (initial) learning rate eta.
   Reported per eta: the first iteration with error < TOL, and the largest error after that
   (below TOL = the method also stays converged).
   Sensitivity = the range of eta for which the method reaches, and stays below, TOL.
   Theory lines: plain GD converges for eta < 2/h_max, momentum (heavy ball,
   v <- gamma v + eta g) for eta < 2(1 + gamma)/h_max.
2) Convergence curves for each method at its best learning rate from the scan, together with
   plain GD at eta = 1/h_max (part e) and momentum with the optimal heavy-ball parameters
   eta* = 4/(sqrt(h_max) + sqrt(h_min))^2, gamma* = ((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^2,
   whose rate (sqrt(kappa) - 1)/(sqrt(kappa) + 1) replaces 1 - 1/kappa of plain GD.

Cost: C(theta) = (1/n)||y - X theta||^2 + lam ||theta||^2 (lam = 0: OLS), as in parts b) and e).
Features standardised and y centred with training statistics; theta_0 = 0.
Figures: results/media/f/   Tables and summary: results/output/f/
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (runge_data, design_matrix, Scaler, MSE, ridge_fit, ridge_grad, ridge_hessian,
                   optimise, media_path, save_table, save_text)

PART = "f"
SEED = 2026
N, SIGMA, TEST_SIZE = 100, 0.1, 0.3
DEGREE = 5                    # same degree as the convergence study in part e)
LAM_MAIN = 1e-3               # Ridge penalty
TOL = 1e-6                    # target relative parameter error, as in part e)
N_ITER = 20_000               # iteration budget per run
GAMMA = 0.9                   # momentum parameter in the scan
ETAS = np.logspace(-4, 1, 21)
ENVELOPE = 0.2                # adaptive methods are plotted as max over the last 20 % of iterations

# label -> (method name in utils.optimise, extra parameters)
METHODS = {"GD": ("gd", {}),
           f"Momentum ($\\gamma$ = {GAMMA})": ("momentum", {"gamma": GAMMA}),
           "AdaGrad": ("adagrad", {}),
           "RMSprop": ("rmsprop", {}),
           "Adam": ("adam", {})}
MARKERS = ("o", "s", "^", "v", "d")


def prepare(x_tr, x_te, y_tr, degree):
    """Standardised design matrices (training statistics) and centred training targets.
    
    LLM was used to generate code."""
    sc = Scaler().fit(design_matrix(x_tr, degree), y_tr)
    return (sc.transform_X(design_matrix(x_tr, degree)),
            sc.transform_X(design_matrix(x_te, degree)), y_tr - sc.y_mean, sc.y_mean)


def label(lam):
    return "OLS" if lam == 0 else rf"Ridge, $\lambda$ = {lam:.0e}"


def plain(text):
    """Matplotlib label -> plain text for the summary file."""
    for ch in ("$", "\\", "{", "}"):
        text = text.replace(ch, "")
    return text


def run(method, params, eta, grad, theta_star, n_iter=N_ITER):
    """One run without early stopping. Returns (theta, error history, k_first, late_max, diverged):
    k_first   first iteration with error < TOL (NaN if never),
    late_max  largest error from k_first to the end of the budget (or over the second half if
              TOL is never reached): below TOL means the method also STAYS converged
              (adaptive methods can reach TOL and bounce back up).
              
    LLM was used to generate code."""
    theta, err, _ = optimise(grad, np.zeros(len(theta_star)), eta, n_iter, method,
                             theta_ref=theta_star, **params)
    diverged = bool(len(err) < n_iter + 1 or not np.isfinite(err[-1]))
    below = np.flatnonzero(err < TOL)
    k_first = below[0] if below.size else np.nan
    late_max = np.inf if diverged else err[int(k_first) if below.size else n_iter // 2:].max()
    return theta, err, k_first, late_max, diverged


def upper_envelope(a, frac):
    """Max over the last frac * k entries at iteration k: a smooth upper envelope on a log axis
    that shows the worst error without drawing every spike."""
    return np.array([a[i - max(1, int(frac * i)) + 1:i + 1].max() for i in range(len(a))])


def eta_range(etas, ok):
    """Smallest and largest eta in the scan where ok is True, as text."""
    if not np.any(ok):
        return "none"
    lo, hi = etas[ok].min(), etas[ok].max()
    return f"[{lo:.1e}, {hi:.1e}] ({ok.sum()} of {len(etas)} scanned values)"


if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    X_tr, X_te, yc, y_mean = prepare(x_tr, x_te, y_tr, DEGREE)
    summary = ["Part f) Momentum, AdaGrad, RMSprop and Adam for OLS and Ridge (utils.optimise)",
               f"Data: n = {N}, sigma = {SIGMA}, seed = {SEED}, test_size = {TEST_SIZE} "
               f"(n_train = {len(x_tr)}), degree {DEGREE}",
               f"Accuracy measure: relative parameter error w.r.t. the closed form, target {TOL:.0e}; "
               f"budget {N_ITER:,} iterations; theta_0 = 0",
               f"Adaptive parameters: momentum gamma = {GAMMA}, RMSprop rho = 0.99, "
               "Adam beta1 = 0.9, beta2 = 0.999, eps = 1e-8", ""]

    fig_scan, ax_scan = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    fig_conv, ax_conv = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for row, lam in enumerate((0.0, LAM_MAIN)):
        h = np.linalg.eigvalsh(ridge_hessian(X_tr, lam))
        h_max, h_min = h.max(), h.min()
        kappa = h_max / h_min
        theta_star = ridge_fit(X_tr, yc, lam)
        mse_cf = MSE(y_te, X_te @ theta_star + y_mean)
        grad = lambda t: ridge_grad(t, X_tr, yc, lam)
        summary += [f"{plain(label(lam))}: h_max = {h_max:.4f}, h_min = {h_min:.4g}, kappa = {kappa:.1f}, "
                    f"closed-form test MSE {mse_cf:.6f}",
                    f"   GD stability limit 2/h_max = {2 / h_max:.4f}; "
                    f"momentum limit 2(1 + gamma)/h_max = {2 * (1 + GAMMA) / h_max:.4f}"]

        # --------------------------------------------------------------------
        # 1) Learning-rate scan
        # --------------------------------------------------------------------
        cols = {"eta": ETAS}
        best = {}
        summary.append("   1) Learning-rate scan (eta from 1e-4 to 1e1):")
        for (name, (method, params)), m in zip(METHODS.items(), MARKERS):
            firsts, lates = [], []
            for eta in ETAS:
                _, _, k_first, late_max, _ = run(method, params, eta, grad, theta_star)
                firsts.append(k_first)
                lates.append(late_max)
            firsts, lates = np.array(firsts), np.array(lates)
            ax_scan[row, 0].loglog(ETAS, firsts, f"-{m}", ms=4, label=name)
            ax_scan[row, 1].loglog(ETAS, np.clip(lates, 1e-17, 1e2), f"-{m}", ms=4, label=name)

            cols[f"first_iteration_below_tol_{method}"] = firsts
            cols[f"max_error_after_first_{method}"] = lates
            stays = lates < TOL
            reached = np.isfinite(firsts)
            if np.any(reached):
                i_best = int(np.nanargmin(np.where(stays, firsts, np.nan))) if np.any(stays) \
                    else int(np.nanargmin(firsts))
                best_txt = (f"fastest {int(firsts[i_best])} it. at eta = {ETAS[i_best]:.1e}"
                            + ("" if stays[i_best] else " (does not stay below)"))
            else:
                i_best = int(np.argmin(lates))
                best_txt = f"lowest late error {lates[i_best]:.1e} at eta = {ETAS[i_best]:.1e}"
            best[name] = ETAS[i_best]
            summary += [f"      {plain(name):22s}: reaches {TOL:.0e} for eta in {eta_range(ETAS, reached)}",
                        f"      {'':22s}  stays below for eta in {eta_range(ETAS, stays)}; {best_txt}"]
        save_table(PART, f"learning_rate_scan_{'ols' if lam == 0 else 'ridge'}.csv", cols)

        for ax in ax_scan[row]:
            ax.axvline(2 / h_max, c="C0", ls=":", lw=1, label=r"GD limit $2/h_{max}$")
            ax.axvline(2 * (1 + GAMMA) / h_max, c="C1", ls=":", lw=1,
                       label=r"Momentum limit $2(1+\gamma)/h_{max}$")
        ax_scan[row, 0].axhline(N_ITER, c="gray", ls="--", lw=0.8)
        ax_scan[row, 0].set(ylabel=f"First iteration with error < {TOL:.0e}",
                            title=f"{label(lam)}: first iteration with error < tol (gaps = never)")
        ax_scan[row, 1].axhline(TOL, c="k", ls="-.", lw=0.8)
        ax_scan[row, 1].set(ylabel="Max error after first reaching tol\n(never reached: 2nd half of budget)",
                            title=f"{label(lam)}: does it stay converged? (top = diverged)")

        # --------------------------------------------------------------------
        # 2) Convergence at the best learning rate of each method
        # --------------------------------------------------------------------
        sq = np.sqrt(kappa)
        eta_hb = 4 / (np.sqrt(h_max) + np.sqrt(h_min)) ** 2
        gamma_hb = ((sq - 1) / (sq + 1)) ** 2
        runs = [(r"GD, $\eta = 1/h_{max}$ (part e)", "gd", {}, 1 / h_max, "k", "--")]
        runs += [(f"{name}, $\\eta$ = {best[name]:.1e}", method, params, best[name], f"C{i}", "-")
                 for i, (name, (method, params)) in enumerate(METHODS.items())]
        runs.append((rf"Momentum, optimal $\eta^*, \gamma^*$ = {gamma_hb:.3f}", "momentum",
                     {"gamma": gamma_hb}, eta_hb, "C1", "--"))
        summary.append(f"   2) Convergence at the best learning rate ({N_ITER:,} iterations):")
        rows = {"run": [], "eta": [], "first_iteration_below_tol": [], "max_error_after_first": [],
                "test_mse": []}
        for i, (name, method, params, eta, c, ls) in enumerate(runs):
            th, err, it, late, _ = run(method, params, eta, grad, theta_star)
            k = np.arange(1, len(err) + 1)
            if method in ("rmsprop", "adam"):
                err_plot, name = upper_envelope(err, ENVELOPE), name + " (upper envelope)"
            else:
                err_plot = err
            ax_conv[row].loglog(k, np.maximum(err_plot, 1e-17), ls, c=c, lw=1.4, label=name)
            mse = MSE(y_te, X_te @ th + y_mean)
            for key, v in zip(rows, (i, eta, it, late, mse)):
                rows[key].append(v)
            summary.append(f"      {plain(name.split(' (upper')[0]):42s}: first below {TOL:.0e} after {'-' if np.isnan(it) else int(it):>6} it., "
                           f"max error afterwards {late:.1e}, test MSE {mse:.6f}")
        rate_gd, rate_hb = 1 - 1 / kappa, (sq - 1) / (sq + 1)
        summary += [f"      predicted rates: GD (eta = 1/h_max) 1 - 1/kappa = {rate_gd:.5f} -> "
                    f"{np.log(TOL) / np.log(rate_gd):.0f} iterations; optimal heavy ball "
                    f"(sqrt(kappa)-1)/(sqrt(kappa)+1) = {rate_hb:.4f} -> {np.log(TOL) / np.log(rate_hb):.0f} "
                    "iterations (asymptotic, up to a constant)", ""]
        save_table(PART, f"convergence_{'ols' if lam == 0 else 'ridge'}.csv", rows)
        ax_conv[row].axhline(TOL, c="k", ls="-.", lw=0.8)
        ax_conv[row].set(xlabel="Iteration k + 1", title=f"{label(lam)}, degree {DEGREE}, $\\kappa$ = {kappa:.0f}")
        ax_conv[row].legend(fontsize=7, loc="lower left")

    for ax in ax_scan[1]:
        ax.set_xlabel(r"(Initial) learning rate $\eta$")
    ax_scan[0, 0].legend(fontsize=7, loc="upper right")
    fig_scan.suptitle(f"Sensitivity to the learning rate, degree {DEGREE}")
    fig_scan.tight_layout()
    fig_scan.savefig(media_path(PART, "learning_rate_scan.png"), dpi=150)

    ax_conv[0].set_ylabel(r"$\|\theta_k - \theta^*\| \,/\, \|\theta^*\|$")
    fig_conv.suptitle("Convergence to the closed-form solution, each method at its best learning rate")
    fig_conv.tight_layout()
    fig_conv.savefig(media_path(PART, "convergence.png"), dpi=150)

    summary.append("(run index in convergence_*.csv follows the order in section 2)")
    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))
    plt.show()