"""
FYS-STK3155/4155, Project 1, part h): stochastic (mini-batch) gradient descent for OLS and Ridge.

All runs use utils.sgd, which applies exactly the same update rules (utils.make_update) as the
full-batch optimiser of parts e)-g); only the gradient is computed on a mini-batch:
    grad_B(theta) = (2/|B|) X_B^T (X_B theta - y_B) + 2 lam theta.
Every epoch the training set is reshuffled and split into ceil(n/B) mini-batches; B = n is
full-batch GD.

Accuracy is measured relative to the closed-form solutions of parts a) and b) by the relative
excess training cost (C(theta) - C*) / C* (the quantity being minimised), and reported together
with the relative parameter error and the test MSE. Curves are medians over N_SEEDS shuffles.

1) Mini-batch size: B = 1, 5, 14, 35, 70 (= n), plain SGD with a constant learning rate,
   each B at its best eta from a scan (the stable eta shrinks with B: the curvature of a
   single point, 2||x_i||^2, is much larger than h_max of the full Hessian).
   Accuracy vs epoch (equal number of single-point gradients per epoch) and vs wall time.
2) Learning-rate schedule: constant eta vs eta_t = eta_0 / (1 + t/T) (t = number of updates,
   T = updates in DECAY_EPOCHS epochs), for B = 5. A constant eta leaves a noise floor; the
   decaying schedule satisfies the Robbins-Monro conditions and keeps improving.
3) Update rules of part f) with SGD (B = 5): GD, momentum, AdaGrad, RMSprop and Adam, with
   constant and decaying learning rate, compared with the same rules in full batch.

Cost: C(theta) = (1/n)||y - X theta||^2 + lam ||theta||^2 (lam = 0: OLS); degree DEGREE;
features standardised and y centred with training statistics; theta_0 = 0.
Figures: results/media/h/   Tables and summary: results/output/h/
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (runge_data, design_matrix, Scaler, MSE, ridge_fit, ridge_cost, ridge_grad,
                   ridge_hessian, sgd, media_path, save_table, save_text)

PART = "h"
SEED = 2026
N, SIGMA, TEST_SIZE = 100, 0.1, 0.3
DEGREE = 5                    # same degree as parts e) and f)
LAM_MAIN = 1e-3
N_EPOCHS = 500
N_SEEDS = 5                   # shuffling seeds; curves show the median
BATCH_SIZES = (1, 5, 14, 35, 70)
B_MAIN = 5                    # mini-batch size for sections 2 and 3
DECAY_EPOCHS = 20             # schedule eta_0 / (1 + t/T), T = updates in this many epochs
ETA_FACTORS = np.array([0.01, 0.03, 0.1, 0.3, 1.0])        # x 1/h_max, for GD and momentum
ETA_ADAPTIVE = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])    # AdaGrad, RMSprop, Adam
GAMMA = 0.9

METHODS = {"GD": ("gd", {}), f"Momentum ($\\gamma$ = {GAMMA})": ("momentum", {"gamma": GAMMA}),
           "AdaGrad": ("adagrad", {}), "RMSprop": ("rmsprop", {}), "Adam": ("adam", {})}


def prepare(x_tr, x_te, y_tr, degree):
    sc = Scaler().fit(design_matrix(x_tr, degree), y_tr)
    return (sc.transform_X(design_matrix(x_tr, degree)),
            sc.transform_X(design_matrix(x_te, degree)), y_tr - sc.y_mean, sc.y_mean)


def label(lam):
    return "OLS" if lam == 0 else rf"Ridge, $\lambda$ = {lam:.0e}"


def plain(text):
    for ch in ("$", "\\", "{", "}"):
        text = text.replace(ch, "")
    return text


class Problem:
    """Data, closed-form reference and per-epoch monitor for one penalty value.
    
    LLM was used to generate code."""

    def __init__(self, X, X_te, y, y_te, y_mean, lam):
        self.X, self.y, self.lam, self.n = X, y, lam, len(y)
        self.theta_star = ridge_fit(X, y, lam)
        self.C_star = ridge_cost(self.theta_star, X, y, lam)
        self.h_max = np.linalg.eigvalsh(ridge_hessian(X, lam)).max()
        self.X_te, self.y_te, self.y_mean = X_te, y_te, y_mean
        self.mse_cf = MSE(y_te, X_te @ self.theta_star + y_mean)

    def grad_batch(self, theta, idx):
        return ridge_grad(theta, self.X[idx], self.y[idx], self.lam)

    def monitor(self, theta):
        """(relative excess training cost, relative parameter error, test MSE).
        
        LLM was used to generate code."""
        return ((ridge_cost(theta, self.X, self.y, self.lam) - self.C_star) / self.C_star,
                np.linalg.norm(theta - self.theta_star) / np.linalg.norm(self.theta_star),
                MSE(self.y_te, self.X_te @ theta + self.y_mean))

    def run(self, method, params, eta, batch, decay=False, n_epochs=N_EPOCHS):
        """N_SEEDS runs. Returns (median history (epochs+1, 3), seconds per epoch, diverged).
        
        LLM was used to generate code."""
        updates_per_epoch = int(np.ceil(self.n / batch))
        T = DECAY_EPOCHS * updates_per_epoch
        schedule = (lambda t: eta / (1 + t / T)) if decay else None
        hists, times = [], []
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()
            _, hist, _ = sgd(self.grad_batch, np.zeros(len(self.theta_star)), self.n, eta, n_epochs,
                             batch, method, rng=np.random.default_rng(seed), schedule=schedule,
                             monitor=self.monitor, **params)
            times.append((time.perf_counter() - t0) / max(len(hist) - 1, 1))
            if len(hist) < n_epochs + 1:                    # diverged: stopped early
                return None, np.mean(times), True
            hists.append(hist)
        return np.median(np.array(hists), axis=0), np.mean(times), False

    def best_eta(self, method, params, etas, batch, decay=False):
        """Learning rate from etas with the lowest median final excess cost.
        
        LLM was used to generate code."""
        best = (np.inf, None, None, None)
        for eta in etas:
            hist, t_epoch, diverged = self.run(method, params, eta, batch, decay)
            if not diverged and hist[-1, 0] < best[0]:
                best = (hist[-1, 0], eta, hist, t_epoch)
        return best[1], best[2], best[3]


def floor(a):
    return np.maximum(a, 1e-16)


if __name__ == "__main__":
    x, y = runge_data(n=N, noise=SIGMA, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    X_tr, X_te, yc, y_mean = prepare(x_tr, x_te, y_tr, DEGREE)
    epochs = np.arange(N_EPOCHS + 1)
    summary = ["Part h) Stochastic gradient descent (utils.sgd, same update rules as parts e-g)",
               f"Data: n = {N}, sigma = {SIGMA}, seed = {SEED}, test_size = {TEST_SIZE} "
               f"(n_train = {len(x_tr)}), degree {DEGREE}",
               f"{N_EPOCHS} epochs, medians over {N_SEEDS} shuffling seeds; theta_0 = 0; accuracy = "
               "relative excess training cost (C - C*)/C* w.r.t. the closed form",
               f"Schedule: eta_t = eta_0/(1 + t/T), T = updates in {DECAY_EPOCHS} epochs",
               f"Largest single-point curvature 2||x_i||^2 = {(2 * (X_tr ** 2).sum(axis=1)).max():.1f}", ""]

    fig_b, ax_b = plt.subplots(2, 2, figsize=(13, 9))
    fig_s, ax_s = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig_m, ax_m = plt.subplots(2, 2, figsize=(13, 9), sharey="row")

    for row, lam in enumerate((0.0, LAM_MAIN)):
        P = Problem(X_tr, X_te, yc, y_te, y_mean, lam)
        tag = "ols" if lam == 0 else "ridge"
        summary += [f"{plain(label(lam))}: h_max = {P.h_max:.4f}, closed-form test MSE {P.mse_cf:.6f}"]

        # --------------------------------------------------------------------
        # 1) Mini-batch size, plain SGD, constant learning rate
        # --------------------------------------------------------------------
        summary.append(f"   1) Mini-batch size (plain SGD, constant eta, best of "
                       f"{', '.join(f'{f:g}' for f in ETA_FACTORS)} x 1/h_max):")
        cols = {"epoch": epochs}
        for B, c in zip(BATCH_SIZES, ("C0", "C1", "C2", "C3", "k")):
            eta, hist, t_epoch = P.best_eta("gd", {}, ETA_FACTORS / P.h_max, B)
            name = f"B = {B}" + (" (full batch)" if B == P.n else "")
            ax_b[row, 0].loglog(epochs[1:], floor(hist[1:, 0]), c=c, label=f"{name}, $\\eta$ = {eta * P.h_max:g}/$h_{{max}}$")
            ax_b[row, 1].loglog(epochs[1:] * t_epoch, floor(hist[1:, 0]), c=c, label=name)
            cols[f"excess_cost_B{B}"] = hist[:, 0]
            summary.append(f"      B = {B:2d}: eta = {eta * P.h_max:g}/h_max, {int(np.ceil(P.n / B)):2d} updates/epoch, "
                           f"{1e3 * t_epoch:.3f} ms/epoch; after {N_EPOCHS} epochs: excess cost {hist[-1, 0]:.1e}, "
                           f"parameter error {hist[-1, 1]:.1e}, test MSE {hist[-1, 2]:.5f}")
        save_table(PART, f"batch_size_{tag}.csv", cols)
        ax_b[row, 0].set(xlabel="Epoch (n single-point gradients each)", ylabel=r"$(C(\theta) - C^*)/C^*$",
                         title=f"{label(lam)}: accuracy vs epoch")
        ax_b[row, 1].set(xlabel="Wall time [s] (mean per epoch x epochs)",
                         title=f"{label(lam)}: accuracy vs wall time")
        ax_b[row, 0].legend(fontsize=7)
        ax_b[row, 1].legend(fontsize=7)

        # --------------------------------------------------------------------
        # 2) Learning-rate schedule, B = B_MAIN (OLS and Ridge share the figure)
        # --------------------------------------------------------------------
        summary.append(f"   2) Learning-rate schedule, plain SGD, B = {B_MAIN}:")
        cols = {"epoch": epochs}
        runs = [("constant", 0.1, False, "-"), ("constant", 0.3, False, "-"),
                ("decaying", 0.3, True, "--"), ("decaying", 1.0, True, "--")]
        for (kind, f, decay, ls), c in zip(runs, ("C0", "C1", "C1", "C2")):
            hist, _, diverged = P.run("gd", {}, f / P.h_max, B_MAIN, decay)
            name = f"{kind}, $\\eta_0$ = {f:g}/$h_{{max}}$"
            if diverged:
                summary.append(f"      {plain(name):30s}: diverged")
                continue
            ax_s[row].loglog(epochs[1:], floor(hist[1:, 0]), ls, c=c, label=name)
            cols[f"excess_cost_{kind}_{f:g}"] = hist[:, 0]
            summary.append(f"      {plain(name):30s}: excess cost after 50 / {N_EPOCHS} epochs "
                           f"{hist[50, 0]:.1e} / {hist[-1, 0]:.1e}, parameter error {hist[-1, 1]:.1e}")
        full, _, _ = P.run("gd", {}, 1 / P.h_max, P.n)
        ax_s[row].loglog(epochs[1:], floor(full[1:, 0]), ":", c="k", label=r"Full-batch GD, $\eta = 1/h_{max}$")
        summary.append(f"      {'Full-batch GD, eta = 1/h_max':30s}: excess cost after 50 / {N_EPOCHS} epochs "
                       f"{full[50, 0]:.1e} / {full[-1, 0]:.1e}")
        save_table(PART, f"schedule_{tag}.csv", cols)
        ax_s[row].set(xlabel="Epoch", title=f"{label(lam)}: SGD, B = {B_MAIN}")
        ax_s[row].legend(fontsize=8)

        # --------------------------------------------------------------------
        # 3) Update rules of part f) with SGD, constant and decaying learning rate
        # --------------------------------------------------------------------
        summary.append(f"   3) Update rules with SGD (B = {B_MAIN}) vs full batch (B = {P.n}), "
                       f"best eta for each; final excess cost / test MSE:")
        rows = {"method": [], "eta_sgd_constant": [], "excess_sgd_constant": [], "eta_sgd_decay": [],
                "excess_sgd_decay": [], "eta_full_batch": [], "excess_full_batch": []}
        for i, (name, (method, params)) in enumerate(METHODS.items()):
            etas = ETA_FACTORS / P.h_max if method in ("gd", "momentum") else ETA_ADAPTIVE
            res = {}
            for col, (B, decay) in enumerate(((B_MAIN, False), (B_MAIN, True))):
                eta, hist, _ = P.best_eta(method, params, etas, B, decay)
                res[decay] = (eta, hist)
                ax_m[row, col].loglog(epochs[1:], floor(hist[1:, 0]), c=f"C{i}", label=f"{name}, $\\eta$ = {eta:.1e}")
            eta_fb, hist_fb, _ = P.best_eta(method, params, etas, P.n)
            for key, v in zip(rows, (i, res[False][0], res[False][1][-1, 0], res[True][0], res[True][1][-1, 0],
                                     eta_fb, hist_fb[-1, 0])):
                rows[key].append(v)
            summary.append(f"      {plain(name):22s}: SGD constant {res[False][1][-1, 0]:.1e} / {res[False][1][-1, 2]:.5f}, "
                           f"SGD decaying {res[True][1][-1, 0]:.1e} / {res[True][1][-1, 2]:.5f}, "
                           f"full batch {hist_fb[-1, 0]:.1e} / {hist_fb[-1, 2]:.5f}")
        save_table(PART, f"methods_{tag}.csv", rows)
        for col, kind in enumerate(("constant", "decaying")):
            ax_m[row, col].set(xlabel="Epoch", title=f"{label(lam)}: SGD, B = {B_MAIN}, {kind} $\\eta$")
            ax_m[row, col].legend(fontsize=7)
        ax_m[row, 0].set_ylabel(r"$(C(\theta) - C^*)/C^*$")
        summary.append("")

    fig_b.suptitle(f"Mini-batch size, plain SGD with constant learning rate, degree {DEGREE}")
    fig_b.tight_layout()
    fig_b.savefig(media_path(PART, "batch_size.png"), dpi=150)
    ax_s[0].set_ylabel(r"$(C(\theta) - C^*)/C^*$")
    fig_s.suptitle("Constant vs decaying learning rate")
    fig_s.tight_layout()
    fig_s.savefig(media_path(PART, "schedule.png"), dpi=150)
    fig_m.suptitle("Update rules of part f) with stochastic gradients")
    fig_m.tight_layout()
    fig_m.savefig(media_path(PART, "methods.png"), dpi=150)

    summary.append("(method index in methods_*.csv: 0 GD, 1 momentum, 2 AdaGrad, 3 RMSprop, 4 Adam)")
    save_text(PART, "summary.txt", summary)
    print("\n".join(summary))
    plt.show()