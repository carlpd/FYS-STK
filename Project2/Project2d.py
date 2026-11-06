import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
import jax
jax.config.update("jax_enable_x64", True)   # float64, so that the checks agree with numpy to 1e-15
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)


from Project2b import(
    Runge,
    create_layers,
    feed_forward,
    ReLU,
    LReLU,
    sigmoid,
    softmax,
    softmax_vec,
    identity,
    sigmoid_prime,
    ReLU_prime,
    LReLU_prime,
    identity_prime,
    backpropagation,
    cost,
    train,
    accuracy,
    predict,
    NeuralNetRegressor,
    runge_split,
    MSE_2
)

if __name__ == "__main__":
    X, y, X_test, y_test, sc = runge_split()

    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    net = make_pipeline(StandardScaler(), NeuralNetRegressor(hidden=(50,20, 10), gamma=0.03, activation="sigmoid", epochs=1000))
    scores = -cross_val_score(net, X, y, cv=cv, scoring="neg_mean_squared_error")
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    param_grid = {"neuralnetregressor__gamma": [0.001, 0.003, 0.01, 0.03, 0.1],
              "neuralnetregressor__lmbd": [0.0, 1e-5, 1e-4, 1e-3, 1e-2]}
    search = GridSearchCV(net, param_grid, cv=cv, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
    search.fit(X, y)
    results = search.cv_results_          # a dict of arrays, one entry per grid point
    k = 5
    gammas = param_grid["neuralnetregressor__gamma"]
    lmbds = param_grid["neuralnetregressor__lmbd"]
    mse = -results["mean_test_score"].reshape(5, 5)   # rows = gamma, columns = lambda
    se = results["std_test_score"].reshape(5, 5) / np.sqrt(k - 1)
    tr = -results["mean_train_score"].reshape(5,5)
    i_best = np.unravel_index(np.argmin(mse), (5,5))
    within = mse <= mse[i_best] + se[i_best]
    cand = [(j, i) for i in range(5) for j in range(5) if within[i, j]]
    j1, i1 = max(cand, key=lambda t: (lmbds[t[0]], -gammas[t[1]]))
    print("minimum:", search.best_params_, f"CV MSE {mse[i_best]:.4f} +/- {se[i_best]:.4f}")
    print(f"1SE choice: gamma = {gammas[i1]}, lambda = {lmbds[j1]}, CV MSE {mse[i1, j1]:.4f} +/- {se[i1, j1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, data, title in zip(ax, [mse, tr], ["CV MSE", "Training MSE"]):
        im = a.imshow(data, cmap="viridis_r")
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=8,
                    color="w" if data[i, j] > data.mean() else "k")
        a.set_xticks(range(5), lmbds)
        a.set_yticks(range(5), gammas)
        a.set_xlabel(r"$\lambda$")
        a.set_ylabel(r"$\gamma$")
        a.set_title(title)
        fig.colorbar(im, ax=a)
    # red boxes = within one SE of the minimum, star = minimum
    for j, i in cand:
        ax[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="r", lw=2))
    ax[0].plot(i_best[1], i_best[0] - 0.3, "r*", ms=10)
    plt.tight_layout()
    plt.show()
        
