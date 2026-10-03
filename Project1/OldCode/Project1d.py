import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample

xmin=(-1)
xmax=1
seed=2026
n=100
noise=0.1
rng = np.random.default_rng(seed)
x = rng.uniform(xmin, xmax, n).reshape(-1,1)
y = 1.0 / (1.0 + 25.0 * x**2) + np.random.normal(0, noise, x.shape)


def cv_mse(model, X, y, k=5, random_state=2026):

    kfold = KFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    mse_values = []

    for train_idx, test_idx in kfold.split(X):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)

        mse = np.mean((y_test.ravel() - y_pred)**2)

        mse_values.append(mse)

    return np.mean(mse_values), np.std(mse_values)

def polynomial_model(degree, method="OLS", alpha=1.0):

    if method == "OLS":

        return make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )

    elif method == "Ridge":

        return make_pipeline(
            PolynomialFeatures(degree),
            StandardScaler(),
            Ridge(alpha=alpha)
        )

degrees = range(14)

cv_means = []
cv_stds = []

for degree in degrees:

    model = polynomial_model(degree, method="OLS")

    mean_mse, std_mse = cv_mse(
        model,
        x,
        y,
        k=5
    )

    cv_means.append(mean_mse)
    cv_stds.append(std_mse)

best_degree = degrees[np.argmin(cv_means)]

print(f"Optimal degree: {best_degree}")

plt.figure(figsize=(8, 5))

plt.errorbar(
    degrees,
    cv_means,
    yerr=cv_stds,
    fmt="o-",
    capsize=4
)

plt.xlabel("Polynomial degree")
plt.ylabel("CV MSE")
plt.title("5-fold cross-validation for OLS")

plt.grid()
plt.show()

#Ridge
lams=np.array((0, 0.01, 0.1, 0.5))
cv_means = []
cv_stds = []
plt.figure(figsize=(8, 5))
for lam in lams:
    cv_means = []
    cv_stds = []
    for degree in degrees:
        model = polynomial_model(degree, method="Ridge", alpha=lam)

        mean_mse, std_mse = cv_mse(
            model,
            x,
            y,
            k=5
        )

        cv_means.append(mean_mse)
        cv_stds.append(std_mse)

    best_degree = degrees[np.argmin(cv_means)]

    print(f"Optimal degree: {best_degree}")

    plt.errorbar(
        degrees,
        cv_means,
        yerr=cv_stds,
        fmt="o-",
        capsize=4,
        label=fr"$\lambda=${lam}"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("CV MSE")
plt.title("5-fold cross-validation")
plt.legend()

plt.grid()
plt.show()