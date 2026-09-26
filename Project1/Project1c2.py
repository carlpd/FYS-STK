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

n_bootstrap = 100
degrees = np.arange(1, 21)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=2026
)



test_errors = []
bias_squared = []
variances = []


for degree in degrees:

    y_pred_bootstrap = np.zeros(
        (n_bootstrap, len(x_test))
    )

    for i in range(n_bootstrap):

        x_boot, y_boot = resample(
            x_train,
            y_train,
            replace=True,
            random_state=2026 + i
        )

        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )

        model.fit(x_boot, y_boot)

        y_pred_bootstrap[i, :] = model.predict(x_test).ravel()


    y_pred_mean = np.mean(
        y_pred_bootstrap,
        axis=0
    )


    error = np.mean(
        (y_test.ravel() - y_pred_bootstrap) ** 2
    )


    bias = np.mean(
        (y_test.ravel() - y_pred_mean) ** 2
    )


    variance = np.mean(
        (y_pred_bootstrap - y_pred_mean) ** 2
    )


    test_errors.append(error)
    bias_squared.append(bias)
    variances.append(variance)

plt.title(f"n={n}")
plt.plot(degrees, test_errors, label="Test error")
plt.plot(degrees, bias_squared, label="Bias²")
plt.plot(degrees, variances, label="Variance")
plt.yscale("log")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.savefig(f"Project2c2n{n}.png")
plt.show()
print("Optimal degree", degrees[np.argmin(test_errors)])
