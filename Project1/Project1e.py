import matplotlib.pyplot as plt
import numpy as np
import jax

xmin=(-1)
xmax=1
seed=2026
n=100
noise=0.1
rng = np.random.default_rng(seed)
x = rng.uniform(xmin, xmax, n).reshape(-1,1)
y = 1.0 / (1.0 + 25.0 * x**2) + np.random.normal(0, noise, x.shape)

def grad_cost(X, y, theta, lam = 0):
    """Returns gradient of the cost function (defaults to OLS cost function)"""
    n = len(y)
    extra_term = 0
    return 2 / n * X.T @ (X @ theta - y) + 2 * lam * theta

def AGD(X, y, eta, method = "OLS" threshold = 1e-6):
    """Gradient descent using analytical expression for gradient"""

    n = len(y)
    theta = np.zeros(n)                 # initialise theta

    # gradient descent
    while true:
        grad_C = grad_cost(X, y, theta) # gradient of cost function

        # stopping condition
        if np.linalg.norm(grac_C) < threshold:
            break
        
        theta = theta - eta * grad_C    # calculate next theta-value

    return theta
