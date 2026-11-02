import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp  

from Project2b_new import(
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
    NeuralNetRegressor
)



if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    sigma = 0.1
    y = Runge(x) + rng.normal(0.0, sigma, n)
    X = x.reshape(-1, 1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = y - np.mean(y)
    y = y.reshape(-1,1)
    x_test = np.sort(rng.uniform(-1.0, 1.0, n))
    y_test = Runge(x_test) + rng.normal(0.0, sigma, n)
    X_test = x.reshape(100, 1)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
    y_test = y_test - np.mean(y_test)
    y_test=y_test.reshape(-1,1)
    ReLU3=NeuralNetRegressor(hidden=(50, 40, 10), activation="relu", epochs=10000, gamma=0.01).fit(X,y)
    LReLU3=NeuralNetRegressor(hidden=(50, 40 ,10), activation="lrelu", epochs=10000, gamma=0.01).fit(X,y)
    sigmoid3=NeuralNetRegressor(hidden=(50, 40, 10), activation="sigmoid", epochs=10000, gamma=0.5).fit(X,y)
    ReLU3_pred=ReLU3.predict(X_test)
    LReLU3_pred=LReLU3.predict(X_test)
    sigmoid3_pred=sigmoid3.predict(X_test)
          
    plt.plot(X_test, y_test, label="Test data")
    plt.plot(X_test, ReLU3_pred, label="ReLU, 3 HL")
    plt.plot(X_test, LReLU3_pred, label="LReLU, 3 HL")
    plt.plot(X_test, sigmoid3_pred, label="sigmoid, 3 HL")
    plt.legend()
    plt.show()