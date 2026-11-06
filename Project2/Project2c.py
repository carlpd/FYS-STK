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
    ReLU3=NeuralNetRegressor(hidden=(50, 40, 10), activation="relu", epochs=10000, gamma=0.01).fit(X,y)
    LReLU3=NeuralNetRegressor(hidden=(50, 40 ,10), activation="lrelu", epochs=10000, gamma=0.01).fit(X,y)
    sigmoid3=NeuralNetRegressor(hidden=(50, 40, 10), activation="sigmoid", epochs=10000, gamma=0.5).fit(X,y)
    ReLU3_pred=ReLU3.predict(X_test)
    LReLU3_pred=LReLU3.predict(X_test)
    sigmoid3_pred=sigmoid3.predict(X_test)
    for name, p in [("ReLU", ReLU3_pred), ("LReLU", LReLU3_pred), ("sigmoid", sigmoid3_pred)]:
        print(f"{name:8s} 3 HL: test MSE {MSE_2(p, y_test):.4f}")

    plt.plot(X_test, y_test, "o", label="Test data")
    plt.plot(X_test, ReLU3_pred, label="ReLU, 3 HL")
    plt.plot(X_test, LReLU3_pred, label="LReLU, 3 HL")
    plt.plot(X_test, sigmoid3_pred, label="sigmoid, 3 HL")
    plt.legend()
    plt.show()
