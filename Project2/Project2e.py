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
from sklearn.model_selection import train_test_split


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
    digits = datasets.load_digits()
    inputs = digits.data / 16.0                    # 1797 rows, 64 features, scaled to [0, 1]
    labels = digits.target
    targets = np.zeros((len(labels), 10))
    targets[np.arange(len(labels)), labels] = 1.0   # one-hot rows
    print("inputs", inputs.shape, " labels", labels.shape, " one-hot targets", targets.shape)
    print("first label:", labels[0], " its one-hot row:", targets[0])
    """
    fig, axes = plt.subplots(1, 10, figsize=(10, 1.4))
    for ax, k in zip(axes, range(10)):
        ax.imshow(digits.images[k], cmap="gray_r"); ax.set_title(labels[k], fontsize=9); ax.axis("off")
    plt.show()
    """

    xtr, xte, ytr, yte = train_test_split(inputs, labels, test_size=0.3, random_state=50)
    model=NeuralNetRegressor(hidden=(50, 40, 10), activation="softmax", epochs=1000, gamma=0.5)
    fitted=model.fit(xtr, ytr)

