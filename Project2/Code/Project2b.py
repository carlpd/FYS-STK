import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)
from scipy.special import expit
from sklearn.model_selection import train_test_split
from utils import runge_data, Scaler

from utils import make_update


def Runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def layer_activations(hidden_activation, n_hidden):
    if isinstance(hidden_activation, str):
        return [hidden_activation] * n_hidden
    acts = list(hidden_activation)
    if len(acts) != n_hidden:
        raise ValueError(f"{len(acts)} activations given for {n_hidden} hidden layers")
    return acts


def create_layers(sizes, hidden_activation="sigmoid", rng=None):
    """A list of (W, b) pairs, W of shape (inputs, outputs) as in the template.

    sizes = [784, 100, 10] gives two layers: 784 -> 100 -> 10.  The weights are
    drawn with the variance of Eq. (8.74) (He) for ReLU and 1/inputs otherwise,
    the biases are set to 0.01 as in the template.
    """
    rng = np.random.default_rng(2026) if rng is None else rng
    n_hidden = len(sizes) - 2
    if isinstance(hidden_activation, str):
        init = [hidden_activation] * (n_hidden + 1)
    else:
        acts = layer_activations(hidden_activation, n_hidden)
        init = acts + [acts[-1] if acts else "sigmoid"]
    layers = []
    for n_in, n_out, act in zip(sizes[:-1], sizes[1:], init):
        scale = np.sqrt(2.0 / n_in) if act in ("relu", "lrelu") else np.sqrt(1.0 / n_in)
        W = rng.normal(0.0, scale, (n_in, n_out))
        b = np.zeros(n_out) + 0.01
        layers.append((W, b))
    return layers


def feed_forward(X, layers, hidden_activation="sigmoid", head="mse"):
    """Eq. (8.21) layer by layer, samples in rows.

    Returns the lists z (pre-activations) and a (activations, a[0] = X): every
    one of them is needed on the way back, so the forward pass stores them.
    """
    acts = layer_activations(hidden_activation, len(layers) - 1)
    a, z = [X], []
    for l, (W, b) in enumerate(layers):
        zl = a[-1] @ W + b                       # the template's  x @ w_1 + b_1
        z.append(zl)
        if l == len(layers) - 1:
            a.append(OUTPUT[head](zl))           # the output head
        else:
            a.append(ACTIVATIONS[acts[l]][0](zl))
    return z, a

def ReLU(z):
    return np.where(z > 0, z, 0.0)
def LReLU(z):
    """
    Leaky ReLU

    Argument:
        z

    Returns:
        z,          if z >  0,
        alpha * z,  if z <= 0,      alpha = 0.01
    """
    alpha = 0.01
    return np.where(z > 0, z, alpha * z)

def sigmoid(z):
    return expit(z)

def softmax(z):
    """Softmax of each row of the matrix z: one row of scores per sample."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Softmax of a single vector of scores."""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def identity(z):
    return z


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def ReLU_prime(z):
    return (z > 0).astype(float)

def LReLU_prime(z):
    alpha=0.01
    return np.where(z>0, 1.0, alpha)

def identity_prime(z):
    return np.ones_like(z)

def softmax_prime(z):
    s = softmax_vec(z)
    return np.diag(s) - np.outer(s, s)

ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_prime),
               "relu": (ReLU, ReLU_prime),
               "lrelu": (LReLU, LReLU_prime),
               "identity": (identity, identity_prime),
               "linear": (identity, identity_prime)}

OUTPUT = {"mse": identity, "binary": sigmoid, "softmax": softmax}

def MSE_2(pred, y_te):
    pred, y_te = np.ravel(pred), np.ravel(y_te)
    if pred.shape != y_te.shape:
        raise ValueError(f"MSE_2: {pred.shape} predictions vs {y_te.shape} targets")
    return np.mean((y_te - pred) ** 2)

def backpropagation(X, Y, layers, hidden_activation="sigmoid", head="mse", lmbd2=0.0, lmbd1=0.0,
                    penalize_bias=True):
    """Eqs. (8.65)-(8.68) for all samples at once; returns a list of (dW, db) with the shapes of (W, b)."""
    acts = layer_activations(hidden_activation, len(layers) - 1)
    n = X.shape[0]
    z, a = feed_forward(X, layers, hidden_activation, head)
    delta = (a[-1] - Y) / n                      # Eq. (8.69): the same line for all three heads
    grads = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        W, b = layers[l]
        dW = a[l].T @ delta                      # Eq. (8.68): outer products, summed over the samples
        dW += 2.0 * lmbd2 * W + lmbd1 * np.sign(W)   # the penalties of Section 8.12
        db = np.sum(delta, axis=0)               # Eq. (8.67)
        if penalize_bias:
            db += 2.0 * lmbd2 * b + lmbd1 * np.sign(b)
        grads[l] = (dW, db)
        if l > 0:
            fp = ACTIVATIONS[acts[l - 1]][1]
            delta = (delta @ W.T) * fp(z[l - 1])     # Eq. (8.66): pull back through W, modulate by f'
    return grads
def cost(X, Y, layers, hidden_activation="sigmoid", head="mse", lmbd2=0.0, lmbd1=0.0, penalize_bias=True):
    """The cost of Eq. (8.31): data term for the chosen head plus L2 and L1 penalties on the weights."""
    out = predict(X, layers, hidden_activation, head)
    n = X.shape[0]
    if head == "mse":                            # Eq. (8.32)
        data = 0.5 * np.sum((out - Y) ** 2) / n
    elif head == "binary":                       # Eq. (5.13), averaged
        p = np.clip(out, 1e-12, 1 - 1e-12)
        data = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    else:                                        # Eq. (5.28), averaged, one-hot Y
        p = np.clip(out, 1e-12, 1.0)
        data = -np.mean(np.sum(Y * np.log(p), axis=1))
    params = [p for W, b in layers for p in ((W, b) if penalize_bias else (W,))]
    reg = sum(lmbd2 * np.sum(p ** 2) + lmbd1 * np.sum(np.abs(p)) for p in params)
    return data + reg
def train(X, Y, layers, hidden_activation="sigmoid", head="mse", gamma=0.1, epochs=100,
          batch_size=None, lmbd2=0.0, lmbd1=0.0, rng=None, X_test=None, y_test=None, verbose=False,
          optimizer="gd", penalize_bias=True):
    """Plain gradient descent, Eq. (8.70), on minibatches (batch_size=None: the whole set).

    Returns the trained layers and a history with the training cost after
    every epoch and, when a test set is given, the test accuracy.
    """
    rng = np.random.default_rng(2026) if rng is None else rng
    n = X.shape[0]
    M = n if batch_size is None else batch_size
    updaters = [(make_update(optimizer, W.shape), make_update(optimizer, b.shape)) for W, b in layers]
    history = {"cost": [], "test_accuracy": []}
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, M):
            idx = order[s:s + M]
            grads = backpropagation(X[idx], Y[idx], layers, hidden_activation, head, lmbd2, lmbd1, penalize_bias)
            layers = [(uW(W, dW, gamma), ub(b, db, gamma))
                      for (W, b), (dW, db), (uW, ub) in zip(layers, grads, updaters)]
        history["cost"].append(cost(X, Y, layers, hidden_activation, head, lmbd2, lmbd1, penalize_bias))
        if X_test is not None:
            history["test_accuracy"].append(accuracy(X_test, y_test, layers, hidden_activation, head))
        if verbose:
            msg = f"epoch {epoch + 1:3d}: cost {history['cost'][-1]:.4f}"
            if X_test is not None:
                msg += f"  test accuracy {history['test_accuracy'][-1]:.4f}"
            print(msg)
    return layers, history

def accuracy(X, y, layers, hidden_activation, head):
    """Fraction of correct labels; y holds integer labels for softmax, 0/1 for binary."""
    out = predict(X, layers, hidden_activation, head)
    if head == "softmax":
        return np.mean(np.argmax(out, axis=1) == y)
    return np.mean((out[:, 0] > 0.5) == (y > 0.5))
def predict(X, layers, hidden_activation="sigmoid", head="mse"):
    """The output of the network: a vector of predictions, probabilities, or class probabilities."""
    return feed_forward(X, layers, hidden_activation, head)[1][-1]



class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """Your Project 2 network with the scikit-learn estimator interface."""

    def __init__(self, hidden=(50,), activation="sigmoid", gamma=0.03, lmbd=0.0, epochs=100, batch_size=10, seed=2026, L2=True,
                 optimizer="gd", penalize_bias=True):
        # store the hyperparameters under the same names; no computation here
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.L2 = L2
        self.optimizer = optimizer
        self.penalize_bias = penalize_bias

    def fit(self, X, y):
        # X has shape (n, 1), y shape (n,): build the layers from self.seed,
        # train with your SGD for self.epochs epochs, store the layers in self.layers_
        sizes = [X.shape[1], *self.hidden, 1]
        layers = create_layers(sizes, self.activation, np.random.default_rng(self.seed))
        lmbd2, lmbd1 = (self.lmbd, 0.0) if self.L2 else (0.0, self.lmbd)
        self.layers_, self.history_ = train(X, np.reshape(y, (-1, 1)), layers, self.activation, "mse", gamma=self.gamma,
                                            epochs=self.epochs, batch_size=self.batch_size, lmbd2=lmbd2, lmbd1=lmbd1,
                                            rng=np.random.default_rng(self.seed), optimizer=self.optimizer,
                                            penalize_bias=self.penalize_bias)
        return self

    def predict(self, X):
        # the network output as a 1D array of length len(X)
        return predict(X, self.layers_, self.activation, "mse")[:, 0]


def runge_split(n=100, sigma=0.1, seed=2026, test_size=0.3):
    x, yy = runge_data(n=n, noise=sigma, seed=seed)
    x_tr, x_te, y_tr, y_te = train_test_split(x, yy, test_size=test_size, random_state=seed)
    i_tr, i_te = np.argsort(x_tr), np.argsort(x_te)
    x_tr, y_tr, x_te, y_te = x_tr[i_tr], y_tr[i_tr], x_te[i_te], y_te[i_te]
    sc = Scaler().fit(x_tr.reshape(-1, 1), y_tr)
    X, X_test = sc.transform_X(x_tr.reshape(-1, 1)), sc.transform_X(x_te.reshape(-1, 1))
    return X, (y_tr - sc.y_mean).reshape(-1, 1), X_test, (y_te - sc.y_mean).reshape(-1, 1), sc


if __name__ == "__main__":
    X, y, X_test, y_test, sc = runge_split()
    act="sigmoid"
    model=NeuralNetRegressor(hidden=(50, 40, 4), activation="sigmoid", epochs=100, gamma=0.5)
    fitted=model.fit(X, y)
    pred=model.predict(X_test)
    print("test MSE:", MSE_2(pred, y_test))
    plt.plot(X_test, y_test, "o", label="test data")
    plt.plot(X_test, pred, label="Sigmoid NN")
    plt.legend()
    plt.show()
