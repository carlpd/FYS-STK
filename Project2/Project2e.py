import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
import jax
jax.config.update("jax_enable_x64", True)   # float64, so that the checks agree with numpy to 1e-15
import jax.numpy as jnp
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)
from sklearn.model_selection import train_test_split


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
    NeuralNetRegressor
)

def one_hot(y, n_classes=None):
    """Integer labels -> one-hot rows."""
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64,), activation="relu", gamma=0.1, lmbd=0.0, epochs=10, batch_size=100, seed=2026,
                 L2=True, optimizer="gd", penalize_bias=True):
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
        self.classes_ = np.unique(y)
        Y = one_hot(np.searchsorted(self.classes_, y), len(self.classes_))
        layers = create_layers([X.shape[1], *self.hidden, len(self.classes_)], self.activation, np.random.default_rng(self.seed))
        lmbd2, lmbd1 = (self.lmbd, 0.0) if self.L2 else (0.0, self.lmbd)
        self.layers_, self.history_ = train(X, Y, layers, self.activation, "softmax", gamma=self.gamma, epochs=self.epochs,
                                            batch_size=self.batch_size, lmbd2=lmbd2, lmbd1=lmbd1,
                                            rng=np.random.default_rng(self.seed), optimizer=self.optimizer,
                                            penalize_bias=self.penalize_bias)
        return self

    def predict_proba(self, X):
        return predict(X, self.layers_, self.activation, "softmax")

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def accuracy_class(pr, te):
    pr, te = np.ravel(pr), np.ravel(te)
    assert len(pr) == len(te)
    return np.mean(pr == te)
if __name__ == "__main__":
    """
    digits = datasets.load_digits()
    inputs = digits.data / 16.0                    # 1797 rows, 64 features, scaled to [0, 1]
    labels = digits.target
    targets = np.zeros((len(labels), 10))
    targets[np.arange(len(labels)), labels] = 1.0   # one-hot rows
    print("inputs", inputs.shape, " labels", labels.shape, " one-hot targets", targets.shape)
    print("first label:", labels[0], " its one-hot row:", targets[0])
    """
    """
    fig, axes = plt.subplots(1, 10, figsize=(10, 1.4))
    for ax, k in zip(axes, range(10)):
        ax.imshow(digits.images[k], cmap="gray_r"); ax.set_title(labels[k], fontsize=9); ax.axis("off")
    plt.show()
    """
    
    with np.load("mnist.npz") as data:
        xtr, ytr = data["x_train"], data["y_train"]
        xte, yte = data["x_test"], data["y_test"] # untouched until the end
    print(xtr.shape, ytr.shape, xte.shape, yte.shape)
    xtr = xtr.reshape(len(xtr), -1) / 255.0
    xte = xte.reshape(len(xte), -1) / 255.0
    """
        print(targets[0])
    y_label=np.zeros(targets.shape[0])
    print(y_label)
    for i in range(len(targets)):
        #print(i)
        #print(np.where(targets[i]==1)[0][0])
        y_label[i]=np.where(targets[i]==1)[0][0]
    print(y_label)
    """
    #print(labels)
    #xtr, xte, ytr, yte = train_test_split(inputs, y_label, test_size=0.3, random_state=50)
    xfit, xval, yfit, yval = train_test_split(xtr, ytr, test_size=10000, stratify=ytr, random_state=2026)

    baseline = NeuralNetClassifier(hidden=(), epochs=20, gamma=0.5).fit(xfit, yfit)
    print("logistic regression (no hidden layers), validation accuracy:", accuracy_class(baseline.predict(xval), yval))
    model=NeuralNetClassifier(hidden=(50,), activation="sigmoid", epochs=20, gamma=0.5)
    fitted=model.fit(xfit, yfit)
    print("1 hidden layer (50, sigmoid), validation accuracy:", accuracy_class(model.predict(xval), yval))

    pred=model.predict(xte)
    acc=accuracy_class(pred, yte)
    print("test accuracy:", acc)

