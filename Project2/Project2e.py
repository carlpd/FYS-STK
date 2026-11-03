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

def one_hot(y, n_classes=None):
    """Integer labels -> one-hot rows."""
    n_classes = int(np.max(y)) + 1 if n_classes is None else n_classes
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64,), activation="relu", gamma=0.1, lmbd=0.0, epochs=10, batch_size=100, seed=2026):
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = one_hot(np.searchsorted(self.classes_, y), len(self.classes_))
        layers = create_layers([X.shape[1], *self.hidden, len(self.classes_)], self.activation, np.random.default_rng(self.seed))
        self.layers_, self.history_ = train(X, Y, layers, self.activation, "softmax", gamma=self.gamma, epochs=self.epochs,
                                            batch_size=self.batch_size, lmbd2=self.lmbd, rng=np.random.default_rng(self.seed))
        return self

    def predict_proba(self, X):
        return predict(X, self.layers_, self.activation, "softmax")

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

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
    print(targets[0])
    y_label=np.zeros(targets.shape[0])
    print(y_label)
    for i in range(len(targets)):
        #print(i)
        #print(np.where(targets[i]==1)[0][0])
        y_label[i]=np.where(targets[i]==1)[0][0]
    print(y_label)
    
    #print(labels)
    xtr, xte, ytr, yte = train_test_split(inputs, y_label, test_size=0.3, random_state=50)
    #print(ytr.shape)
    #print(ytr)
    #print(xtr.shape)
    model=NeuralNetClassifier(hidden=(50, 40, 10), activation="relu", epochs=1000, gamma=0.5)
    fitted=model.fit(xtr, ytr)
    pred=model.predict(xte)
    print(pred)
    points=0
    for i in range(len(pred)):
        if pred[i]==yte[i]:
            points+=1
    acc=points/len(pred)
    print(acc)

