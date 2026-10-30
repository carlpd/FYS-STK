import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
import jax
jax.config.update("jax_enable_x64", True)   # float64, so that the checks agree with numpy to 1e-15
import jax.numpy as jnp

#Defining the Runge function
def Runge(x):
    return 1/(1+25*x**2)

#Defining activation functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def sigmoid(z):
    return 1.0/(1.0 + jnp.exp(-z))

def softmax(z):
    """Softmax of each row of the matrix z: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z/jnp.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Softmax of a single vector of scores."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z/jnp.sum(e_z)


def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W =np.random.randn(i_size, layer_output_size)
        b =np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a@W+b
        a = activation_func(z)
    return a
def cross_entropy(predict, target):
    return jnp.sum(-target * jnp.log(predict))


def cost(input, layers, activation_funcs, target):
    predict = feed_forward(input, layers, activation_funcs)
    return cross_entropy(predict, target)

def train_network(
    inputs, layers, activation_funcs, targets, gamma=0.001, epochs=100
):
    for i in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*W_g, b -gamma*b_g) for (W, b), (W_g, b_g) in zip(layers, layers_grad)]
        # layersnew = []
        # for (W, b), (W_g, b_g) in zip(layers, layers_grad):
        #     W=W - W_g
        #     b=b -b_g
        #     layersnew.append((W,b))
        #     if np.isnan(np.any(W)):
        #         print("W", i)
        #     if np.isnan(np.any(b)):
        #         print("b", i)
        # layers = layersnew
    return layers
"""
def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)
"""

rng = np.random.default_rng(2026)
n = 100
x = np.sort(rng.uniform(-1.0, 1.0, n))
sigma = 0.1
y = Runge(x) + rng.normal(0.0, sigma, n)
X= x.reshape(-1, 1)
#X=(X-X.mean(axis=0))/X.std(axis=0) #Scaling

input_size=1
act_funcs=[sigmoid, sigmoid, sigmoid]
layer_output_sizes=[10, 5, 1]
layers=create_layers(input_size,layer_output_sizes)



gradient_func = jax.grad(
    cost, argnums=1
) 
layers_grad = gradient_func(
    X, layers, act_funcs, y
)

lays_t=train_network(
    X, layers, act_funcs, y, gamma=0.001, epochs=100
)
"""
for i in lays_t:
    W,b = i
    print("W shape: ", W.shape)
    print("b shape: ", b.shape)
"""
#print(lays_t)
preds=feed_forward(X, lays_t, act_funcs)
print(preds)
print(preds.shape[0])
y=y.reshape(100, 1)
print(y)
print(y.shape)

mse=np.sum((preds-y)**2)/preds.shape[0]
print(mse)

plt.plot(X, preds,label="Predictions")
plt.plot(X, y, label="data")
plt.show()
