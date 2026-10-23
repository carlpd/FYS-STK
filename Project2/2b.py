import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def Runge(x):
    return 1/(1+25*x**2)


# Defining some activation functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def create_layers(network_input_size, layer_output_sizes): #batch
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W =np.random.randn(i_size, layer_output_size)
        b =np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward(inputs, layers, activation_funcs): #batch
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a@W+b
        a = activation_func(z)
    return a