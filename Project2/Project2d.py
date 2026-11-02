import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
import jax
jax.config.update("jax_enable_x64", True)   # float64, so that the checks agree with numpy to 1e-15
import jax.numpy as jnp

from Project2b import (
    Runge,
    ReLU,
    sigmoid,
    softmax,
    softmax_vec,
    identity,
    create_layers,
    feed_forward,
    cross_entropy,
    cost,
    train_network

)

from Project2c import (
    LRelU
    make_model
)