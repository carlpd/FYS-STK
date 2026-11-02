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
    return jax.nn.leaky_relu(z, negative_slope=alpha)
    #jax.lax.select(z>0, z, alpha*z)
    #return jnp.where(z > 0, z, alpha*z)
    """
    if z > 0:
        return z
    return alpha * z
    """

def make_model_FF(X, y, input_size=1, layer_output_sizes=[10, 1], act_funcs=[sigmoid, identity], gamma=0.5, epochs=100, rng=np.random.default_rng(2026), X_test=["Donotuse"]):
    if "Donotuse" in X_test:
        X_test=X
    layers = create_layers(input_size, layer_output_sizes)
    gradient_func = jax.grad(cost, argnums=1)
    layers_grad = gradient_func(X, layers, act_funcs, y)
    lays_t = train_network(X, layers, act_funcs, y, gamma=gamma, epochs=epochs)
    preds = feed_forward(X_test, lays_t, act_funcs)
    return preds

if __name__ == "__main__":
    print("For z = 1:", f"ReLU(z) = {ReLU(1)}", f"LReLU(z) = {LReLU(1)}")
    print("For z = -1:", f"ReLU(z) = {ReLU(-1)}", f"LReLU(z) = {LReLU(-1)}")
    rng = np.random.default_rng(2026)
    n = 100
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    sigma = 0.01
    y = Runge(x) + rng.normal(0.0, sigma, n)
    X = x.reshape(100, 1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = y - np.mean(y)
    y = y.reshape(-1,1)
    x_test = np.sort(rng.uniform(-1.0, 1.0, n))
    y_test = Runge(x) + rng.normal(0.0, sigma, n)
    X_test = x.reshape(100, 1)
    y_test=y_test.reshape(-1,1)

    input_size = 1

    act_funcs1 = [ReLU, identity]
    layer_output_sizes1 = [10, 1]
    pred1=make_model_FF(X,y,layer_output_sizes=layer_output_sizes1, act_funcs=act_funcs1, epochs=10000, X_test=X_test)
    
    act_funcs2 = [LReLU, identity]
    layer_output_sizes2 = [10, 1]
    pred2=make_model_FF(X,y,layer_output_sizes=layer_output_sizes2, act_funcs=act_funcs2, epochs=10000, X_test=X_test)

    act_funcs3 = [sigmoid, identity]
    layer_output_sizes3 = [10, 1]
    pred3=make_model_FF(X,y,layer_output_sizes=layer_output_sizes3, act_funcs=act_funcs3, epochs=10000, X_test=X_test)

    act_funcs4=[sigmoid, LReLU, ReLU]
    layer_output_sizes4 = [10, 5, 1]
    pred4=make_model_FF(X,y,layer_output_sizes=layer_output_sizes4, act_funcs=act_funcs4, epochs=10000, X_test=X_test)
    
    plt.plot(X_test, y_test, label="Test data")
    plt.plot(X_test, pred1, label="All RelU, sizes: 10, 1")
    plt.plot(X_test, pred2, label="All Leaky ReLU, sizes: 10, 1")
    plt.plot(X_test, pred3, label="All Sigmoid, sizes: 10, 5, 1")
    plt.plot(X_test, pred4, label="Sigmoid->Leaky->ReLU, sizes, 10, 5, 1")
    plt.legend()
    plt.show()


