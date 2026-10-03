import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import grad
from sklearn.model_selection import train_test_split

xmin=(-1)
xmax=1
seed=2026
n=100
noise=0.1
test_size=0.3
rng = np.random.default_rng(seed)
x = rng.uniform(xmin, xmax, n).reshape(-1,1)
y = 1.0 / (1.0 + 25.0 * x**2) + np.random.normal(0, noise, x.shape)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)

degree=5
y_n_tr=y_tr-y_tr.mean()
y_n_te=y_te-y_te.mean()
n_tr=len(x_tr)
n_te=len(x_te)
X_tr=np.zeros((n_tr, degree))
#print(X_tr)
for k in range(0, degree):
    for i in range(0, n_tr):
        X_tr[i,k]=x_tr[i,0]**(k+1)
X_n_tr = (X_tr - X_tr.mean(axis=0)) / X_tr.std(axis=0)

def grad_cost(X, y, theta, lam = 0.0):
    """Returns gradient of the cost function (defaults to OLS cost function)"""
    n = len(y)
    extra_term = 0
    return 2 / n * X.T @ (X @ theta - y) + 2 * lam * theta
def grad_cost_jax(theta, X, y, lam=0.0):
    return jnp.sum((X @ theta - y)**2) / len(y) + lam * jnp.sum(theta**2)

def GD(X, y, eta, lam=0.0, threshold = 1e-6, ad=False):
    """Gradient descent using analytical expression for gradient"""

    n = len(y)
    theta = np.zeros(degree)             # initialise theta
    """
    print(theta.shape)
    print(X.shape)
    print((y[:,0]).shape)
    print((X@theta-y).shape)
    """
    # gradient descent
    
    for it in range(0,10000):
        if ad==True:
            #grc=grad_cost_jax(theta, jnp.asarray(X_n_tr), jnp.asarray(y_n_tr))
            #grad_C=grad(grc)(theta)#,argnums=(theta, jnp.asarray(X), jnp.asarray(y)))
            grad_C = grad(grad_cost_jax)(theta, jnp.asarray(X_n_tr), jnp.asarray(y_n_tr[:, 0]), lam) # Used Claude for debugging
            #print(grad_C)
        elif ad==False:
            grad_C = grad_cost(X_n_tr, y_n_tr[:,0], theta) # gradient of cost function, Gradient descent using analytical expression for gradient
            

        # stopping condition
        if np.linalg.norm(grad_C) < threshold:
            break
        
        theta = theta - eta * grad_C    # calculate next theta-value

    return theta

eta=0.1
th_A=GD(X_n_tr, y_n_tr, eta, lam=0.1)
th_AD=GD(X_n_tr, y_n_tr, eta, lam=0.1, ad=True)
print("A", th_A)
print("AD", th_AD)
matches=0
for i in range(degree):
    print(i)
    print(abs(th_A[i]-th_AD[i]))
    if abs(th_A[i]-th_AD[i])<10**(-16):
        matches+=1
        print(matches)
if matches==degree:
    print("They match!")