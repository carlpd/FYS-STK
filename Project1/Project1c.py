import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Copyed from Project1a.py and increased the number of polynomials and n
def runge_data(n=100, noise=0.1, xmin=(-1), xmax=1, seed=2026):
    """Runge function 1/(1+25x^2) on [-1,1], standardised polynomial features, centred y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, n)
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n)
    return x, y


def closed_form(X, y, lam=0.0):
    """Lukket OLS/Ridge-løsning."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)

def MSE(y_data, y_model):
    return np.mean((y_data - y_model) ** 2)

n=150
sigma=0.1
max_degrees=17
x, y=runge_data(n=n, noise=sigma, xmin=-1, xmax=1)
x_tr, x_te, y_tr, y_te = train_test_split(x, y,test_size=0.3,random_state=2026)
y_n_tr=y_tr-y_tr.mean()
y_n_te=y_te-y_te.mean()
ys=[]
n_tr=len(x_tr)
n_te=len(x_te)
MSEs_tr=np.zeros(max_degrees)
MSEs_te=np.zeros(max_degrees)


degrees=np.linspace(1,max_degrees, max_degrees)
print(degrees)
for degree in range(1, max_degrees+1):
    X_tr=np.zeros((n_tr, degree))
    for k in range(0, degree):
        for i in range(0, n_tr):
            X_tr[i,k]=x_tr[i]**(k+1)
    X_n_tr = (X_tr - X_tr.mean(axis=0)) / X_tr.std(axis=0)
    X_te=np.zeros((n_te, degree))
    for k in range(0, degree):
        for i in range(0, n_te):
            X_te[i,k]=x_te[i]**(k+1)
    X_n_te = (X_te - X_te.mean(axis=0)) / X_te.std(axis=0)
    th=closed_form(X_n_tr, y_n_tr)
    sol=(th @ X_n_te.T)
    ys.append(sol)
    
    MSEs_te[degree-1]=(MSE(y_n_te, sol))
    sol_tr=(th @ X_n_tr.T)
    MSEs_tr[degree-1]=(MSE(y_n_tr, sol_tr))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#line, = ax.plot(a, color='blue', lw=2)

ax.semilogy(degrees, MSEs_te, label="OLS test")
ax.semilogy(degrees, MSEs_tr, label="OLS training")
#ax.semilogy(degrees, MSEs_te, label="OLS test")
#ax.plot(degrees, MSEs_tr, label="OLS training")
#ax.plot(degrees, MSEs_te, label="OLS test")
#ax.set_yscale('log')
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("Mean Squared Error")
ax.legend()
plt.show()

