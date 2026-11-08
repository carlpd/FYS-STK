# FYS-STK3155 Project 2: Neural networks for regression and classification

**Group members:** Carl Petter Duedahl, Ole Morten Gården and Ingi Benedikt Jónasson

Welcome to our second project in FYS-STK3155!

In this project we write a feed-forward neural network from scratch in NumPy, with backpropagation, the sigmoid, ReLU and Leaky ReLU activation functions, $L_1$ and $L_2$ penalties, and the optimisers plain gradient descent, stochastic gradient descent, RMSprop and Adam from project 1. We use the network for

* **regression** on the noisy one-dimensional Runge function $f(x) = 1/(1 + 25x^2)$, where we compare it with OLS, Ridge and Lasso regression from project 1, and
* **classification** of the MNIST data set of handwritten digits, where we compare it with logistic regression and with Scikit-Learn.

The report is `Project2.pdf` in this folder.

## Folder structure

```
Project2/
├── Project2.pdf          the report
├── README.md             this file
├── Requirements.txt      Python packages needed to run the code
├── Code/
│   ├── Project2.ipynb    the main notebook, produces all results and figures
│   ├── Project2b.py      the neural network and NeuralNetRegressor (parts b-d)
│   ├── Project2e.py      NeuralNetClassifier for MNIST (part e)
│   ├── Project2c.py      small test scripts used while developing (can be ignored)
│   ├── Project2d.py      small test script used while developing (can be ignored)
│   ├── Project1bb.py     OLS from project 1, used for the comparison
│   ├── utils.py          shared functions from project 1: data, scaling, regression, optimisers
│   ├── mnist.npz         the MNIST data set
│   └── Plots/            all figures used in the report
└── Old Files/            earlier versions of the code, not used
```

## Installation

We use Python 3 (version 3.10 or newer) with Anaconda, which already includes NumPy, SciPy, Matplotlib, Scikit-Learn and Jupyter: <https://www.anaconda.com/docs/getting-started/installation>.

Alternatively, all packages can be installed with pip from the `Project2` folder:

```
pip install -r Requirements.txt
```

JAX is only used to check our backpropagation against automatic differentiation.

## How to run

Open and run the notebook from inside the `Code` folder, since it reads `mnist.npz` and saves the figures to `Plots/` relative to that folder:

```
cd Code
jupyter notebook Project2.ipynb
```

Use *Restart & Run All*. The notebook imports the functions and classes from `Project2b.py`, `Project2e.py`, `Project1bb.py` and `utils.py`. All random numbers, data splits and weight initialisations use the seed 2026. A full run takes about an hour on an 8-core laptop; the cross-validation grids in parts b-d and the MNIST runs in part e take the longest.

On a different machine, the regression results for the Runge function are reproduced to about 15 digits. Two kinds of results can differ slightly, because the order of floating-point operations depends on the hardware and the linear algebra library:

* grid points where the training diverges (very large CV MSEs or `nan`), since round-off errors grow quickly once the training is unstable;
* the MNIST runs, where the validation and test accuracies can change by about 0.001. This is within the standard error of about 0.0015 discussed in the report, so the best activation function in the right panel of `e_heatmaps.pdf` can change between runs.

In the files `Project2[letter].py` you can ignore what happens after `if __name__ == "__main__":`. These parts were mostly used to test that the code worked before we moved it into the notebook.

## Contents of the notebook and the results it produces

| Section | What it does | Results |
|---|---|---|
| Task b | Sigmoid networks with one and two hidden layers of 50 and 100 nodes, compared with OLS of degree 14 from project 1 | Table I, `Plots/b_comparison_to_data.pdf` |
| Gradient check | Compares our backpropagation with `jax.grad` for 54 combinations of activation, cost function, depth and penalty | Largest difference $8.9 \times 10^{-16}$ |
| Learning rates | Five-fold CV of plain GD, SGD, RMSprop and Adam for different learning rates, and the training cost per epoch | `Plots/b_learning_rates.pdf`, `Plots/b_convergence.pdf`, Table II |
| Task c | Sigmoid, ReLU and Leaky ReLU as functions of the number of hidden layers and nodes, and a check for overfitting | `Plots/nhl_mse.pdf`, `Plots/c_depth_width.pdf`, `Plots/c_overfitting.pdf` |
| Task d | $L_2$ and $L_1$ penalties: learning rate against $\lambda$, architecture against $\lambda$, and comparison with Ridge and Lasso from project 1 | `Plots/d_gamma_lambda.pdf`, `Plots/d_arch_lambda.pdf`, Table III |
| Task e | Logistic regression as the baseline, optimisers and learning rates, heat maps of the hyperparameters, the best model on the test set with its confusion matrix, and comparison with Scikit-Learn | `Plots/e_optimisers.pdf`, `Plots/e_heatmaps.pdf`, `Plots/e_confusion_matrix.pdf`, Table IV |

Happy evaluating!
