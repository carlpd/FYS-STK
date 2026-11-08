# FYS-STK3155 Project 2

Carl Petter Duedahl, Ole Morten Gården and Ingi Benedikt Jónasson

Welcome to our second project in FYS-STK3155!

In this project we made our own feed forward neural network in NumPy, with backpropagation, sigmoid, ReLU and leaky ReLU, L1 and L2 regularization, and the optimizers from project 1 (GD, SGD, RMSprop and Adam). We used it for regression on the Runge function, where we compare with OLS, Ridge and Lasso from project 1, and for classification of the MNIST handwritten digits, where we compare with logistic regression and Scikit-Learn.

The report is Project2.pdf in this folder.

## What's in the folder

- `Code/Project2.ipynb`: the main notebook. All the results and plots in the report come from here.
- `Code/Project2b.py`: the neural network and the `NeuralNetRegressor` class (task b-d)
- `Code/Project2e.py`: the `NeuralNetClassifier` class for MNIST (task e)
- `Code/Project1bb.py` and `Code/utils.py`: code from project 1 that we reuse (data, scaling, OLS and the optimizers)
- `Code/Project2c.py` and `Code/Project2d.py`: small test scripts we used before moving things into the notebook, you can ignore these
- `Code/mnist.npz`: the MNIST data
- `Plots/`: our results (the plots we use in the report)
- `Old Files/`: older versions of the code that we don't use anymore

## Installation

We used Python 3 with Anaconda (https://www.anaconda.com/docs/getting-started/installation), which already has numpy, scipy, matplotlib, scikit-learn and jupyter. In addition you need jax, which we only use to check our backpropagation:

```
pip install jax jaxlib
```

You can also install everything at once with

```
pip install -r Requirements.txt
```

## How to run

Run the notebook from inside the Code folder, since it loads `mnist.npz` from there and saves the plots to `../Plots/`:

```
cd Code
jupyter notebook Project2.ipynb
```

Then do Restart & Run All. It takes about an hour on our laptops, mostly because of the cross validation grids and the MNIST runs. We use the seed 2026 everywhere.

In the `Project2[letter].py` files you can ignore what happens after `if __name__ == "__main__":`. That part was mostly for testing the code before we put it in the notebook.

If you run the notebook on another computer, the Runge results should come out the same. The runs that blow up (huge MSE or nan in the heatmaps) and the MNIST accuracies can be a bit different though, around 0.001 for MNIST, because of small floating point differences between machines. When we reran it ourselves, the best activation on MNIST switched between sigmoid and leaky ReLU, which fits with the differences being within the standard error we discuss in the report.

## The notebook

- **Task b:** sigmoid networks with one and two hidden layers of 50 and 100 nodes compared with OLS of degree 14 from project 1, a gradient check against `jax.grad`, and cross validation over the learning rate for GD, SGD, RMSprop and Adam. Plots: `b_comparison_to_data`, `b_learning_rates`, `b_convergence`
- **Task c:** sigmoid, ReLU and leaky ReLU for different numbers of layers and nodes, and a check for overfitting. Plots: `nhl_mse`, `c_depth_width`, `c_overfitting`
- **Task d:** L2 and L1 regularization, and a comparison with Ridge and Lasso from project 1. Plots: `d_gamma_lambda`, `d_arch_lambda`
- **Task e:** logistic regression as the baseline, the different optimizers, heatmaps of the hyperparameters, the best model on the test set with a confusion matrix, and a comparison with Scikit-Learn. Plots: `e_optimisers`, `e_heatmaps`, `e_confusion_matrix`

Happy evaluating!
