"""
Answers to the main tutorial notebooks.
"""

import jax.numpy as np
import numpy.random as npr
from jax import grad


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error function."""
    return np.mean(np.power(y_true - y_pred, 2))


def linear_model(theta, x):
    w = theta["w"]
    b = theta["b"]
    return w * x + b


def initialize_linear_params():
    params = dict()
    params["w"] = npr.normal()
    params["b"] = npr.normal()
    return params


def mseloss(theta, model, x, y):
    y_est = model(theta, x)
    return mse(y, y_est)


dmseloss = grad(mseloss)

from tqdm.autonotebook import tqdm


def model_optimization_loop(
    theta, model, loss, x, y, n_steps=3000, step_size=0.001
):
    dloss = grad(loss)
    losses = []
    for i in tqdm(range(n_steps)):
        grads = dloss(theta, model, x, y)
        for name, param in theta.items():
            theta[name] = theta[name] - grads[name] * step_size
        losses.append(loss(theta, model, x, y))
    return losses, theta


def logistic(x):
    """Logistic transform."""
    return 1 / (1 + np.exp(-x))


def logistic_model(theta, x):
    w = theta["w"]
    b = theta["b"]
    z = w * x + b
    y = logistic(z)
    return y


def binary_cross_entropy(y_true, y_preds):
    """Function for binary cross entropy."""
    return np.sum(
        y_true * np.log(y_preds) + (1 - y_true) * np.log(1 - y_preds)
    )


def logistic_loss(params, model, x, y):
    """Logistic loss function.

    Params are in first position
    so that loss function is conveniently differentiable using JAX.
    """
    preds = model(params, x)
    return -binary_cross_entropy(y, preds)


dlogistic_loss = grad(logistic_loss)


def f(w):
    return w**2 + 3 * w - 5


def df(w):
    """The hand-written derivative of f with respect to w."""
    return 2 * w + 3


def noise(n):
    return npr.normal(size=(n))


def make_y(x, w, b):
    return w_true * x + b_true + noise(len(x))


x = np.linspace(-5, 5, 100)
w_true = 2
b_true = 20
