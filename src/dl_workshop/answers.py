"""
Answers to the main tutorial notebooks.
"""
import jax.numpy as np
from jax import grad
import numpy.random as npr


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error function."""
    return np.mean(np.power(y_true - y_pred, 2))


def linear_model(theta, x):
    w, b = theta
    return w * x + b

def initialize_linear_params():
    w = npr.normal()
    b = npr.normal()
    return w, b


def mseloss(theta, model, x, y):
    y_est = model(theta, x)
    return mse(y, y_est)

dmseloss = grad(mseloss)

from tqdm.autonotebook import tqdm

def linear_model_optimization(theta, model, x, y):
    losses = []
    w, b = theta
    for i in tqdm(range(3000)):
        dw, db = dmseloss(theta, model, x, y)
        w = w - dw * 0.001
        b = b - db * 0.001
        theta = (w, b)
        losses.append(mseloss(theta, model, x, y))
    return losses, theta
