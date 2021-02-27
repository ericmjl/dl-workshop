from jax import random, lax, numpy as np, vmap


def loopless_loops_ex1(mat1, mat2):
    return vmap(np.dot)(mat1, mat2)


def loopless_loops_ex2(data):
    """Data is 3D, of shape (n_datasets, n_rows, n_columns)."""

    def inner(dataset):
        """dataset is 2D, of shape (n_rows, n_columns)."""
        cp = vmap(np.cumprod)(dataset)
        s = vmap(np.sum)(cp)
        return s

    return vmap(inner)(data)


from functools import partial


def loopless_loops_ex3(node_feats):
    def concat(n1, n2):
        return np.concatenate([n1, n2])

    def concatenate(node, node_feats):
        return vmap(partial(concat, node))(node_feats)

    return vmap(partial(concatenate, node_feats=node_feats))(node_feats)




def lax_scan_ex_1(prev_wealth, time, interest_factor):
    new_wealth = prev_wealth * interest_factor
    return new_wealth, prev_wealth


def lax_scan_ex_2(num_breaks: int, frac: float) -> np.ndarray:
    def step(stick_length: float, frac: float):
        stick = stick_length * frac 
        remainder = stick_length - stick
        return remainder, stick

    fracs = np.array([frac] * num_breaks)
    final, sticks = lax.scan(step, init=1.0, xs=fracs)
    return sticks


def randomness_ex_1(key, num_breaks, concentration: float):
    def step(stick_length: float, key, concentration: float):
        fraction = random.beta(key, a=1, b=concentration)
        stick = stick_length * fraction
        remainder = stick_length - stick
        return remainder, stick

    step_one_conc = partial(step, concentration=concentration)
    keys = random.split(key, num_breaks)
    final, sticks = lax.scan(step_one_conc, init=1.0, xs=keys)
    return final, sticks