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


def make_gaussian_random_walk_func(num_steps):
    def gaussian_random_walk(key):
        keys = random.split(key, num=num_steps)
        final, result = lax.scan(generate_new_gaussian, 0.0, keys)
        return final, result

    return gaussian_random_walk


def generate_new_gaussian(old_gaussian, key):
    new_gaussian = random.normal(key) + old_gaussian
    return new_gaussian, old_gaussian
