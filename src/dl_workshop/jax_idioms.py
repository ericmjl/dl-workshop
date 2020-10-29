from jax import random, lax


def make_gaussian_random_walk_func(num_steps):
    def gaussian_random_walk(key):
        keys = random.split(key, num=num_steps)
        final, result = lax.scan(generate_new_gaussian, 0.0, keys)
        return final, result

    return gaussian_random_walk


def generate_new_gaussian(old_gaussian, key):
    new_gaussian = random.normal(key) + old_gaussian
    return new_gaussian, old_gaussian
