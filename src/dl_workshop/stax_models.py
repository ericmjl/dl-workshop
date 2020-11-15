from jax import lax
import jax.numpy as np


def step(i, state, dlossfunc, get_params, update, model, x, y_true):
    params = get_params(state)
    g = dlossfunc(params, model, x, y_true)
    state = update(i, g, state)
    return state


def make_scannable_step(stepfunc):
    def scannable_step(previous_state, iteration):
        new_state = stepfunc(iteration, previous_state)
        return new_state, previous_state

    return scannable_step


def make_training_start(params_initializer, state_initializer, scanfunc, n_steps):
    def train_one_start(key):
        output_shape, params = params_initializer(key)
        initial_state = state_initializer(params)
        final_state, states_history = lax.scan(
            scanfunc, initial_state, np.arange(n_steps)
        )
        return final_state, states_history

    return train_one_start
