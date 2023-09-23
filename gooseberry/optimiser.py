from .environment import env

import jax
import jax.numpy as np


class GradientDescent:
    def __new__(cls, learning_rate):
        state = env.add_optimiser()
        state["learning_rate"] = learning_rate
        return cls.advance

    @staticmethod
    def advance(state, params, grads):
        def update_parameters(parameter, grad):
            return parameter - state["learning_rate"] * grad
        params = jax.tree_map(update_parameters, params, grads)

        return state, params
