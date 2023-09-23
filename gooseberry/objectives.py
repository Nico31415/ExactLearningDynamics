from .environment import env

import jax.numpy as np


class MeanSquaredError:
    def __new__(cls):
        _ = env.add_objective()
        return cls.advance

    @staticmethod
    def advance(state, y_hat, y):
        loss = 0.5 * np.mean(np.sum((y_hat - y)**2, axis=1))
        return state, loss
