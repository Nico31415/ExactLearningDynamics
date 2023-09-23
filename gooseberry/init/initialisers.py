from functools import partial

import jax
import jax.numpy as np


def FromFixedValue(value):
    def set_shape(shape):
        shape = np.asarray(shape)
        assert tuple(shape) == tuple(value.shape)

        def init(_):
            return np.asarray(value)

        return init
    return set_shape


def Constant(const):
    def set_shape(shape):
        shape = np.asarray(shape)

        def init(_):
            return np.ones(shape) * const

        return init
    return set_shape


def Normal(mean=0., std=1.):
    def set_shape(shape):
        shape = np.asarray(shape)

        def init(rng_key):
            return (jax.random.normal(rng_key, shape) * std) + mean

        return init
    return set_shape
