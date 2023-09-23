import jax.numpy as np


class Identity:
    def __new__(cls):
        return cls.advance

    @staticmethod
    def advance(x):
        return x


class Tanh:
    def __new__(cls):
        return cls.advance

    @staticmethod
    def advance(x):
        return np.tanh(x)


class ReLU:
    def __new__(cls):
        return cls.advance

    @staticmethod
    def advance(x):
        return x * (x > 0)
