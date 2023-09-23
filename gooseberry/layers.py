from .environment import env
from .nonlinearities import Identity
from .init import Normal, Constant

from functools import partial


class Network:
    def __new__(cls, layers):
        state, params = env.add_network()
        advances = []
        for i, layer in enumerate(layers):
            state["layer-" + str(i)], params["layer-" + str(i)], advance = layer
            advances.append(advance)
        return partial(cls.advance, layers=advances)

    @staticmethod
    def advance(state, params, x, layers):
        new_state = {}
        for i, layer in enumerate(layers):
            name = "layer-" + str(i)
            new_state[name], x = layer(state[name], params[name], x)
        return state, x


class Linear:
    def __new__(cls, out_dim, bias=True, nonlinearity=Identity(), weight_init=Normal(), bias_init=Constant(0.)):
        state, params, in_dim = env.add_layer(out_dim)
        if len(in_dim) == 1:
            in_dim = int(in_dim[0])

        params["w"] = weight_init((out_dim, in_dim))
        if bias:
            params["b"] = bias_init((out_dim, in_dim))

        return state, params, partial(cls.advance, bias=bias, f=nonlinearity)

    @staticmethod
    def advance(state, params, x, bias, f):
        if bias:
            return state, f(params["w"] @ x + params["b"])
        else:
            return state, f(params["w"] @ x)


"""
class MLP:
    def __new__(cls, layers):
        for layer in layers:
            if len(layer) < 5:

            out_dim, bias, activation_function,

    def __new__(cls, in_dim, layer_dims, bias=False):
        state, params = env.add_network()

        if isinstance(layer_dims, (int, float)):
            layer_dims = [layer_dims]

        keys = env.create_rng_keys(len(layer_dims))
        for i, (out_dim, key) in enumerate(zip(layer_dims, keys)):
            std = 1. / np.sqrt(in_dim)
            params["W" + str(i)] = (jax.random.normal(key, (out_dim, in_dim)) * std)
            if bias:
                params["b" + str(i)] = np.zeros(out_dim)
            in_dim = out_dim
        return partial(cls.advance, bias=bias, layers_n=len(layer_dims))

    @staticmethod
    def advance(state, params, x, bias, layers_n):
        for i in range(layers_n):
            b = params["b" + str(i)] if bias else 0.
            x = params["W" + str(i)] @ x + b

        return state, x
"""