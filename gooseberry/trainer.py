from .environment import env

import jax
import jax.numpy as np

from functools import partial


class Trainer:
    def __new__(cls, task, network, objective, optimiser):
        env.add_trainer()

        # Map over batch dimension
        network = jax.vmap(network, in_axes=(None, None, 0))

        forward = partial(cls._forward, network=network, objective=objective)
        v_and_g = jax.value_and_grad(forward, has_aux=True, argnums=1)

        # v_and_g = jax.vmap(v_and_g, in_axes=(None, None, 0, 0), out_axes=(None, None))
        # Map over seed dimension
        v_and_g = jax.vmap(v_and_g, in_axes=(None, 0, None, None))

        return jax.jit(partial(cls.advance, task=task, v_and_g=v_and_g, optimiser=optimiser))
        # return partial(cls.advance, task=task, v_and_g=v_and_g, optimiser=optimiser)

    @staticmethod
    def _forward(states, params, x, y, network, objective):
        new_states = {}
        new_states["network"], y_hat = network(states["network"], params["network"], x)
        new_states["objective"], loss = objective(states["objective"], y_hat, y)
        return loss, new_states

    @staticmethod
    def advance(states, params, task, v_and_g, optimiser):
        new_states = {}
        new_states["task"], x, y = task(states["task"])
        (loss, forward_states), grads = v_and_g(states, params, x, y)

        # Average over batch dimension
        # grads = jax.tree_map(partial(np.mean, axis=0), grads)
        # loss = jax.tree_map(partial(np.mean, axis=0), loss)


        new_states["optimiser"], new_params = optimiser(states["optimiser"], params, grads)

        return {**new_states, **forward_states}, new_params, loss


"""
class Trainer:
    def __new__(cls, task, network, objective, optimiser):
        env.add_trainer()

        forward = functools.partial(cls._forward, network=network, objective=objective)
        v_and_g = jax.value_and_grad(forward, has_aux=True, argnums=1)

        backward = functools.partial(cls._backward, v_and_g=v_and_g, optimiser=optimiser)
        backward = jax.vmap(backward, in_axes=(None, None, 0, 0))
        
        return functools.partial(cls.advance, task=task, backward=backward)

    @staticmethod
    def _forward(states, params, x, y, network, objective):
        new_states = {}
        new_states["network"], y_hat = network(states["network"], params["network"], x)
        new_states["objective"], loss = objective(states["objective"], y_hat, y)
        return loss, new_states
    
    @staticmethod
    def _backward(states, params, x, y, v_and_g, optimiser):
        new_states = {}
        (loss, forward_states), grads = v_and_g(states, params, x, y)
        new_states["optimiser"], new_params = optimiser(states["optimiser"], params, grads)
        print(loss.shape)
        for key, value in new_params["network"].items():
            print(key, value.shape)
        return {**new_states, **forward_states}, new_params, loss

    @staticmethod
    def advance(states, params, task, backward):
        print("W1", params["network"]["w_0"].shape)
        print("W2", params["network"]["w_1"].shape)
        new_states = {}
        new_states["task"], x, y = task(states["task"])
        states, new_params, loss = backward(states, params, x, y)

        return {**new_states, **states}, new_params, loss
"""
