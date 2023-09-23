from ..environment import env

import functools


class FullBatchLearning:
    def __new__(cls, dataset):
        out_dim = dataset(0)[0].shape
        state = env.add_task(out_dim)
        state["epoch"] = 0
        return functools.partial(cls.advance, dataset=dataset)

    @staticmethod
    def advance(state, dataset):
        x, y = dataset(None)
        return {"epoch": state["epoch"] + 1}, x, y
