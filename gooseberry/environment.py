from types import SimpleNamespace as Ns

import jax

from functools import partial


class Environment(Ns):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.s = cls.states = {}
            cls.p = cls.parameters = {}
            cls._rng_key = jax.random.PRNGKey(1)
            cls.out_dim = None
        return cls._instance

    def create_rng_keys(self, n):
        keys = jax.random.split(self._rng_key, n + 1)
        self._rng_key = keys[0]
        return keys[1:]

    def assemble(self, seeds_n, root_seed=1):
        self._rng_key = jax.random.PRNGKey(root_seed)

        def seed_tree(seeds_):
            def seed_it(_):
                return seeds_
            return jax.tree_map(seed_it, self.p)

        def init_weights(fnct, seed):
            return fnct(seed)

        keys = jax.random.split(self._rng_key, seeds_n + 1)
        self._rng_key = keys[0]
        seeds = keys[1:]

        init_weights = jax.vmap(init_weights, (None, 1))
        self.p = jax.tree_map(init_weights, self.p, seed_tree(seeds.T))

        return self.s, self.p

    def add_dataset(self):
        pass

    def add_task(self, out_dim):
        self.out_dim = out_dim
        return self._add_entry("task")

    def add_network(self):
        return self._add_entry("network", True)

    def add_layer(self, out_dim):
        if isinstance(out_dim, (int, float)):
            out_dim = (int(out_dim), )
        in_dim = self.out_dim
        self.out_dim = out_dim
        return {}, {}, in_dim

    def add_objective(self):
        return self._add_entry("objective")

    def add_optimiser(self):
        return self._add_entry("optimiser")

    def add_trainer(self):
        return self._add_entry("trainer")

    def _add_entry(self, name, has_parameters=False):
        self.s[name] = {}
        if not has_parameters:
            return self.s[name]
        else:
            self.p[name] = {}
            return self.s[name], self.p[name]


env = Environment()
assemble = env.assemble
