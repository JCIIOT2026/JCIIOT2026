from robomimic.algo.algo import (
    register_algo_factory_func, algo_name_to_factory_func, algo_factory,
    Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy,
    REGISTERED_ALGO_FACTORY_FUNCS,
)

# Lazy-load individual algos to avoid requiring all dependencies at import time.
_ALGO_MODULES = {
    "bc": "robomimic.algo.bc",
    "bcq": "robomimic.algo.bcq",
    "cql": "robomimic.algo.cql",
    "iql": "robomimic.algo.iql",
    "gl": "robomimic.algo.gl",
    "hbc": "robomimic.algo.hbc",
    "iris": "robomimic.algo.iris",
    "td3_bc": "robomimic.algo.td3_bc",
    "diffusion_policy": "robomimic.algo.diffusion_policy",
}

_original_algo_factory = algo_factory


def _lazy_algo_factory(algo_name, *args, **kwargs):
    if algo_name in _ALGO_MODULES and algo_name not in REGISTERED_ALGO_FACTORY_FUNCS:
        import importlib
        importlib.import_module(_ALGO_MODULES[algo_name])
    return _original_algo_factory(algo_name, *args, **kwargs)


import sys as _sys
_sys.modules[__name__].algo_factory = _lazy_algo_factory
