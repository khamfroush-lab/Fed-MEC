import numpy as np
import random

from typing import Any, Dict, Tuple

from .proposed import greedy_placement, greedy_scheduling
from ..env.environment import Environment

def random_placement(env: Environment) -> Dict[Any, int]:
    x = dict()
    for e in env.edges:
        used_storage = 0
        service_models = list(env.service_models)
        random.shuffle(service_models)
        for (s, m) in service_models:
            if used_storage == env.edges[e]["stor"]:
                break
            elif env.services[s][m]["stor_cost"] + used_storage <= env.edges[e]["stor"]:
                x[e, s, m] = 1
                used_storage += env.services[s][m]["stor_cost"]

    return x

def random_scheduling(x: Dict[Any, int], env: Environment) -> Dict[Any, int]:
    y = dict()
    for u in env.requests:
        e = env.covering_edge(u)
        s = env.req_service(u)
        available_models = [
            m for m in env.models[s] if x.get((e, s, m), 0) == 1
        ]
        if available_models:
            m = random.choice(available_models)
            y[u, m] = 1

    return y

def random_random(env: Environment) -> Tuple[Dict, Dict]:
    x = random_placement(env)
    y = random_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y

def random_greedy(env: Environment) -> Tuple[Dict, Dict]:
    x = random_placement(env)
    y = greedy_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y

def greedy_random(env: Environment) -> Tuple[Dict, Dict]:
    x = greedy_placement(env)
    y = random_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y