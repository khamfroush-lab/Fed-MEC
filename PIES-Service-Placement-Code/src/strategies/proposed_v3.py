# -- coding: future_fstrings --
import math

from collections import defaultdict
from typing import Any, Dict, Tuple

from ..env.qos import *
from ..env.environment import Environment

def ReLU(x):
    return max(0, x)

def greedy_placement(env: Environment) -> Dict[Any, int]:
    x = {}

    placed_services = defaultdict(set)

    for e in env.edges:
        edge_users = set(env.covered_requests(e))
        encountered_models = set()
        satisfied_users = set()
        remaining_storage = env.storage_capacity(e)

        values = {
            (s, m): sum(QoS_coeff(u, s, m, env)
                        for u in edge_users)
            for (s, m) in set(env.service_models)
        }

        while True:
            # Essentially an argmax that does not consider already encountered models
            # (i.e., placed models or models that cannot be placed due to storage).
            s_opt, m_opt = next(iter(values.keys()))
            qos = float("-inf")
            for (s, m) in set(values.keys()) - encountered_models:
                if values[s, m] > qos:
                    s_opt, m_opt = s, m
                    qos = values[s, m]

            if env.storage_cost(s_opt, m_opt) <= remaining_storage:
                placed_services[s_opt].add(m_opt)
                x[e, s_opt, m_opt] = 1
                remaining_storage -= env.storage_cost(s_opt, m_opt)
                
                for m in env.models_for_service(s_opt):
                    if (s_opt, m) not in encountered_models:
                        values[s_opt, m] = sum(
                            ReLU(QoS_coeff(u, s_opt, m, env) - QoS_coeff(u, s_opt, m_opt, env))
                            for u in edge_users #- satisfied_users
                        )

            encountered_models.add((s_opt, m_opt))

            # Find the satisfied users.
            for u in edge_users:
                qos = QoS_coeff(u, s_opt, m_opt, env)
                if qos >= 1.0:
                    satisfied_users.add(u)

            # Break condition.
            if not (remaining_storage > 0) or \
                len(values) <= len(encountered_models) or \
                len(satisfied_users) == len(edge_users):
                break

    return x
    

def greedy_scheduling(x: Dict[Any, int], env: Environment) -> Dict[Any, int]:
    y = dict()
    for u in env.requests:
        e = env.covering_edge(u)
        s = env.req_service(u)
        choices = {m: QoS_coeff(u, s, m, env) 
                   for m in env.models_for_request(u)
                   if x.get((e, s, m), 0) == 1}
        if len(choices) > 0:
            m = max(choices, key=choices.get)
            y[u, m] = 1

    return y

def greedy_v3(env: Environment) -> Tuple[Dict, Dict]:
    x = greedy_placement(env)
    y = greedy_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y