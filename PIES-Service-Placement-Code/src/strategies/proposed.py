# -- coding: future_fstrings --
import math

from typing import Any, Dict, Tuple

from ..env.qos import *
from ..env.environment import Environment

def is_satisfied(user: int, service: int, model: int, env: Environment) -> bool:
    if env.req_service(user) != service:
        return False
    accuracy_satisfied = env.accuracy(service, model) >= env.req_accuracy(user)
    delay_satisfied = delay_fn(user, service, model, env) <= env.req_accuracy(user)
    return accuracy_satisfied or delay_satisfied

def value(u: int, s: int, m: int, env: Environment, x: Dict[Any, int]=None) -> float:
    if env.req_service(u) != s:
        return 0
    if x is not None:
        e = env.covering_edge(u)
        if x.get((e, s, m), 0) != 1:
            return 0
    return QoS_coeff(u, s, m, env)

def greedy_placement(env: Environment) -> Dict[Any, int]:
    x = dict()
    QOS = {
        (u, env.req_service(u), m): value(u, env.req_service(u), m, env)
        for u in env.requests for m in env.models_for_request(u)
    }

    for e in env.edges:
        edge_users = set(env.covered_requests(e))
        total_vals = dict()

        for u in edge_users:
            s = env.req_service(u)
            for m in env.models_for_request(u):
                if env.storage_cost(s, m) <= env.storage_capacity(e):
                    total_vals[s, m] = total_vals.get((s, m), 0) + \
                                    QOS.get((u, s, m), 0)/env.storage_cost(s, m)
                                    # min(QOS.get((u, s, m), 0), QOS.get((u,s,m), 0)/env.storage_cost(s,m))

        satisfied = set()
        used_storage = 0
        while len(satisfied) < len(edge_users):
            if len(total_vals) == 0:
                break

            s_opt, m_opt = max(total_vals.keys(), key=(lambda k: total_vals[k]))
            if used_storage+env.storage_cost(s_opt, m_opt) > env.storage_capacity(e):
                del total_vals[s_opt, m_opt]
                continue
            
            newly_satisfied = set()
            for user in edge_users - satisfied:
                if is_satisfied(user, s_opt, m_opt, env):
                    newly_satisfied.add(user)

            if len(newly_satisfied) == 0:
                user_opt = None
                val_opt = float("inf")
                for user in edge_users - satisfied:
                    # TODO: Needs to be reinvestigated given the change to the QoS func.
                    dist = math.sqrt(
                        (env.req_accuracy(user) - env.accuracy(s_opt, m_opt))**2 +
                        (env.req_delay(user) - delay_fn(user, s_opt, m_opt, env))**2
                    )
                    if dist <= val_opt:
                        user_opt = user
                        val_opt = dist
                newly_satisfied.add(user_opt)

            for u in newly_satisfied:
                s = env.req_service(u)
                for m in env.models_for_request(u):
                    if (s, m) in total_vals:
                        total_vals[s, m] -= QOS[u, s, m] / env.storage_cost(s, m)

            x[e, s_opt, m_opt] = 1
            used_storage += env.storage_cost(s_opt, m_opt)
            satisfied = satisfied.union(newly_satisfied)
            del total_vals[s_opt, m_opt]

    return x

def greedy_scheduling(x: Dict[Any, int], env: Environment) -> Dict[Any, int]:
    y = dict()
    for u in env.requests:
        e = env.covering_edge(u)
        s = env.req_service(u)
        choices = {
            m: value(u, s, m, env, x) 
            for m in env.models_for_request(u)
            if x.get((e, s, m), 0) == 1
        }
        if len(choices) > 0:
            m = max(choices, key=choices.get)
            y[u, m] = 1

    return y

def greedy(env: Environment) -> Tuple[Dict, Dict]:
    x = greedy_placement(env)
    y = greedy_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y