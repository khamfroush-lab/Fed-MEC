import pulp

from typing import Dict, Tuple

from ..env.environment import Environment
from ..env.qos import *

def ILP(env: Environment, **kwargs):
    # Initialize decision variables.
    esm = [(e, s, m) 
        for e in env.edges  
        for s in env.services 
        for m in env.models_for_service(s)
    ]
    um  = [(u, m) 
        for u in env.requests  
        for m in env.models_for_request(u)
    ]
    x = pulp.LpVariable.dicts("placement", esm, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("scheduling", um, cat=pulp.LpBinary)

    # Initialize the model and the objective function.
    model = pulp.LpProblem("PIES", pulp.LpMaximize)
    model += pulp.lpSum(y[u, m] * QoS_coeff(u, env.req_service(u), m, env)
                        for u in env.requests
                        for m in env.models_for_request(u))

    # Constraint 1: Service model constraing; user `u` can be served by, at most, 1 model.
    for u in env.requests:
        s = env.req_service(u)
        model += pulp.lpSum(y[u,m] for m in env.models_for_request(u)) <= 1

    # Constraint 2: Storage resource capacity constraint.
    for e in env.edges:
        model += pulp.lpSum(x[e,s,m] * env.storage_cost(s,m)
                            for s in env.services
                            for m in env.models_for_service(s)) <= env.storage_capacity(e)

    # Constraint 3: Ensures no user is served by an edge without a model for its service.
    for u in env.requests:
        e = env.covering_edge(u)
        s = env.req_service(u)
        for m in env.models_for_request(u):
            model += y[u,m] <= x[e,s,m]

    solver = pulp.PULP_CBC_CMD(**kwargs)
    model.solve(solver)
    return x, y

def optimal(env: Environment, **kwargs) -> Tuple[Dict, Dict]:
    x, y = ILP(env, **kwargs)
    x = {k: v.value() for k, v in x.items() if v.value() == 1}
    y = {k: v.value() for k, v in y.items() if v.value() == 1}
    env.validate_decisions(x, y)
    return x, y