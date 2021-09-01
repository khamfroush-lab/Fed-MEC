# -- coding: future_fstrings --
import math
import numpy as np
import random

from typing import Any, Dict, Tuple

from ..env.qos import *
from ..env.environment import Environment
from .proposed import greedy_scheduling

def Omega(placement: Dict[Tuple[int, int, int], int], env: Environment) -> float:
    """This implements the $\Omega(S)$ function used for Algorithm 2 in He et al.'s 2018
       paper on joint service placement and request scheduling. Essentially, it takes a
       placement decision variable and then generates a scheduling decision that provides
       the *optimal* objective value and returns that objective value.

       Upon some evaluation, this algorithm does not work (or make sense) for our problem.
       This is due to it enforcing strict computation and communication capacity 
       constraints (in addition to storage).

    Args:
        placement (Dict[Tuple[int, int, int], int]): Placement decision variable.
        env (Environment): The environment variable under consideration.

    Returns:
        float: Optimal objective value given the provided placement decision.
    """
    scheduling = greedy_scheduling(placement, env)
    return QoS_objective(placement, scheduling, env)
    

def feasible_placements(x: Dict, env: Environment):
    arr = []
    for e in env.edges:
        for s in env.services:
            for m in env.models_for_service(s):
                cond1 = x.get((e,s,m), 0) == 0
                cond2 = sum(val*env.storage_cost(_s,_m) for (_,_s,_m), val in x.items()) \
                        < env.storage_capacity(e) - env.storage_cost(s, m)
                if cond1 and cond2:
                    arr.append((e, s, m))
                   
    return arr

    # return [(e, s, m)
    #         for e in env.edges
    #         for s in env.services
    #         for m in env.models_for_service(s)
    #         if x.get((e,s,m), 0) == 0 and \
    #             sum(x.get((e,s,m), 0) * env.storage_cost(s, m) for e in env.edges) 
    #             < env.storage_capacity(e)]


def GSP_GRS(env: Environment) -> Dict[Any, int]:
    x = dict()
    y = dict()
    res_w = dict()
    res_k = dict()
    unserved = dict()

    for e in env.edges:
        res_w[e] = env.computation_capacity(e)
        res_k[e] = env.communication_capacity(e)
        for s in env.services:
            unserved[s, e] = set([u for u in env.covered_requests(e) 
                                  if env.req_service(u) == s])
    
    phi = feasible_placements(x, env)
    iteration = 0
    while phi:
        # print(f"He 2018: iteration #{iteration}")
        # print(f"phi -> {len(phi)}\n")

        # TODO: This is incorrect because "phi" is not the same size as that for argmax
        rank = {(e, s, m): min(res_w[e], 
                               sum(min(len(unserved[s,e_]), res_k[e]) 
                                   for e_ in env.edges))
                for (e, s, m) in phi}
        e_opt, s_opt, m_opt = min(rank, key=rank.get)
        # index = np.argmax([min(res_w[e], sum(min(len(unserved[s, e_]), res_k[e]) 
        #                   for e_ in env.edges))
        #                   for (e, s, m) in phi])
        # e_opt, s_opt, m_opt = phi[index]
        
        x[e_opt, s_opt, m_opt] = 1
        o_opt = min(res_w[e_opt], sum(min(len(unserved[s_opt, e]), res_k[e]) 
                                      for e in env.edges))
        o_opt = int(o_opt)
        if o_opt == 0:
            print("Oh nooooo...!")
            break

        res_w[e_opt] -= o_opt
        for e in env.edges:
            o = min(o_opt, min(len(unserved[s_opt, e_opt]), res_k[e]))
            o = int(o)
            res_k[e] -= o
            if o > len(unserved[s_opt, e]):
                user_subset = set(unserved[s_opt, e])
            else:
                user_subset = set(random.sample(unserved[s_opt, e], o))
            for u in user_subset:
                y[u, m_opt] = 1
            unserved[s_opt, e] -= user_subset
            o_opt -= o
            if o_opt == 0:
                break 

        phi = feasible_placements(x, env)
        iteration += 1

    env.validate_decisions(x, y)

    print(f"He 2018: Number of placed models: {sum(x.values())}, number of scheduled requests: {sum(y.values())}")
    return x, y
