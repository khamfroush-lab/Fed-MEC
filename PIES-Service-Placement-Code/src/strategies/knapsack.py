import numpy as np

from typing import Any, Dict, Tuple
from .proposed import greedy_scheduling, value
from ..env.environment import Environment

"""
## Initialize model placement decision variable as an empty dictionary.
x = {}

for e in env.edges:
    
    ## First, let's assign values to each service by only considering its most
    ## beneficial model given the user requests.
    edge_users = set(env.covers[e])
    values = {(s, m): sum(value(u, s, m, env) for u in edge_users) 
                for s in env.services for m in env.models}

    SM = {}
    n_items = 0
    sorted_models_by_weight = sorted(values.keys(), key=lambda x: env.stor[x])
    for service_model in sorted_models_by_weight:
        SM[n_items] = service_model
        n_items += 1

    ## PART 1: Generate the values for the `sack` matrix to find the maximal value that can be
    ##         attained with our constraints in mind.
    sack = [[0 for x in range(env.stor[e] + 1)] for x in range(n_items + 1)] 
    for i in range(n_items + 1): 
        for w in range(env.stor[e] + 1): 
            if i == 0 or w == 0: 
                sack[i][w] = 0
            ## Added the second condition to avoid Python's negative indexing breaking the logic of the code.
            elif (env.stor[SM[i-1]] <= w) and (w-env.stor[SM[i-1]] >= 0): 
                sack[i][w] = max(values[SM[i-1]] + sack[i-1][int(w-env.stor[SM[i-1]])],  sack[i-1][w]) 
            else: 
                sack[i][w] = sack[i-1][w]

    ## PART 2: Once the sack values matrix is created, backtrack and identify the selected services.
    i = n_items - 1
    k = env.stor[e]
    used_storage = 0

    while i > 0 and k > 0:
        if (sack[i][int(k)] != sack[i-1][int(k)]) and (used_storage + env.stor[SM[i]] <= env.stor[e]):
            x[e, SM[i][0], SM[i][1]] = 1 # Mark the i-th service as "placed".
            k -= env.stor[SM[i]]
            used_storage += env.stor[SM[i]]
        i -= 1

## With the placement decision variable finished, return it.
return x
"""

def knapsack_placement(env: Environment) -> Dict[Any, int]:
    x = {}
    for e in env.edges:
        edge_users = set(env.covered_requests(e))
        values = {(s, m): sum(value(u, s, m, env) for u in edge_users)
                  for (s, m) in env.service_models}
        
        SM = {}
        sorted_models_by_weight = sorted(values.keys(), key=lambda k: env.storage_cost(*k))
        for index, service_model in enumerate(sorted_models_by_weight):
            SM[index] = service_model

        new_values = {index: values[s, m] for index, (s, m) in SM.items()}
        values = new_values

        # Generate values for the `sack` matrix to find the maximal value that can be
        # attained with the constraints in mind.
        '''
        storage_capacity = int(env.storage_capacity(e))
        sack = [[0 for _ in range(storage_capacity+1)] for _ in range(n_items+1)]
        for i in range(n_items+1):
            for w in range(storage_capacity+1):
                r_sm = env.storage_cost(*SM[i-1])
                if i == 0 or w == 0:
                    sack[i][w] = 0
                elif (r_sm <= w) and (w-r_sm >= 0):
                    option = values[SM[i-1]] + sack[i-1][int(w-r_sm)]
                    sack[i][w] = max(option, sack[i-1][w])
                else:
                    sack[i][w] = sack[i-1][w]
        '''
        R_e = int(env.storage_capacity(e))
        sack = np.zeros(shape=(len(SM)+1, R_e+1))
        for i in range(len(SM)+1):
            for w in range(R_e+1):
                if i == 0 or w == 0:
                    sack[i, w] = 0
                elif (env.storage_cost(*SM[i-1]) <= w) and (w-env.storage_cost(*SM[i-1]) >= 0):
                    r_sm = env.storage_cost(*SM[i-1])
                    sack[i, w] = max(values[i-1] + sack[i-1, w-int(r_sm)], 
                                     sack[i-1, w])
                else:
                    sack[i, w] = sack[i-1, w]

        '''
        # Once the sack values matrix is created, backtrack and identify the selected
        # service models.
        i = n_items - 1
        k = storage_capacity
        used_storage = 0
        
        while (i > 0) and (k > 0):
            if (sack[i][int(k)] != sack[i-1][int(k)]) and \
               (used_storage + env.storage_cost(*SM[i]) <= storage_capacity):
               s, m = SM[i][0], SM[i][1]
               x[e, s, m] = 1
               k -= env.storage_cost(s, m)
               used_storage += env.storage_cost(s, m)
            i -= 1
        '''

        w = R_e
        i = len(SM)
        while (i > 0) and (w > 0):
            if (sack[i, int(w)] != sack[i-1, int(w)]):
                s, m = SM[i-1][0], SM[i-1][1]
                x[e, s, m] = 1
                w -= env.storage_cost(s, m)
            i -= 1

    return x

def knapsack(env: Environment) -> Tuple[Dict, Dict]:
    x = knapsack_placement(env)
    y = greedy_scheduling(x, env)
    env.validate_decisions(x, y)
    return x, y