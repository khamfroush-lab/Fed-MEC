# -- coding: future_fstrings --
from typing import Any, Dict, List, Set, Tuple
from ..env.qos import *
from ..env.environment import Environment
from .proposed import greedy_scheduling

def get_valid_models(
    edge_id: int,
    remaining_storage: float,
    env: Environment
) -> Set[Tuple]:
    """Returns the set of models that can be placed without exceeding the remaining
        storage capacity.

    Args:
        edge_id (int): ID of the edge cloud.
        remaining_storage (float): Amount of storage remaining.
        env (Environment): The MEC environment.

    Returns:
        Set[Tuple]: The set of models that can be placed.
    """
    return set([(edge_id, s, m) 
                for (s, m) in env.service_models 
                if env.storage_cost(s, m) <= remaining_storage])


def placement_objective(placements: Set, env: Environment) -> float:
    """Get the QoS objective given a set of placement decisions, using optimal/greedy
        model scheduling.

    Args:
        placements (Set): Placement decisions.
        env (Environment): The MEC environment.

    Returns:
        float: QoS objective value.
    """
    placements_dict = {(e, s, m): 1 for (e, s, m) in placements}
    scheduling_dict = greedy_scheduling(placements_dict, env)
    return QoS_objective(placements_dict, scheduling_dict, env)


def simple_greedy(env: Environment) -> Dict[Any, int]:
    placements = set()
    for e in env.edges:
        remaining_storage = env.storage_capacity(e)
        models = get_valid_models(e, remaining_storage, env) - placements
        while models:
            max_val = 0
            _, s_opt, m_opt = next(iter(models))
            for (_, s, m) in models:
                val = placement_objective(placements.union({(e, s, m)}), env)
                if val > max_val:
                    max_val = val
                    s_opt, m_opt = s, m

            placements.add((e, s_opt, m_opt))
            remaining_storage -= env.storage_cost(s_opt, m_opt)

            models_to_prune = set((e, s_opt, m_opt))
            for (e, s, m) in models:
                r_sm = env.storage_cost(s, m)
                if r_sm > remaining_storage:
                    models_to_prune.add((e, s, m))
            models = models - models_to_prune

    x = {(e, s, m): 1 for (e, s, m) in placements}
    y = greedy_scheduling(x, env)
    return x, y