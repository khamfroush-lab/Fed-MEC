# -- coding: future_fstrings --
import random

from tabulate import tabulate
from typing import Any, Dict, List, Set

from .configs.final_config import *


class Environment:

    edges: Dict
    requests: Dict
    services: Dict
    max_delay: Dict

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.edges = dict()
        self.requests = dict()
        self.services = dict()
        self.max_delay = dict()
        self.load_dict(config)

    def load_dict(self, config: Dict[Any, Any]) -> None:
        self.edges = config["edges"]
        self.requests = config["requests"]
        self.services = config["services"]
        self.service_models = {(s, m) for s in self.services for m in self.services[s]}
        self.max_delay = config["max_delay"]

    def validate_decisions(
        self,
        placement_decs: Dict[Any, int],
        scheduling_decs: Dict[Any, int]
    ) -> None:
        """Validates provided placement (x) and scheduling (y) decisions with this
           environment. Essentially, this method throws a ValueError only if the decisions
           invalidate the PIES constraints.

        Args:
            placement_decs (Dict[Any, int]): Placement decision variable.
            scheduling_decs (Dict[Any, int]): Scheduling decision variable.
        """
        for value in placement_decs.values():
            if value != 0 and value != 1:
                return False

        for value in scheduling_decs.values():
            if value != 0 and value != 1:
                return False

        consumed_storage = {e: 0 for e in self.edges}
        for (e, s, m) in placement_decs:
            if placement_decs.get((e, s, m), 0) == 1:
                consumed_storage[e] += self.storage_cost(s, m)

        for e in consumed_storage:
            if consumed_storage[e] > self.storage_capacity(e):
                raise ValueError(f"Storage capacity exceeded for edge cloud {e} "
                                 f"({consumed_storage[e]} > {self.storage_capacity(e)}).")

        for u in self.requests:
            total = sum(scheduling_decs.get((u,m), 0)
                        for m in self.models_for_request(u))
            if not total <= 1:
                raise ValueError(f"Scheduling constraint exceeded for request {u}")

    # ==================================================================================== #

    '''
    Edge cloud-related getter methods.
    '''
    def communication_capacity(self, edge_id: int) -> float:
        return self.edges[edge_id]["comm"]

    def computation_capacity(self, edge_id: int) -> float:
        return self.edges[edge_id]["comp"]

    def storage_capacity(self, edge_id: int) -> float:
        return self.edges[edge_id]["stor"]

    def covered_requests(self, edge_id: int) -> Set[int]:
        return self.edges[edge_id]["requests"]

    # ==================================================================================== #

    '''
    Service model-related getter methods.
    '''
    def communication_cost(self, service_id: int, model_id: int) -> float:
        return self.services[service_id][model_id]["comm_cost"]

    def computation_cost(self, service_id: int, model_id: int) -> float:
        return self.services[service_id][model_id]["comp_cost"]

    def storage_cost(self, service_id: int, model_id: int) -> float:
        return self.services[service_id][model_id]["stor_cost"]

    def accuracy(self, service_id: int, model_id: int) -> float:
        return self.services[service_id][model_id]["accuracy"]

    def models_for_service(self, service_id: int) -> Set[int]:
        return self.models[service_id]

    # ==================================================================================== #

    '''
    Request-related getter methods.
    '''
    def covering_edge(self, request_id: int) -> int:
        return self.requests[request_id]["covered_by"]

    def models_for_request(self, request_id: int) -> Set[int]:
        return self.models[self.req_service(request_id)]

    def req_accuracy(self, request_id: int) -> float:
        return self.requests[request_id]["req_accuracy"]

    def req_delay(self, request_id: int) -> float:
        return self.requests[request_id]["req_delay"]

    def req_service(self, request_id: int) -> int:
        return self.requests[request_id]["req_service"]

    # ==================================================================================== #

    @property
    def models(self) -> Dict[int, List[int]]:
        return self.services

    def __repr__(self) -> str:
        edge_columns = ["edge_id", "comm", "comp", "stor", "num_requests", "services"]
        edge_table = [
            [idx, e["comm"], e["comp"], e["stor"], len(e["requests"]), e["services"]]
            for idx, e in self.edges.items()
        ]
        edge_table_str = tabulate(edge_table, headers=edge_columns)
        return edge_table_str


def random_env(seed: int=None, **kwargs) -> Environment:
    """Generates a random instance of the PIES environment space.

    Args:
        seed (int, optional): Seed for generating random values. Defaults to None.

    Returns:
        Environment: Random environment
    """
    if seed is not None: random.seed(seed)

    def rand_edge() -> Dict[str, Any]:
        return {
            "comm": kwargs.get("comm_e_range_func", COMM_E_FUNC)(),
            "comp": kwargs.get("comp_e_range_func", COMP_E_FUNC)(),
            "stor": kwargs.get("stor_e_range_func", STOR_E_FUNC)(),
            "requests": set(),  # (requests that are served by edge cloud -- Set[int]).
            "services": dict(), # (services hosted by edge cloud -- Dict[int, Set[int]]).
        }

    def rand_request(n_services: int) -> Dict[str, Any]:
        return {
            "covered_by": None,
            "req_accuracy": kwargs.get("random_req_accuracy_func", ACC_R_FUNC)(),
            "req_delay": kwargs.get("random_req_delay_func", DEL_R_FUNC)(),
            "req_service": random.choice(range(n_services)),
        }

    def rand_service(n_models_max: int) -> Dict[str, Any]:
        def rand_model() -> Dict[str, Any]:
            return {
                "accuracy":  kwargs.get("rand_accuracy_func", ACC_S_FUNC)(),
                "comm_cost": kwargs.get("comm_s_func", COMM_S_FUNC)(),
                "comp_cost": kwargs.get("comp_s_func", COMP_S_FUNC)(),
                "stor_cost": kwargs.get("stor_s_func", STOR_S_FUNC)(),
            }

        n_models = random.randint(1, n_models_max)
        return {
            idx: rand_model() for idx in range(n_models)
        }

    # PART 1: Load specified number of each entity and the max number of possible models.
    n_edges = kwargs.get("n_edges_func", N_EDGES_FUNC)()
    n_requests = kwargs.get("n_requests_func", N_USERS_FUNC)()
    n_services = kwargs.get("n_services_func", N_SERVICES_FUNC)()
    n_models_max = kwargs.get("n_models_max_func", N_MODELS_MAX_FUNC)()
    max_delay = kwargs.get("max_delay", MAX_DELAY)

    # PART 2: Create the set of the three main entities.
    edges = {idx: rand_edge()  for idx in range(n_edges)}
    requests = {idx: rand_request(n_services) for idx in range(n_requests)}
    services = {idx: rand_service(n_models_max) for idx in range(n_services)}

    # PART 3: Assign requests to edges at random.
    for r in requests:
        e = random.choice(list(edges.keys()))
        edges[e]["requests"].add(r)
        requests[r]["covered_by"] = e

    return Environment(config={
        "edges": edges,
        "requests": requests,
        "services": services,
        "max_delay": max_delay
    })
