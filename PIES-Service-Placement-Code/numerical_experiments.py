# -- coding: future_fstrings --
import argparse
import numpy.random as rand
import numpy as np
import pandas as pd
import os
import random
import time

from collections import defaultdict
from datetime import datetime
from src.env.environment import random_env
from src.env.qos import QoS_coeff, QoS_objective, delay_fn, num_req_satisfied, num_partially_req_satisfied, quality_of_accuracy, quality_of_delay
from src.strategies.knapsack import knapsack
from src.strategies.optimal import optimal
from src.strategies.proposed_v2 import greedy_v2
from src.strategies.random import random_random
from src.strategies.simple_greedy import simple_greedy
from tqdm import tqdm
from typing import Any, Callable, Dict, Tuple


SEED_COEFF = 123
ALGORITHMS = {
    "Optimal": lambda env: optimal(env, msg=False),
    "EGP": greedy_v2,
    "AGP": simple_greedy,
    "Knapsack": knapsack,
    "Random": random_random,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_rounds", default=10, type=int)
    parser.add_argument("-l", "--lower_users", default=100, type=int)
    parser.add_argument("-u", "--upper_users", default=1000, type=int)
    parser.add_argument("-s", "--step_users", default=100, type=int)  
    args = parser.parse_args()
    assert args.lower_users <= args.upper_users
    return args


def num_served(y: Dict[Any, int]):
    return sum(y.values())


def main(
    args: argparse.Namespace, 
    algorithms: Dict[str, Callable]=ALGORITHMS, 
    seed_coeff: int=SEED_COEFF
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qos_data = defaultdict(list)
    req_data = defaultdict(list)
    n_rounds = args.n_rounds
    num_requests = list(range(args.lower_users, args.upper_users+1, args.step_users))
    make_random_env = lambda seed, n_req=1000: random_env(
        seed=seed*seed_coeff,
        n_requests_func=lambda: n_req,
    )

    pbar = tqdm(total=n_rounds * len(algorithms) * len(num_requests))
    for alg_label, alg_func in algorithms.items():
        pbar.set_description(f"Alg={alg_label}")
        for n_req in num_requests:
            for r in range(n_rounds):
                env = make_random_env(seed=r, n_req=n_req)
                start_time = time.time()
                x, y = alg_func(env)
                end_time = time.time()

                qos_data["round"].append(r)
                qos_data["algorithm"].append(alg_label)
                qos_data["QoS"].append(QoS_objective(x, y, env))
                qos_data["runtime"].append(end_time-start_time)
                qos_data["n_requests"].append(n_req)
                qos_data["n_served"].append(num_served(y))
                qos_data["n_satisfied"].append(num_req_satisfied(env, y))
                qos_data["n_partial_satisfied"].append(num_partially_req_satisfied(env, y))

                for u in env.requests:
                    s = env.req_service(u)
                    for m in env.models_for_request(u):
                        if y.get((u, m), 0) == 1:
                            req_data["round"].append(r)
                            req_data["algorithm"].append(alg_label)
                            req_data["QoA"].append(quality_of_accuracy(u, s, m, env))
                            req_data["QoD"].append(quality_of_delay(u, s, m, env))
                            req_data["QoS"].append(QoS_coeff(u, s, m, env))
                            req_data["runtime"].append(end_time-start_time)
                            req_data["n_requests"].append(n_req)
                            req_data["req_id"].append(u)
                            req_data["req_service"].append(s)
                            req_data["req_acc"].append(env.req_accuracy(u))
                            req_data["req_delay"].append(env.req_delay(u))
                            req_data["served_acc"].append(env.accuracy(s, m))
                            req_data["served_delay"].append(delay_fn(u, s, m, env))

                pbar.update()

    qos_data = pd.DataFrame.from_dict(qos_data)
    req_data = pd.DataFrame.from_dict(req_data)
    return qos_data, req_data


if __name__ == "__main__":
    args = get_args()
    qos_data, req_data = main(args)

    date = datetime.now()
    qos_dir = os.path.join("out", "numerical_data", "qos_data", str(date.date()))
    if not os.path.exists(qos_dir):
        os.makedirs(qos_dir)
    df = pd.DataFrame.from_dict(qos_data)
    df.to_csv(os.path.join(qos_dir, f"{date.time()}.csv"))

    req_dir = os.path.join("out", "numerical_data", "request_data", str(date.date()))
    if not os.path.exists(req_dir):
        os.makedirs(req_dir)
    df = pd.DataFrame.from_dict(req_data)
    df.to_csv(os.path.join(req_dir, f"{date.time()}.csv"))