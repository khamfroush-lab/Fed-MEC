# -- coding: future_fstrings --
import argparse
import os

from datetime import datetime
from src.strategies.knapsack import knapsack
from src.strategies.optimal import optimal
from src.strategies.proposed_v2 import greedy_v2
from src.strategies.random import random_random
from src.strategies.simple_greedy import simple_greedy
from typing import Any, Dict

from numerical_experiments import main

SEED_COEFF = 123
OUT_DIR = os.path.join("out", "numerical_data", "key_results")
ALGORITHMS = {
    "Optimal": lambda env: optimal(env, msg=False, timeLimit=36120),
    "EGP": greedy_v2,
    "AGP": simple_greedy,
    "Knapsack": knapsack,
    "Random": random_random,
}

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_rounds", default=10, type=int)
    parser.add_argument("-l", "--lower_users", default=50, type=int)
    parser.add_argument("-u", "--upper_users", default=250, type=int)
    parser.add_argument("-s", "--step_users", default=50, type=int)    
    
    args = parser.parse_args()
    assert args.lower_users <= args.upper_users
    return args


if __name__ == "__main__":
    args = get_args()
    qos_data, req_data = main(args, ALGORITHMS, SEED_COEFF)

    date = datetime.now()
    filename = f"small_case.csv"
    qos_path = os.path.join(OUT_DIR, "qos_data", filename)
    req_path = os.path.join(OUT_DIR, "request_data", filename)

    qos_data.to_csv(qos_path)
    req_data.to_csv(req_path)


