import time

from tabulate import tabulate

from src.env.environment import Environment, random_env
from src.env.qos import QoS_objective
from src.strategies.knapsack import knapsack
from src.strategies.optimal import optimal
from src.strategies.proposed import greedy
from src.strategies.random import random_random

env = random_env(
    # n_edges_func=lambda: 1,
    # n_requests_func=lambda: 100,
    # n_services_func=lambda: 25,
    # n_models_max_func=lambda: 10,
)

algorithms = {
    "Opt": lambda env: optimal(env, msg=False),
    "Gr": greedy,
    "KS": knapsack,
    "RR": random_random
}

headers = ["alg", "num_place_sm", "num_scheduled_sm", "expected_QoS", "alg_runtime"]
data =[]

for label, alg in algorithms.items():
    start = time.time()
    x, y = alg(env)
    end = time.time()
    data.append([
        label, len(x.items()), len(y.items()), QoS_objective(x, y, env), end-start
    ])

results = tabulate(data, headers=headers)
config = """N_EDGES_FUNC = lambda: 10
N_USERS_FUNC = lambda: 1000
N_SERVICES_FUNC = lambda: 100
N_MODELS_MAX_FUNC = lambda: random.randint(10, 20)

COMM_E_FUNC = lambda: random.randint(100, 200)
COMP_E_FUNC = lambda: random.randint(100, 200)
STOR_E_FUNC = lambda: random.randint(100, 200)

COMM_S_FUNC = lambda: random.randint(10, 50)
COMP_S_FUNC = lambda: random.randint(10, 50)
STOR_S_FUNC = lambda: random.randint(10, 50)

ACC_R_FUNC = lambda: random.uniform(0.5, 1.0)
DEL_R_FUNC = lambda: random.uniform(1.0, 5.0)
ACC_S_FUNC = lambda: random.uniform(0.75, 1.0)"""

with open("results.txt", "w") as f:
    f.write("[===== CONFIGURATION ======]\n")
    f.write(config)
    f.write("\n\n[===== RESULTS ======]\n")
    f.write(results)