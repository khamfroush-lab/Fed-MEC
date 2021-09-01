import numpy as np

from numpy import random as rand

N_EDGES_FUNC = lambda: 10
N_USERS_FUNC = lambda: 1000
N_SERVICES_FUNC = lambda: 100
N_MODELS_MAX_FUNC = lambda: 10
MAX_DELAY = 5 # 100

# Changed 250 to 2500
COMM_E_FUNC = lambda: int(np.clip(rand.normal(250, 50), 1, float("inf")))
COMP_E_FUNC = lambda: int(np.clip(rand.normal(250, 50), 1, float("inf")))
STOR_E_FUNC = lambda: int(np.clip(rand.normal(250, 50), 1, float("inf"))) # Changed to 2500 from 250

COMM_S_FUNC = lambda: int(np.clip(rand.normal(50, 15), 1, float("inf")))
COMP_S_FUNC = lambda: int(np.clip(rand.normal(50, 15), 1, float("inf")))
STOR_S_FUNC = lambda: int(np.clip(rand.normal(25, 10), 1, float("inf"))) # Shrink this

ACC_R_FUNC = lambda: rand.triangular(0, 1, 1)
DEL_R_FUNC = lambda: int(rand.triangular(0, 0, 100))

ACC_S_FUNC = lambda: np.clip(rand.normal(0.65, 0.1), 0.0, 1.0)