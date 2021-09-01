import numpy as np

from numpy import random as rand

N_EDGES_FUNC = lambda: 10
N_USERS_FUNC = lambda: 1000
N_SERVICES_FUNC = lambda: 100
N_MODELS_MAX_FUNC = lambda: 10

ACC_LAMBDA =  0.125
DEL_LAMBDA =  1.500
MAX_DELAY  = 10.000

# Changed 250 to 2500
COMM_E_FUNC = lambda: rand.randint(300, 600)
COMP_E_FUNC = lambda: rand.randint(300, 600)
STOR_E_FUNC = lambda: rand.randint(100, 200)

COMM_S_FUNC = lambda: rand.randint(15, 30)
COMP_S_FUNC = lambda: rand.randint(15, 30)
STOR_S_FUNC = lambda: rand.randint(10, 20)
ACC_S_FUNC = lambda: np.clip(rand.normal(0.65, 0.1), 0.0, 1.0)

ACC_R_FUNC = lambda: 1 - np.clip(rand.exponential(ACC_LAMBDA), 0, 1)
DEL_R_FUNC = lambda: np.clip(rand.exponential(DEL_LAMBDA), 0, MAX_DELAY)
