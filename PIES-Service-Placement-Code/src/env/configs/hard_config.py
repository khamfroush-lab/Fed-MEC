import random

N_EDGES_FUNC = lambda: 10 
N_USERS_FUNC = lambda: 1000 
N_SERVICES_FUNC = lambda: 100 
N_MODELS_MAX_FUNC = lambda: random.randint(10, 20)
MAX_DELAY = 10.0

COMM_E_FUNC = lambda: random.randint(100, 200) 
COMP_E_FUNC = lambda: random.randint(100, 200) 
STOR_E_FUNC = lambda: random.randint(100, 200) 

COMM_S_FUNC = lambda: random.randint(10, 50) 
COMP_S_FUNC = lambda: random.randint(10, 50) 
STOR_S_FUNC = lambda: random.randint(10, 50) 

ACC_R_FUNC = lambda: random.uniform(0.5, 1.0)
DEL_R_FUNC = lambda: random.uniform(1.0, 5.0)
ACC_S_FUNC = lambda: random.uniform(0.75, 1.0)