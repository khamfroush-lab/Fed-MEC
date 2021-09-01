DEFAULT_HOST = "localhost"
DEFAULT_PORT = 21567
DEFAULT_BUFF = 1024

ID_2_MODEL = {
    0: "AlexNet",
    1: "DenseNet",
    2: "GoogLeNet",
    3: "MobileNet",
    4: "ResNet",
    5: "SqueezeNet",
}

STORE_DIV = 1e6


SERVICES = {
    # Image Classification
    0: {
        # AlexNet
        0: {
            "accuracy":  0.56522,
            "comm_cost": 1,
            "comp_cost": 129257051.19,
            "stor_cost": 1, 
            "comp_delay": 0.04094290733337402,
        },
        # DenseNet
        1: {
            "accuracy":  0.77138,
            "comm_cost": 1,
            "comp_cost": 1288675947.21,
            "stor_cost": 1,
            "comp_delay": 0.4718599319458008,
        },
        # GoogLeNet
        2: {
            "accuracy":  0.69778,
            "comm_cost": 1,
            "comp_cost": 367018478.63,
            "stor_cost": 1, 
            "comp_delay": 0.12846112251281738,
        },
        # MobileNet
        3: {
            "accuracy":  0.71878,
            "comm_cost": 1,
            "comp_cost": 158662135.32,
            "stor_cost": 1, 
            "comp_delay": 0.05797386169433594,
        },
        # ResNet
        4: {
            "accuracy":  0.69758,
            "comm_cost": 1,
            "comp_cost": 267029373.18,
            "stor_cost": 1, 
            "comp_delay": 0.08414006233215332,
        },
        # SqueezeNet
        5: {
            "accuracy":  0.58092,
            "comm_cost": 1,
            "comp_cost": 213847898.91,
            "stor_cost": 1,
            "comp_delay": 0.07048892974853516,
        },
    },
}