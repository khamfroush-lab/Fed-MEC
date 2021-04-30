#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import torch
import math

from utils.sampling import (
    mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_unequal,
    cifar_noniid, create_shared_dataset
)
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFashion_Mnist
from models.Fed import FedAvg, fed_avg
from models.test import test_img, test_img_user
import random
from mpl_toolkits import mplot3d
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

matplotlib.use('Agg')


def muhammed(
    network: torch.nn.Module,
    user_data_indices: Dict[int, Any],
    labels_counter: Dict[int, Any],
    args: Any,
    cost: Any,
    train_data: DataLoader,
    test_data: DataLoader,
    shared_data: DataLoader,
    **kwargs
) -> Tuple:
    global_data = defaultdict(list)
    global_eval_data = defaultdict(list)
    workers_index = defaultdict(list)
    client_participation_counter = defaultdict(int)

    # r1=1 and r2=4 are the best values based on Muhammed's results.
    r1 = kwargs.get("r1", 1)
    r2 = kwargs.get("r2", 4)

    comm_cost_total = []

    n_k = np.zeros(shape=(args.num_users))
    for i in range(len(user_data_indices)):
        n_k[i] = len(user_data_indices[i])

    pre_net_glob = copy.deepcopy(network)

    def compute_alpha_star(n_clients, r1, r2) -> int:
        fact = math.factorial
        coeff = math.exp(-(fact(r2)/fact(r1 - 1)) ** (1/(r2-r1+1)))
        return n_clients * coeff

    for comm_round in range(args.epochs):
        selected = set()
        comm_cost = []
        randomized_clients = list(user_data_indices.keys())
        random.shuffle(randomized_clients)

        network.eval()
        with torch.no_grad():
            sds_test_acc, sds_test_loss = test_img(
                network, shared_data, args
            )

        # _______________________________________________ #
        # =================== STAGE 1 =================== #
        alpha_star = compute_alpha_star(len(randomized_clients), r1, r2)
        acc_best = 0
        for m in range(int(alpha_star)):
            # Train client `m` using a copy of the global model and then test its
            # accuracy on the test data set. This is to find the "optimal" test threshold
            # value for client selection.
            trainer = LocalUpdate(args, train_data,
                                  user_data_indices[m])
            local_model, local_loss = trainer.train(
                net=copy.deepcopy(network).to(args.device))
            local_network = copy.deepcopy(network)
            local_network.load_state_dict(local_model)
            local_network.eval()

            acc_client, loss_client = test_img(
                local_network, datatest=test_data, args=args)
            comm_cost.append(cost[m])
            if acc_client > acc_best:
                acc_best = acc_client
            # selected[clients[m]] = False

        # _______________________________________________ #
        # =================== STAGE 2 =================== #
        set_best = set()
        num_best = 0
        R = max(int(args.frac * args.num_users), 1)
        for m in range(int(alpha_star), len(randomized_clients)):
            if num_best == R:
                continue  # "Rejects" the client m.
            elif (len(randomized_clients) - m) <= (R - num_best):
                c = randomized_clients[m]
                selected.add(c)
                set_best.add(c)
                num_best += 1
            else:
                # client data m
                # acc_client, loss_client = test_img_user(network, datatest=train_data, idxs=user_data_indices[m],
                #                                         args=args)
                acc_client, loss_client = test_img(
                    network, datatest=test_data, args=args)
                comm_cost.append(cost[m])
                if acc_client > acc_best:
                    c = randomized_clients[m]
                    selected.add(c)
                    set_best.add(c)
                    num_best += 1

        # _______________________________________________ #
        # =================== STAGE 3 =================== #
        # NOTE: Just use 1 to make the algorithm make sense for our setup.
        K = 1
        for _ in range(K):
            local_models, local_losses = [], []
            for client in selected:
                trainer = LocalUpdate(args, train_data,
                                      user_data_indices[client])
                local_model, local_loss = trainer.train(
                    net=copy.deepcopy(network).to(args.device))
                local_models.append(local_model)
                local_losses.append(local_loss)
                comm_cost.append(cost[client])
                client_participation_counter[client] += 1

            for n in range(args.num_users - len(local_models)):
                local_models.append(pre_net_glob.state_dict())
            new_weights = fed_avg(local_models, n_k)
            # new_weights = FedAvg(local_models)
            network.load_state_dict(new_weights)
            pre_net_glob = copy.deepcopy(network)

        # _______________________________________________ #
        # ================= DATA SAVING ================= #
        network.eval()
        with torch.no_grad():
            test_acc, test_loss = test_img(
                network, test_data, args
            )

        global_data["Round"].append(comm_round)
        global_data["C"].append(args.frac)
        global_data["Average Loss Train"].append(np.mean(local_losses))
        global_data["SDS Loss"].append(float(sds_test_loss))
        global_data["SDS Accuracy"].append(float(sds_test_acc))
        global_data["Workers Number"].append(int(len(selected)))
        global_data["Large Test Loss"].append(float(test_loss))
        global_data["Large Test Accuracy"].append(float(test_acc))
        global_data["Communication Cost"].append(sum(comm_cost))

        comm_cost_total.append(sum(comm_cost))

    # Calculate the percentage of each workers' participation.
    for client in client_participation_counter:
        client_participation_counter[client] /= args.epochs

    final_train_acc, final_train_loss = test_img(network, train_data, args)
    final_test_acc, final_test_loss = test_img(network, test_data, args)

    network.eval()
    with torch.no_grad():
        global_eval_data["C"].append(args.frac)
        global_eval_data["Test Loss"].append(float(final_test_loss))
        global_eval_data["Test Accuracy"].append(float(final_test_acc))
        global_eval_data["Train Loss"].append(float(final_train_loss))
        global_eval_data["Train Accuracy"].append(float(final_train_acc))
        global_eval_data["Communication Cost"].append(sum(comm_cost_total))
        global_eval_data["Total Rounds"].append(args.epochs)

    return global_eval_data, global_data
