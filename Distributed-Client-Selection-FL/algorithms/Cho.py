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

from heapq import nlargest
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


def pi_pow_d(
    network: torch.nn.Module,
    user_data_indices: Dict[int, Any],
    labels_counter: Dict[int, Any],
    args: Any,
    cost: Any,
    train_data: DataLoader,
    test_data: DataLoader,
    shared_data: DataLoader,
    d,
    **kwargs
) -> Tuple:
    assert int(args.frac * args.num_users) <= d <= args.num_users

    global_data = defaultdict(list)
    global_eval_data = defaultdict(list)
    workers_index = defaultdict(list)
    # client_participation_counter = defaultdict(list)
    comm_cost_total = []
    n_k = np.zeros(shape=(args.num_users))
    for i in range(len(user_data_indices)):
        n_k[i] = len(user_data_indices[i])

    for comm_round in range(args.epochs):
        network.eval()
        with torch.no_grad():
            sds_test_acc, sds_test_loss = test_img(
                network, shared_data, args
            )

        ##
        # PART 1: Sample the candidate client set.
        ##
        comm_cost = []
        num_selected = max(1, int(d))
        selected = random.sample(list(user_data_indices.keys()), num_selected)

        ##
        # PART 2: Estimate local losses.
        ##
        local_losses = {}
        with torch.no_grad():
            for client in selected:
                _, local_test_loss = test_img_user(copy.deepcopy(network).to(args.device),
                                                   train_data, user_data_indices[client],
                                                   args)
                local_losses[client] = local_test_loss
                # I think this part should be removed
                # comm_cost.append(cost[m])

        ##
        # PART 3: Select highest loss clients.
        ##
        m = max(1, int(args.frac * args.num_users))
        top_m_clients = nlargest(m, local_losses, key=local_losses.get)
        local_models, local_losses = [], []
        for client in top_m_clients:
            trainer = LocalUpdate(args, train_data,
                                  user_data_indices[client])
            local_model, local_loss = trainer.train(
                net=copy.deepcopy(network).to(args.device))
            local_models.append(local_model)
            local_losses.append(local_loss)
            comm_cost.append(cost[client])
            # client_participation_counter[client] += 1

        for _ in range(args.num_users - len(local_models)):
            local_models.append(copy.deepcopy(network.state_dict()))
        new_weights = fed_avg(local_models, n_k)
        network.load_state_dict(new_weights)

        ##
        # PART 4: Data saving.
        ##
        network.eval()
        with torch.no_grad():
            test_acc, test_loss = test_img(
                network, shared_data, args
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

    # # Calculate the percentage of each workers' participation.
    # for client in client_participation_counter:
    #     client_participation_counter[client] /= args.epochs

    final_train_acc, final_train_loss = test_img(network, train_data, args)
    final_test_acc, final_test_loss = test_img(network, test_data, args)


    # print(comm_cost_total)
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
