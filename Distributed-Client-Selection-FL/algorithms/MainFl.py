#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_unequal, cifar_noniid, create_shared_dataset
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFashion_Mnist
from models.Fed import FedAvg, FedAvg_2, fed_avg
from models.test import test_img
import random
from mpl_toolkits import mplot3d
import json
from collections import defaultdict
from typing import Dict, Any


def mainFl(
    net_glob_mainFL: Any,
    dict_users_mainFL: Dict[int, Any],
    dict_labels_counter_mainFL,
    args,
    cost,
    dataset_train,
    dataset_test,
    small_shared_dataset
):
    """

    Args:
        net_glob_mainFL (torch.nn.Module): global model
        dict_users_mainFL (Dict: dict_users_mainFL[idx_user]): dict contains users data indexes, to access one user's data index,
        write dict_users_mainFL[idx_user]
        dict_labels_counter_mainFL: dict contains each users's labels total number, we do not use it right now
        args: all args. You can look for details in utils/options.py.
        cost: An array contains cost of sending locally updated models from users to server. We do not use it right now.
        dataset_train (torch dataset): Total train data set. We need it for train part, as in dict_users, we just have index of data.
        dataset_test (torch dataset): Total test dataset.
        small_shared_dataset (torch dataset): The small shared dataset that we just use it here for tracking and comparing with our
        algorithm, not for decision making.

    Returns:
        float(loss_test_final_main): final loss test over main test dataset
        dict_workers_index: index of selected workers in each round, we need it for use in other algorithms
        Final_LargeDataSetTest_MainFL: A dict contains macroscopic data to be saved after each total FL process (each C)
        data_Global_main: A dict contains microscopic data to be saved after each total FL process (each round)

    """

    data_Global_main = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                        "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [], "Communication Cost": []}
    Final_LargeDataSetTest_MainFL = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                     "Train Accuracy": [], "Total Rounds": [], "Communication Cost": []}

    # saving index of workers
    dict_workers_index = defaultdict(list)

    n_k = np.zeros(shape=(args.num_users))
    for i in range(len(dict_users_mainFL)):
        n_k[i] = len(dict_users_mainFL[i])
    # print(n_k)

    # Main FL

    # contains average loss over each clients' loss
    loss_train_mainFL = []
    # contains loss of
    Loss_local_each_global_total_mainFL = []
    Accuracy_local_each_global_total_mainFL = []
    # contains loss of each workers over small shared dataset in each round
    loss_workers_total_mainFL = np.zeros(shape=(args.num_users, args.epochs))
    label_workers_mainFL = {i: np.array(
        [], dtype='int64') for i in range(args.num_users)}

    #
    validation_test_mainFed = []
    acc_test, loss_test = test_img(net_glob_mainFL, dataset_test, args)
    workers_participation_main_fd = np.zeros((args.num_users, args.epochs))
    workers_percent_main = []

    net_glob_mainFL.eval()
    acc_test_final_mainFL, loss_test_final_mainFL = test_img(
        net_glob_mainFL, dataset_test, args)
    print("main fl initial loss is ", loss_test_final_mainFL)

    # while counter initialization
    iter_mainFL = 0

    # assign index to each worker in workers_mainFL arr
    workers_mainFL = []
    for i in range(args.num_users):
        workers_mainFL.append(i)

    temp_netglob_mainFL = copy.deepcopy(net_glob_mainFL)

    selected_clients_costs_total = []
    total_rounds_mainFL = 0

    pre_net_glob = copy.deepcopy(net_glob_mainFL)

    while iter_mainFL < (args.epochs):
        # print(f"iter {iter_mainFL} is started")
        selected_clients_costs_round = []
        w_locals_mainFL, loss_locals_mainFL = [], []
        m_mainFL = max(int(args.frac * args.num_users), 1)

        # selecting some clients randomly and save the index of them for use in other algorithms
        list_of_random_workers = random.sample(workers_mainFL, m_mainFL)
        # print("list of random workers is ", list_of_random_workers)
        for i in range(len(list_of_random_workers)):
            dict_workers_index[iter_mainFL].append(list_of_random_workers[i])

        # calculating and saving initial loss of global model over small shared dataset for just record
        x_mainFL = copy.deepcopy(net_glob_mainFL)
        x_mainFL.eval()
        acc_test_global_mainFL, loss_test_global_mainFL = test_img(
            x_mainFL, small_shared_dataset, args)
        Loss_local_each_global_total_mainFL.append(loss_test_global_mainFL)
        Accuracy_local_each_global_total_mainFL.append(acc_test_global_mainFL)
        # print("loss global is ", loss_test_global_mainFL)
        # print("accuracy global is ", acc_test_global_mainFL)
        workers_count_mainFL = 0
        for idx in list_of_random_workers:
            # start training each selected client
            # print("idx is ", idx)
            local_mainFL = LocalUpdate(
                args=args, dataset=dataset_train, idxs=dict_users_mainFL[idx])
            w_mainFL, loss_mainFL = local_mainFL.train(
                net=copy.deepcopy(net_glob_mainFL).to(args.device))

            # copy its updated weights
            w_locals_mainFL.append(copy.deepcopy(w_mainFL))
            # copy the training loss of that client
            loss_locals_mainFL.append(loss_mainFL)

            temp_netglob_mainFL.load_state_dict(w_mainFL)
            # test the locally updated model over small shared dataset and save its loss and accuracy for record
            temp_netglob_mainFL.eval()
            acc_test_local_mainFL, loss_test_local_mainFL = test_img(
                temp_netglob_mainFL, small_shared_dataset, args)
            # print("client loss is ", loss_test_local_mainFL)
            # print("accuracy of client is ", acc_test_local_mainFL)
            # loss_workers_total_mainFL[idx, iter_mainFL] = acc_test_local_mainFL
            # saving how many times each client is participating for just record
            workers_participation_main_fd[idx][iter_mainFL] = 1
            # saving total number of clients participated in that round (equal to C*N)
            workers_count_mainFL += 1
            selected_clients_costs_round.append(cost[idx])

        # Add others clients weights who did not participate
        # for i in range(args.num_users - len(list_of_random_workers)):
        #     w_locals_mainFL.append(pre_weights.state_dict())

        # update global weights
        # w_glob_mainFL = FedAvg(w_locals_mainFL)


        for n in range(args.num_users - m_mainFL):
            w_locals_mainFL.append(pre_net_glob.state_dict())
        # NOTE: Updated weights (@author Nathaniel).
        w_glob_mainFL = fed_avg(w_locals_mainFL, n_k)

        # copy weight to net_glob
        net_glob_mainFL.load_state_dict(w_glob_mainFL)
        # print("after ", net_glob_mainFL)

        # calculating average training loss
        # print(loss_locals_mainFL)
        loss_avg_mainFL = sum(loss_locals_mainFL) / len(loss_locals_mainFL)
        loss_train_mainFL.append(loss_avg_mainFL)
        # print(loss_avg_mainFL)

        # calculating test loss and accuracy over main large test dataset
        acc_test_round_mainfed, loss_test_round_mainfed = test_img(
            net_glob_mainFL, dataset_test, args)
        validation_test_mainFed.append(acc_test_round_mainfed)
        workers_percent_main.append(workers_count_mainFL / args.num_users)
        # calculating accuracy and loss over small shared dataset
        acc_test_final_mainFL, loss_test_final_mainFL = test_img(
            net_glob_mainFL, dataset_test, args)

        data_Global_main["Round"].append(iter_mainFL)
        data_Global_main["C"].append(args.frac)
        data_Global_main["Average Loss Train"].append(float(loss_avg_mainFL))
        data_Global_main["SDS Loss"].append(float(loss_test_global_mainFL))
        data_Global_main["SDS Accuracy"].append(float(acc_test_global_mainFL))
        data_Global_main["Workers Number"].append(float(workers_count_mainFL))
        data_Global_main["Large Test Loss"].append(
            float(loss_test_final_mainFL))
        data_Global_main["Large Test Accuracy"].append(
            float(acc_test_final_mainFL))
        data_Global_main["Communication Cost"].append(
            sum(selected_clients_costs_round))

        # TODO: This doesn't make sense?
        selected_clients_costs_total.append(sum(selected_clients_costs_round))

        iter_mainFL += 1
        # total_rounds_mainFL = iter_mainFL
        pre_net_glob = copy.deepcopy(net_glob_mainFL)

        # print(f"iter {iter_mainFL} is finished")

    # calculating the percentage of each workers participation
    workers_percent_final_mainFL = np.zeros(args.num_users)
    workers_name_mainFL = np.empty(args.num_users)
    for i in range(len(workers_participation_main_fd[:, 1])):
        workers_percent_final_mainFL[i] = sum(
            workers_participation_main_fd[i, :]) / args.epochs
        workers_name_mainFL[i] = i

    net_glob_mainFL.eval()
    # print("train test started")
    acc_train_final_main, loss_train_final_main = test_img(
        net_glob_mainFL, dataset_train, args)
    # print("train test finished")
    acc_test_final_main, loss_test_final_main = test_img(
        net_glob_mainFL, dataset_test, args)

    Final_LargeDataSetTest_MainFL["C"].append(args.frac)
    Final_LargeDataSetTest_MainFL["Test Loss"].append(
        float(loss_test_final_main))
    Final_LargeDataSetTest_MainFL["Test Accuracy"].append(
        float(acc_test_final_main))
    Final_LargeDataSetTest_MainFL["Train Loss"].append(
        float(loss_train_final_main))
    Final_LargeDataSetTest_MainFL["Train Accuracy"].append(
        float(acc_train_final_main))
    Final_LargeDataSetTest_MainFL["Communication Cost"].append(
        sum(selected_clients_costs_total))
    Final_LargeDataSetTest_MainFL["Total Rounds"].append(args.epochs)

    return float(loss_test_final_main), dict_workers_index, Final_LargeDataSetTest_MainFL, data_Global_main
