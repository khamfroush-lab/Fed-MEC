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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_unequal,cifar_noniid, create_shared_dataset
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFashion_Mnist
from models.Fed import FedAvg
from models.test import test_img
import random
from mpl_toolkits import mplot3d
import json
from collections import defaultdict
from utils.helper_functions import compare_models, pnorm




def Proposed_G2(net_glob, dict_workers_index, dict_users_data, dict_labels_counter_mainFL, args, cost,
           dataset_train, dataset_test, valid_ds, loss_test_final_main, optimal_delay):


    data_Global_DCFL = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                        "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [], "Communication Cost": []}
    Final_LargeDataSetTest_DCFL = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                   "Train Accuracy": [],
                                   "Total Rounds": [], "Communication Cost": []}
    # copy weights
    # w_glob = net_glob.state_dict()

    temp = copy.deepcopy(net_glob)

    # training
    loss_train = []
    Loss_local_each_global_total = []
    selected_clients_costs_total = []

    loss_workers_total = np.zeros(shape=(args.num_users, 2 * args.epochs))

    workers_percent_dist = []
    workers_participation = np.zeros((args.num_users, 2 * args.epochs))
    workers = []
    for i in range(args.num_users):
        workers.append(i)

    counter_threshold_decrease = np.zeros(2 * args.epochs)
    Global_Accuracy_Tracker = np.zeros(2 * args.epochs)
    Global_Loss_Tracker = np.zeros(2 * args.epochs)
    threshold = 1.0

    Goal_Loss = float(loss_test_final_main)

    net_glob.eval()
    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
    while_counter = float(loss_test_final)
    iter = 0

    total_rounds_dcfl = 0
    while (while_counter + 0.01) >= Goal_Loss:
        selected_clients_costs_round = []
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        counter_threshold = 0

        x = net_glob
        x.eval()
        acc_test_global, loss_test_global = test_img(x, valid_ds, args)
        Loss_local_each_global_total.append(acc_test_global)
        Global_Accuracy_Tracker[iter] = acc_test_global
        Global_Loss_Tracker[iter] = loss_test_global
        threshold = loss_test_global
        if threshold <= 0.0:
            threshold = 1.0
        workers_count = 0


        temp_w_locals = []
        temp_workers_loss = np.empty(args.num_users)
        temp_workers_accuracy = np.empty(args.num_users)
        temp_workers_loss_test = np.empty(args.num_users)
        temp_workers_loss_difference = np.empty((args.num_users, 2))
        flag = np.zeros(args.num_users)

        list_of_random_workers_newfl = []
        if iter < (args.epochs):
            for key, value in dict_workers_index.items():
                if key == iter:
                    list_of_random_workers_newfl = dict_workers_index[key]
        else:
            list_of_random_workers_newfl = random.sample(workers, m)


        for idx in list_of_random_workers_newfl:
            initial_global_model = copy.deepcopy(net_glob).to(args.device)
            initial_global_model.eval()
            acc_test_local_initial, loss_test_local_initial = test_img(initial_global_model, valid_ds, args)


            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_data[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            temp_w_locals.append(copy.deepcopy(w))
            temp_workers_loss[idx] = copy.deepcopy(loss)

            temp.load_state_dict(w)
            temp.eval()

            acc_test_local_after, loss_test_local_after = test_img(temp, valid_ds, args)
            loss_workers_total[idx, iter] = loss_test_local_after
            temp_workers_accuracy[idx] = acc_test_local_after
            temp_workers_loss_test[idx] = loss_test_local_after
            temp_workers_loss_difference[idx, 0] = idx
            temp_workers_loss_difference[idx, 1] = loss_test_local_initial - loss_test_local_after

        while len(w_locals) < 1:
            index = 0
            for idx in list_of_random_workers_newfl:
                if workers_count >= m:
                    break
                elif temp_workers_loss_difference[idx, 1] >= threshold and flag[idx]==0 and cost[idx] <= optimal_delay:
                    w_locals.append(copy.deepcopy(temp_w_locals[index]))
                    loss_locals.append(temp_workers_loss[idx])
                    flag[idx] = 1
                    workers_count += 1
                    workers_participation[idx][iter] = 1
                    selected_clients_costs_round.append(cost[idx])
                index += 1
            if len(w_locals) < 1:
                threshold = threshold/2




        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        workers_percent_dist.append(workers_count/args.num_users)


        counter_threshold_decrease[iter] = counter_threshold
        print(iter, " round G2 fl finished")


        acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
        while_counter = loss_test_final


        data_Global_DCFL["Round"].append(iter)
        data_Global_DCFL["C"].append(args.frac)
        data_Global_DCFL["Average Loss Train"].append(loss_avg)
        data_Global_DCFL["SDS Accuracy"].append(Global_Accuracy_Tracker[iter])
        data_Global_DCFL["SDS Loss"].append(Global_Loss_Tracker[iter])
        data_Global_DCFL["Workers Number"].append(workers_count)
        data_Global_DCFL["Large Test Loss"].append(float(loss_test_final))
        data_Global_DCFL["Large Test Accuracy"].append(float(acc_test_final))
        data_Global_DCFL["Communication Cost"].append(sum(selected_clients_costs_round))

        selected_clients_costs_total.append(sum(selected_clients_costs_round))

        total_rounds_dcfl = iter
        iter += 1

    # plot workers percent of participating
    workers_percent_final = np.zeros(args.num_users)
    workers_name = np.empty(args.num_users)
    for i in range(len(workers_participation[:, 1])):
        workers_percent_final[i] = sum(workers_participation[i, :]) / (iter -1 )
        workers_name[i] = i


    # testing
    net_glob.eval()
    acc_train_final, loss_train_final = test_img(net_glob, dataset_train, args)
    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)


    Final_LargeDataSetTest_DCFL["C"].append(args.frac)
    Final_LargeDataSetTest_DCFL["Test Loss"].append(float(loss_test_final))
    Final_LargeDataSetTest_DCFL["Test Accuracy"].append(float(acc_test_final))
    Final_LargeDataSetTest_DCFL["Train Loss"].append(float(loss_train_final))
    Final_LargeDataSetTest_DCFL["Train Accuracy"].append(float(acc_train_final))
    Final_LargeDataSetTest_DCFL["Total Rounds"].append(int(total_rounds_dcfl))
    Final_LargeDataSetTest_DCFL["Communication Cost"].append(sum(selected_clients_costs_total))

    return Final_LargeDataSetTest_DCFL, data_Global_DCFL

