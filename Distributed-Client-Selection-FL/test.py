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
from algorithms.MainFl import mainFl
from algorithms.ICC_algorithm import ICC_FL
from algorithms.proposed_G import Proposed_G1
from algorithms.Proposed_G2 import Proposed_G2
from algorithms.Cho import pi_pow_d
from algorithms.Muhammed import muhammed
from utils.helper_functions import merge
import time
from datetime import datetime
import os
import pandas as pd
import pickle

OUT_DIR = os.path.join("out", "data")
RUN_INFO_DIR = os.path.join("out", "runInfo")

def main():

    manualSeed = 1

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_DCFL = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users_DCFL, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users_DCFL, dict_labels_counter
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users_DCFL = cifar_iid(dataset_train, args.num_users)
            dict_users_mainFL = dict_users_DCFL
            dict_labels_counter_mainFL = dict()
            dict_labels_counter = dict()
        else:
            dict_users_DCFL, dict_labels_counter = cifar_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users_DCFL, dict_labels_counter
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True, transform=trans_fmnist)
        if args.iid:
            print("iid")
            dict_users_DCFL = mnist_iid(dataset_train, args.num_users)
        else:
            print("non iid")
            dict_users_DCFL, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users_DCFL, dict_labels_counter
    else:
        exit('Error: unrecognized dataset')


    img_size = dataset_train[0][0].shape

    # Small shared dataset
    test_ds, valid_ds_before = torch.utils.data.random_split(dataset_test, (9500, 500))
    small_shared_dataset = create_shared_dataset(valid_ds_before, 200)





    optimal_delay = 1.0


    # Start process for each fraction of c
    for c_counter in range(3, 3+1, 2):
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
            # net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
            # net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'cnn' and args.dataset == 'fmnist':
            net_glob = CNNFashion_Mnist(args=args).to(args.device)
            # net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        # Saving data
        data_Global_main = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                            "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                            "Communication Cost": []}
        Final_LargeDataSetTest_MainFL = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                         "Train Accuracy": [], "Total Rounds": [], "Communication Cost": []}

        data_Global_DCFL = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                            "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                            "Communication Cost": []}
        Final_LargeDataSetTest_DCFL = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                       "Train Accuracy": [],
                                       "Total Rounds": [], "Communication Cost": []}

        data_Global_G1 = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                          "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                          "Communication Cost": []}
        Final_LargeDataSetTest_G1 = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                     "Train Accuracy": [],
                                     "Total Rounds": [], "Communication Cost": []}

        data_Global_G2 = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                          "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                          "Communication Cost": []}
        Final_LargeDataSetTest_G2 = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                     "Train Accuracy": [],
                                     "Total Rounds": [], "Communication Cost": []}

        data_Global_Muhammed = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                                "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                                "Communication Cost": []}
        Final_LargeDataSetTest_Muhammed = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                           "Train Accuracy": [],
                                           "Total Rounds": [], "Communication Cost": []}

        data_Global_Cho = {"C": [], "Round": [], "Average Loss Train": [], "SDS Loss": [], "SDS Accuracy": [],
                                "Workers Number": [], "Large Test Loss": [], "Large Test Accuracy": [],
                                "Communication Cost": []}
        Final_LargeDataSetTest_Cho = {"C": [], "Test Accuracy": [], "Test Loss": [], "Train Loss": [],
                                           "Train Accuracy": [],
                                           "Total Rounds": [], "Communication Cost": []}

        net_glob.train()
        net_glob_mainFL = copy.deepcopy(net_glob)
        net_glob_G1 = copy.deepcopy(net_glob)
        net_glob_G2 = copy.deepcopy(net_glob)
        cost = np.random.rand(args.num_users)

        R_G1 = 5
        args.frac = (c_counter/10)


        # Main FL
        loss_main, dict_workers_index, Final_LargeDataSetTest_MainFL_temp, data_Global_main_temp = mainFl(net_glob_mainFL
                                                                                      , dict_users_mainFL,
                                                                    dict_labels_counter_mainFL, args, cost,
                                                                    dataset_train, dataset_test, small_shared_dataset)
        
        Final_LargeDataSetTest_MainFL = merge(Final_LargeDataSetTest_MainFL, Final_LargeDataSetTest_MainFL_temp)
        data_Global_main = merge(data_Global_main, data_Global_main_temp)


        # with open(os.path.join(OUT_DIR, f"dict_users_mainFL-C-{args.frac}-{args.dataset}.pkl"), 'wb') as file:
        #     pickle.dump(dict_users_mainFL, file)

        # with open(os.path.join(OUT_DIR, f"dict_users_mainFL-C-{args.frac}-{args.dataset}.pkl"), 'rb') as file:
        #     dict_users_mainFL = pickle.load(file)

        # with open(os.path.join(OUT_DIR, f"workers_index-C-{args.frac}-{args.dataset}.pkl"), 'wb') as file:
        #     pickle.dump(dict_workers_index, file)

        # with open(os.path.join(OUT_DIR, f"cost-C-{args.frac}-{args.dataset}.pkl"), 'wb') as file:
        #     pickle.dump(cost, file)

        # with open(os.path.join(OUT_DIR, f"cost-C-{args.frac}-{args.dataset}.pkl"), 'rb') as file:
        #     cost = pickle.load(file)

        # print(cost)

        # with open(os.path.join(OUT_DIR, f"GoalLoss-C-{args.frac}-{args.dataset}.pkl"), 'wb') as file:
        #     pickle.dump(loss_main, file)



        date = datetime.now()
        _dir = os.path.join(OUT_DIR, str(date.date()))
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        save_time = time.strftime("%Y%m%d-%H%M%S")


        Final_LargeDataSetTest_MainFL = pd.DataFrame.from_dict(Final_LargeDataSetTest_MainFL)
        data_Global_main = pd.DataFrame.from_dict(data_Global_main)
        Final_LargeDataSetTest_MainFL.to_csv(os.path.join
                                             (_dir, f"{save_time}-{args.dataset}-Final_LargeDataSetTest_MainFL.csv"))
        data_Global_main.to_csv(os.path.join
                                (_dir, f"{save_time}-{args.dataset}-data_Global_main.csv"))


        # Proposed G1
        Final_LargeDataSetTest_G1_temp, data_Global_G1_temp = Proposed_G1(net_glob_G1, dict_workers_index, dict_users_DCFL,
                                                                         dict_labels_counter_mainFL, args, cost,
                                                                         dataset_train, dataset_test,
                                                                         small_shared_dataset, loss_main, R_G1, optimal_delay)
        Final_LargeDataSetTest_G1 = merge(Final_LargeDataSetTest_G1, Final_LargeDataSetTest_G1_temp)
        data_Global_G1 = merge(data_Global_G1, data_Global_G1_temp)

        
        Final_LargeDataSetTest_G1 = pd.DataFrame.from_dict(Final_LargeDataSetTest_G1)
        data_Global_G1 = pd.DataFrame.from_dict(data_Global_G1)
        Final_LargeDataSetTest_G1.to_csv(os.path.join
                                         (_dir, f"{save_time}-{args.dataset}-Final_LargeDataSetTest_G1.csv"))
        data_Global_G1.to_csv(os.path.join
                              (_dir, f"{save_time}-{args.dataset}-data_Global_G1.csv"))
        
        print("G1 alg is done")

        # Muhammed
        # Final_LargeDataSetTest_muhammed_temp, data_Global_muhammed_temp = muhammed(net_glob_G2, dict_users_mainFL,
        #                                                                            dict_users_mainFL,
        #                                                                   args, cost,
        #                                                                  dataset_train, dataset_test,
        #                                                                  small_shared_dataset)

        # Final_LargeDataSetTest_Muhammed = merge(Final_LargeDataSetTest_Muhammed, Final_LargeDataSetTest_muhammed_temp)
        # data_Global_Muhammed = merge(data_Global_Muhammed, data_Global_muhammed_temp)


        # Final_LargeDataSetTest_Muhammed = pd.DataFrame.from_dict(Final_LargeDataSetTest_Muhammed)
        # data_Global_Muhammed = pd.DataFrame.from_dict(data_Global_Muhammed)
        # Final_LargeDataSetTest_Muhammed.to_csv(os.path.join
        #                                        (_dir,
        #                                         f"{save_time}-{args.dataset}-Final_LargeDataSetTest_Muhammed.csv"))
        # data_Global_Muhammed.to_csv(os.path.join
        #                             (_dir, f"{save_time}-{args.dataset}-data_Global_Muhammed.csv"))

        # print("Muhammed alg is done")

        # # Cho
        # d = (args.frac + 0.1) * args.num_users
        # Final_LargeDataSetTest_Cho_temp, data_Global_Cho_temp = pi_pow_d(net_glob_G2, dict_users_mainFL,
        #                                                                            dict_users_mainFL,
        #                                                                            args, cost,
        #                                                                            dataset_train, dataset_test,
        #                                                                            small_shared_dataset, d)

        # Final_LargeDataSetTest_Cho = merge(Final_LargeDataSetTest_Cho, Final_LargeDataSetTest_Cho_temp)
        # data_Global_Cho = merge(data_Global_Cho, data_Global_Cho_temp)

        # Final_LargeDataSetTest_Cho = pd.DataFrame.from_dict(Final_LargeDataSetTest_Cho)
        # data_Global_Cho = pd.DataFrame.from_dict(data_Global_Cho)
        # Final_LargeDataSetTest_Cho.to_csv(os.path.join
        #                                        (_dir,
        #                                         f"{save_time}-{args.dataset}-Final_LargeDataSetTest_Cho.csv"))
        # data_Global_Cho.to_csv(os.path.join
        #                             (_dir, f"{save_time}-{args.dataset}-data_Global_Cho.csv"))

        # print("Cho alg is done")






if __name__ == '__main__':
    main()