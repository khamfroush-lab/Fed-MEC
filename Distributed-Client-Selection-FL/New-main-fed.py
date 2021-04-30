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
from utils.helper_functions import pnorm_2



def main():
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
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users, dict_labels_counter
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users, dict_labels_counter = cifar_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users, dict_labels_counter
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist', train=False, download=True, transform=trans_fmnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
            dict_users_mainFL, dict_labels_counter_mainFL = dict_users, dict_labels_counter
    else:
        exit('Error: unrecognized dataset')


    img_size = dataset_train[0][0].shape

    acc_full_distributed = []
    acc_full_main = []
    loss_full_ditributed = []
    loss_full_main = []

    SD_acc_full_distributed = []
    SD_acc_full_main = []
    SD_loss_full_ditributed = []
    SD_loss_full_main = []

    workers_percent_full_distributed = []
    workers_percent_full_main = []
    variable_start = 0.1
    variable_end = 1.0
    while_counter = 0.1
    counter_array = []
    Accuracy_Fraction = []
    Workers_Fraction = []

    accuracy_fraction_each_round_newFL = 0
    workers_fraction_each_round_newFL = 0
    accuracy_fraction_each_round_mainFL = 0
    workers_fraction_each_round_mainFL = 0

    data_main = {}
    data_DCFL = {}
    data_Global_main = {"C": [], "Round":[], "Average Loss Train": [], "Average Loss Test": [], "Accuracy Test": [],
                        "Workers Number": [], "Large Test Loss":[], "Large Test Accuracy":[]}
    data_Global_DCFL = {"C": [], "Round":[], "Average Loss Train": [], "Average Loss Test": [], "Accuracy Test": [],
                        "Workers Number": [], "Large Test Loss":[], "Large Test Accuracy":[]}
    Final_LargeDataSetTest_DCFL = {"C":[], "Test Accuracy":[], "Test Loss":[], "Train Loss":[], "Train Accuracy":[],
                                   "Total Rounds":[]}
    Final_LargeDataSetTest_MainFL = {"C":[], "Test Accuracy": [], "Test Loss": [], "Train Loss": [], "Train Accuracy":[]}



    # build model
    args.frac = variable_start

    test_ds, valid_ds_before = torch.utils.data.random_split(dataset_test, (9500, 500))
    valid_ds = create_shared_dataset(valid_ds_before, 200)

    #while variable_start <= variable_end:
    for c_counter in range(1, 11, 3):
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
            net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
            net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'cnn' and args.dataset == 'fmnist':
            net_glob = CNNFashion_Mnist(args=args).to(args.device)
            net_glob_mainFL = copy.deepcopy(net_glob)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        counter_array.append((c_counter/10))
        args.frac = (c_counter/10)

        ######saving index of workers
        dict_workers_index = defaultdict(list)


        #############Main FL

        w_glob_mainFL = net_glob_mainFL.state_dict()

        loss_train_mainFL = []
        # cv_loss_2, cv_acc_2 = [], []
        # val_loss_pre_2, counter_2 = 0, 0
        # net_best_2 = None
        # best_loss_2 = None
        # val_acc_list_2, net_list_2 = [], []

        Loss_local_each_global_total_mainFL = []
        Accuracy_local_each_global_total_mainFL = []

        loss_workers_total_mainFL = np.zeros(shape=(args.num_users, args.epochs))
        label_workers_mainFL = {i: np.array([], dtype='int64') for i in range(args.num_users)}

        validation_test_mainFed = []
        acc_test, loss_test = test_img(net_glob_mainFL, dataset_test, args)
        workers_participation_main_fd = np.zeros((args.num_users, args.epochs))
        workers_percent_main = []

        # for iter in range(args.epochs):
        net_glob_mainFL.eval()
        acc_test_final_mainFL, loss_test_final_mainFL = test_img(net_glob_mainFL, dataset_test, args)
        while_counter_mainFL = loss_test_final_mainFL
        iter_mainFL = 0

        workers_mainFL = []
        for i in range(args.num_users):
            workers_mainFL.append(i)

        temp_netglob_mainFL = net_glob_mainFL

        while iter_mainFL < (args.epochs/2):

            data_main['round_{}'.format(iter_mainFL)] = []
            # data_Global_main['round_{}'.format(iter)] = []
            # print("round started")
            Loss_local_each_global_mainFL = []
            loss_workers_mainFL = np.zeros((args.num_users, args.epochs))
            w_locals_mainFL, loss_locals_mainFL = [], []
            m_mainFL = max(int(args.frac * args.num_users), 1)
            idxs_users_mainFL = np.random.choice(range(args.num_users), m_mainFL, replace=False)
            list_of_random_workers = random.sample(workers_mainFL, m_mainFL)
            for i in range(len(list_of_random_workers)):
                dict_workers_index[iter_mainFL].append(list_of_random_workers[i])

            x_mainFL = net_glob_mainFL
            x_mainFL.eval()
            acc_test_global_mainFL, loss_test_global_mainFL = test_img(x_mainFL, valid_ds, args)
            Loss_local_each_global_total_mainFL.append(loss_test_global_mainFL)
            Accuracy_local_each_global_total_mainFL.append(acc_test_global_mainFL)
            SD_acc_full_main.append(acc_test_global_mainFL)
            SD_loss_full_main.append(loss_test_global_mainFL)

            workers_count_mainFL = 0
            temp_accuracy = np.zeros(1)
            temp_loss_test = np.zeros(1)
            temp_loss_train = np.zeros(1)
            for idx in list_of_random_workers:
                # print("train started")
                local_mainFL = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_mainFL[idx])
                w_mainFL, loss_mainFL = local_mainFL.train(net=copy.deepcopy(net_glob_mainFL).to(args.device))
                # print(w)
                # print("train completed")
                w_locals_mainFL.append(copy.deepcopy(w_mainFL))
                loss_locals_mainFL.append(copy.deepcopy(loss_mainFL))
                # temp = FedAvg(w)
                temp_netglob_mainFL.load_state_dict(w_mainFL)
                temp_netglob_mainFL.eval()
                print(pnorm_2(temp_netglob_mainFL, 2))
                acc_test_local_mainFL, loss_test_local_mainFL = test_img(temp_netglob_mainFL, valid_ds, args)
                temp_accuracy[0] = acc_test_local_mainFL
                temp_loss_test[0] = loss_test_local_mainFL
                temp_loss_train[0] = loss_mainFL
                loss_workers_total_mainFL[idx, iter_mainFL] = acc_test_local_mainFL
                workers_participation_main_fd[idx][iter_mainFL] = 1
                workers_count_mainFL += 1
                data_main['round_{}'.format(iter_mainFL)].append({
                    'C': args.frac,
                    'User ID': idx,
                    # 'Local Update': copy.deepcopy(w_mainFL),
                    'Loss Train': temp_loss_train[0],
                    'Loss Test': temp_loss_test[0],
                    'Accuracy': temp_accuracy[0]
                })

            # update global weights
            w_glob_mainFL = FedAvg(w_locals_mainFL)

            # copy weight to net_glob
            net_glob_mainFL.load_state_dict(w_glob_mainFL)

            # print("round completed")
            loss_avg_mainFL = sum(loss_locals_mainFL) / len(loss_locals_mainFL)
            # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_mainFL))
            loss_train_mainFL.append(loss_avg_mainFL)
            # print("round completed")

            acc_test_round_mainfed, loss_test_round_mainfed = test_img(net_glob_mainFL, dataset_test, args)
            validation_test_mainFed.append(acc_test_round_mainfed)
            workers_percent_main.append(workers_count_mainFL / args.num_users)

            # plot workers percent of participating
            print(iter_mainFL, " round main fl finished")

            acc_test_final_mainFL, loss_test_final_mainFL = test_img(net_glob_mainFL, dataset_test, args)
            while_counter_mainFL = loss_test_final_mainFL

            data_Global_main["Round"].append(iter_mainFL)
            data_Global_main["C"].append(args.frac)
            data_Global_main["Average Loss Train"].append(float(loss_avg_mainFL))
            data_Global_main["Average Loss Test"].append(float(loss_test_global_mainFL))
            data_Global_main["Accuracy Test"].append(float(acc_test_global_mainFL))
            data_Global_main["Workers Number"].append(float(workers_count_mainFL))
            data_Global_main["Large Test Loss"].append(float(loss_test_final_mainFL))
            data_Global_main["Large Test Accuracy"].append(float(acc_test_final_mainFL))

            iter_mainFL = iter_mainFL + 1

        workers_percent_final_mainFL = np.zeros(args.num_users)
        workers_name_mainFL = np.empty(args.num_users)
        for i in range(len(workers_participation_main_fd[:, 1])):
            workers_percent_final_mainFL[i] = sum(workers_participation_main_fd[i, :]) / args.epochs
            workers_name_mainFL[i] = i

        net_glob_mainFL.eval()
        # print("train test started")
        acc_train_final_main, loss_train_final_main = test_img(net_glob_mainFL, dataset_train, args)
        # print("train test finished")
        acc_test_final_main, loss_test_final_main = test_img(net_glob_mainFL, dataset_test, args)

        Final_LargeDataSetTest_MainFL["C"].append(args.frac)
        Final_LargeDataSetTest_MainFL["Test Loss"].append(float(loss_test_final_main))
        Final_LargeDataSetTest_MainFL["Test Accuracy"].append(float(acc_test_final_main))
        Final_LargeDataSetTest_MainFL["Train Loss"].append(float(loss_train_final_main))
        Final_LargeDataSetTest_MainFL["Train Accuracy"].append(float(acc_train_final_main))






        # copy weights
        w_glob = net_glob.state_dict()

        temp_after = copy.deepcopy(net_glob)
        temp_before = copy.deepcopy(net_glob)

        # training
        loss_train = []
        # cv_loss, cv_acc = [], []
        # val_loss_pre, counter = 0, 0
        # net_best = None
        # best_loss = None
        # val_acc_list, net_list = [], []

        Loss_local_each_global_total = []


        # valid_ds = create_shared_dataset(dataset_test, 500)
        loss_workers_total = np.zeros(shape=(args.num_users, args.epochs))
        label_workers = {i: np.array([], dtype='int64') for i in range(args.num_users)}

        workers_percent_dist = []
        validation_test_newFed = []
        workers_participation = np.zeros((args.num_users, args.epochs))
        workers = []
        for i in range(args.num_users):
            workers.append(i)

        counter_threshold_decrease = np.zeros(args.epochs)
        Global_Accuracy_Tracker = np.zeros(args.epochs)
        Global_Loss_Tracker = np.zeros(args.epochs)
        threshold = 0.5
        alpha = 0.5     ##decrease parameter
        beta = 0.1 ##delta accuracy controller
        gamma = 0.5  ##threshold decrease parameter


        Goal_Loss = float(loss_test_final_main)

        #for iter in range(args.epochs):

        net_glob.eval()
        acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
        while_counter = float(loss_test_final)
        iter = 0

        total_rounds_dcfl = 0

        while (while_counter + 0.01) > Goal_Loss and iter <= args.epochs:

            data_DCFL['round_{}'.format(iter)] = []
            Loss_local_each_global = []
            loss_workers = np.zeros((args.num_users, args.epochs))
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            counter_threshold = 0
            print(iter, " in dist FL started")
            #if iter % 5 == 0:

            x = copy.deepcopy(net_glob)
            x.eval()
            acc_test_global, loss_test_global = test_img(x, valid_ds, args)
            Loss_local_each_global_total.append(acc_test_global)
            Global_Accuracy_Tracker[iter] = acc_test_global
            Global_Loss_Tracker[iter] = loss_test_global
            if iter > 0 & (Global_Loss_Tracker[iter-1] - Global_Loss_Tracker[iter] <= beta):
                threshold = threshold - gamma
                if threshold == 0.0:
                    threshold = 1.0
                print("threshold decreased to", threshold)
            workers_count = 0

            SD_acc_full_distributed.append(acc_test_global)
            SD_loss_full_ditributed.append(loss_test_global)


            temp_w_locals = []
            temp_workers_loss = np.empty(args.num_users)
            temp_workers_accuracy = np.empty(args.num_users)
            temp_workers_loss_test = np.empty(args.num_users)
            temp_workers_loss_differenc = np.empty(args.num_users)
            temp_workers_accuracy_differenc = np.empty(args.num_users)
            flag = np.zeros(args.num_users)

            list_of_random_workers_newfl = []
            if iter < (args.epochs/2):
                for key, value in dict_workers_index.items():
                    # print(value)
                    if key == iter:
                        list_of_random_workers_newfl = dict_workers_index[key]
            else:
                list_of_random_workers_newfl = random.sample(workers, m)


            for idx in list_of_random_workers_newfl:
                #print("train started")

                # before starting train
                temp_before = copy.deepcopy(net_glob)
                # temp_before.load_state_dict(w)
                temp_before.eval()
                acc_test_local_before, loss_test_local_before = test_img(temp_before, valid_ds, args)

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                #print(w)
                #print("train completed")

                #print("type of idx is ", type(temp_w_locals))
                temp_w_locals.append(copy.deepcopy(w))
                temp_workers_loss[idx] = copy.deepcopy(loss)

                temp_after = copy.deepcopy(net_glob)

                temp_after.load_state_dict(w)
                temp_after.eval()
                acc_test_local_after, loss_test_local_after = test_img(temp_after, valid_ds, args)
                loss_workers_total[idx, iter] = loss_test_local_after
                temp_workers_accuracy[idx] = acc_test_local_after
                temp_workers_loss_test[idx] = loss_test_local_after
                temp_workers_loss_differenc[idx] = loss_test_local_before - loss_test_local_after
                temp_workers_accuracy_differenc[idx] = acc_test_local_after - acc_test_local_before

            print("train finished")
            while len(w_locals) < 1:
                #print("recieving started")
                index = 0
                for idx in list_of_random_workers_newfl:
                    #print("acc is ", temp_workers_accuracy[idx])
                    # print(temp_workers_loss_differenc)
                    if workers_count >= m:
                        break
                    elif temp_workers_loss_differenc[idx] >= (threshold) \
                            and temp_workers_loss_differenc[idx] > 0 \
                            and flag[idx]==0:
                        print("Update Received")
                        w_locals.append(copy.deepcopy(temp_w_locals[index]))
                        #print(temp_w_locals[index])
                        loss_locals.append(temp_workers_loss[idx])
                        flag[idx] = 1
                        workers_count += 1
                        workers_participation[idx][iter] = 1

                        data_DCFL['round_{}'.format(iter)].append({
                            'C': args.frac,
                            'User ID': idx,
                            'Loss Train': loss_workers_total[idx, iter],
                            'Loss Test': temp_workers_loss[idx],
                            'Accuracy': temp_workers_accuracy[idx]
                        })
                    index += 1
                if len(w_locals) < 1:
                    threshold = threshold / 2
                    if threshold == -np.inf:
                        threshold = 1
                print("threshold increased to ", threshold)




            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            #print("round completed")
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            workers_percent_dist.append(workers_count/args.num_users)


            counter_threshold_decrease[iter] = counter_threshold
            print(iter, " round dist fl finished")


            acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
            while_counter = loss_test_final


            data_Global_DCFL["Round"].append(iter)
            data_Global_DCFL["C"].append(args.frac)
            data_Global_DCFL["Average Loss Train"].append(loss_avg)
            data_Global_DCFL["Accuracy Test"].append(Global_Accuracy_Tracker[iter])
            data_Global_DCFL["Average Loss Test"].append(Global_Loss_Tracker[iter])
            data_Global_DCFL["Workers Number"].append(workers_count)
            data_Global_DCFL["Large Test Loss"].append(float(loss_test_final))
            data_Global_DCFL["Large Test Accuracy"].append(float(acc_test_final))

            total_rounds_dcfl = iter

            iter = iter + 1


        #plot workers percent of participating
        workers_percent_final = np.zeros(args.num_users)
        workers_name = np.empty(args.num_users)
        #print(workers_participation)
        for i in range(len(workers_participation[:, 1])):
            workers_percent_final[i] = sum(workers_participation[i, :])/args.epochs
            workers_name[i] = i



        workers_fraction_each_round_newFL = sum(workers_percent_final)/len(workers_percent_final)


        # testing
        #print("testing started")
        net_glob.eval()
        #print("train test started")
        acc_train_final, loss_train_final = test_img(net_glob, dataset_train, args)
        #print("train test finished")
        acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)

        acc_full_distributed.append(acc_test_final)
        loss_full_ditributed.append(loss_test_final)

        Final_LargeDataSetTest_DCFL["C"].append(args.frac)
        Final_LargeDataSetTest_DCFL["Test Loss"].append(float(loss_test_final))
        Final_LargeDataSetTest_DCFL["Test Accuracy"].append(float(acc_test_final))
        Final_LargeDataSetTest_DCFL["Train Loss"].append(float(loss_train_final))
        Final_LargeDataSetTest_DCFL["Train Accuracy"].append(float(acc_train_final))
        Final_LargeDataSetTest_DCFL["Total Rounds"].append(int(total_rounds_dcfl))

        variable_start = variable_start + while_counter

        print("C is ", c_counter/10)

    with open('CIFAR_100users_data_main_1229-2020.json', 'w') as outfile:
        json.dump(data_main, outfile)

    with open('CIFAR_100users_data_DCFL_1229-2020.json', 'w') as outfile:
        json.dump(data_DCFL, outfile)

    with open('CIFAR_100users_data_DCFL_Global_1229-2020.json', 'w') as outfile:
        json.dump(data_Global_DCFL, outfile)

    with open('CIFAR_100users_data_main_Global_1229-2020.json', 'w') as outfile:
        json.dump(data_Global_main, outfile)

    with open('Final-CIFAR_100users_data_main_Global_1229-2020.json', 'w') as outfile:
        json.dump(Final_LargeDataSetTest_MainFL, outfile)

    with open('Final-CIFAR_100users_data_DCFL_Global_1229-2020.json', 'w') as outfile:
        json.dump(Final_LargeDataSetTest_DCFL, outfile)


    return 1


if __name__ == '__main__':
    #final_loss_test, final_loss_train = main()
    acc_test_final = main()
    ##print("main #print: ", final_loss_train, final_loss_test)


