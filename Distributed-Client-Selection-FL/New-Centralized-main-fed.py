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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_unequal
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img



def main():
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        print("type of test dataset" ,type(dataset_test))
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
            dict_users_2, dict_labels_counter_2 = dict_users, dict_labels_counter
            #dict_users, dict_labels_counter = mnist_noniid_unequal(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_glob_2 = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        net_glob_2 = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    #print(net_glob)

    #net_glob.train()

    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("val test finished")
    print("{:.2f}".format(acc_test))
    temp = net_glob


    #net_glob_2 = net_glob
    temp_2 = net_glob_2

    # copy weights
    w_glob = net_glob.state_dict()



    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    Loss_local_each_global_total = []

    test_ds, valid_ds = torch.utils.data.random_split(dataset_test, (9500, 500))
    loss_workers_total = np.zeros(shape=(args.num_users, args.epochs))
    label_workers = {i: np.array([], dtype='int64') for i in range(args.num_users)}

    workers_percent = []
    workers_count = 0
    acc_test_global, loss_test_global = test_img(x, valid_ds, args)
    selected_users_index = []

    for idx in range(args.num_users):
        # print("train started")
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        # print(w)
        # print("train completed")

        # temp = FedAvg(w)
        temp.load_state_dict(w)
        temp.eval()
        acc_test_local, loss_test_local = test_img(temp, valid_ds, args)
        loss_workers_total[idx, iter] = acc_test_local

        if workers_count >= (args.num_users / 2):
            break
        elif acc_test_local >= (0.7 * acc_test_global):
            selected_users_index.append(idx)



    for iter in range(args.epochs):
        print("round started")
        Loss_local_each_global = []
        loss_workers = np.zeros((args.num_users, args.epochs))
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        #if iter % 5 == 0:
        # Minoo
        x = net_glob
        x.eval()

        Loss_local_each_global_total.append(acc_test_global)


        for idx in selected_users_index:
            #print("train started")
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #print(w)
            #print("train completed")


            #temp = FedAvg(w)
            temp.load_state_dict(w)
            temp.eval()
            acc_test_local, loss_test_local = test_img(temp, valid_ds, args)
            loss_workers_total[idx, iter] = acc_test_local

            if workers_count >= (args.num_users/2):
                break
            elif acc_test_local >= (0.7 * acc_test_global):
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                print("Update Received")
                workers_count += 1


        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        print("round completed")
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        workers_percent.append(workers_count)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(workers_percent)), workers_percent)
    plt.ylabel('train_loss')
    plt.savefig('./save/Newfed_WorkersPercent_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                   args.iid))
    # print(loss_workers_total)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/Newfed_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    #


    plt.figure()
    for i in range(args.num_users):
        plot = plt.plot(range(len(loss_workers_total[i, :])), loss_workers_total[i, :], label = "Worker {}".format(i))
    plot5 = plt.plot(range(len(Loss_local_each_global_total)), Loss_local_each_global_total, color = '000000', label = "Global")
    plt.legend(loc='best')
    plt.ylabel('Small Test Set Accuracy of workers')
    plt.xlabel('Number of Rounds')
    plt.savefig('./save/NewFed_2workers_Acc_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))


    # plt.figure()
    # bins = np.linspace(0, 9, 3)
    # a = dict_labels_counter[:, 0].ravel()
    # print(type(a))
    # b = dict_labels_counter[:, 1].ravel()
    # x_labels = ['0', '1', '2', '3','4','5','6','7','8','9']
    # # Set plot parameters
    # fig, ax = plt.subplots()
    # width = 0.1  # width of bar
    # x = np.arange(10)
    # ax.bar(x, dict_labels_counter[:, 0], width, color='#000080', label='Worker 1')
    # ax.bar(x + width, dict_labels_counter[:, 1], width, color='#73C2FB', label='Worker 2')
    # ax.bar(x + 2*width, dict_labels_counter[:, 2], width, color='#ff0000', label='Worker 3')
    # ax.bar(x + 3*width, dict_labels_counter[:, 3], width, color='#32CD32', label='Worker 4')
    # ax.set_ylabel('Number of Labels')
    # ax.set_xticks(x + width + width / 2)
    # ax.set_xticklabels(x_labels)
    # ax.set_xlabel('Labels')
    # ax.legend()
    # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    # fig.tight_layout()
    # plt.savefig(
    #     './save/Newfed_2workersLabels_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
    #                                                                args.iid))

    # testing
    print("testing started")
    net_glob.eval()
    print("train test started")
    acc_train_final, loss_train_final = test_img(net_glob, dataset_train, args)
    print("train test finished")
    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
    print("val test finished")
    #print("Training accuracy: {:.2f}".format(acc_train))
    #print("Testing accuracy: {:.2f}".format(acc_test))
    print("{:.2f}".format(acc_test_final))
    #print("{:.2f".format(Loss_local_each_worker))

    # training
    w_glob_2 = net_glob_2.state_dict()

    loss_train_2 = []
    cv_loss_2, cv_acc_2 = [], []
    val_loss_pre_2, counter_2 = 0, 0
    net_best_2 = None
    best_loss_2 = None
    val_acc_list_2, net_list_2 = [], []

    Loss_local_each_global_total_2 = []

    loss_workers_total_2 = np.zeros(shape=(args.num_users, args.epochs))
    label_workers_2 = {i: np.array([], dtype='int64') for i in range(args.num_users)}

    for iter in range(args.epochs):
        print("round started")
        Loss_local_each_global_2 = []
        loss_workers_2 = np.zeros((args.num_users, args.epochs))
        w_locals_2, loss_locals_2 = [], []
        m_2 = max(int(args.frac * args.num_users), 1)
        idxs_users_2 = np.random.choice(range(args.num_users), m_2, replace=False)

        # Minoo
        x_2 = net_glob_2
        x_2.eval()
        acc_test_global_2, loss_test_global_2 = test_img(x_2, valid_ds, args)
        Loss_local_each_global_total_2.append(acc_test_global_2)

        for idx in idxs_users_2:
            #print("train started")
            local_2 = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_2[idx])
            w_2, loss_2 = local_2.train(net=copy.deepcopy(net_glob_2).to(args.device))
            #print(w)
            #print("train completed")
            w_locals_2.append(copy.deepcopy(w_2))
            loss_locals_2.append(copy.deepcopy(loss_2))
            #temp = FedAvg(w)
            temp_2.load_state_dict(w_2)
            temp_2.eval()
            acc_test_local_2, loss_test_local_2 = test_img(temp_2, valid_ds, args)
            loss_workers_total_2[idx, iter] = acc_test_local_2


        # update global weights
        w_glob_2 = FedAvg(w_locals_2)

        # copy weight to net_glob
        net_glob_2.load_state_dict(w_glob_2)

        print("round completed")
        loss_avg_2 = sum(loss_locals_2) / len(loss_locals_2)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg_2))
        loss_train_2.append(loss_avg_2)
        print("round completed")

        # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_2)), loss_train_2, color = '#000000', label = "Main FL")
    plt.plot(range(len(loss_train)), loss_train, color = '#ff0000', label = "Centralized Algorithm")
    plt.ylabel('train_loss')
    plt.savefig('./save/main_fed_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    # print(loss_workers_total)


    plt.figure()
    for i in range(args.num_users):
        plot = plt.plot(range(len(loss_workers_total_2[i, :])), loss_workers_total_2[i, :], label = "Worker {}".format(i))
    plot5 = plt.plot(range(len(Loss_local_each_global_total_2)), Loss_local_each_global_total_2, color = '000000', label = "Global")
    plt.legend(loc='best')
    plt.ylabel('Small Test Set Accuracy of workers')
    plt.xlabel('Number of Rounds')
    plt.savefig('./save/mainfed_Acc_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))


    # plt.figure()
    # bins = np.linspace(0, 9, 3)
    # a = dict_labels_counter_2[:, 0].ravel()
    # print(type(a))
    # b = dict_labels_counter_2[:, 1].ravel()
    # x_labels = ['0', '1', '2', '3','4','5','6','7','8','9']
    # # Set plot parameters
    # fig, ax = plt.subplots()
    # width = 0.1  # width of bar
    # x = np.arange(10)
    # ax.bar(x, dict_labels_counter_2[:, 0], width, color='#000080', label='Worker 1')
    # ax.bar(x + width, dict_labels_counter_2[:, 1], width, color='#73C2FB', label='Worker 2')
    # ax.bar(x + 2*width, dict_labels_counter_2[:, 2], width, color='#ff0000', label='Worker 3')
    # ax.bar(x + 3*width, dict_labels_counter_2[:, 3], width, color='#32CD32', label='Worker 4')
    # ax.set_ylabel('Number of Labels')
    # ax.set_xticks(x + width + width / 2)
    # ax.set_xticklabels(x_labels)
    # ax.set_xlabel('Labels')
    # ax.legend()
    # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    # fig.tight_layout()
    # plt.savefig(
    #     './save/main_fed_2workersLabels_0916_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
    #                                                                args.iid))

    # testing
    print("testing started")
    net_glob.eval()
    print("train test started")
    acc_train_final, loss_train_final = test_img(net_glob, dataset_train, args)
    print("train test finished")
    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
    print("val test finished")
    #print("Training accuracy: {:.2f}".format(acc_train))
    #print("Testing accuracy: {:.2f}".format(acc_test))
    print("{:.2f}".format(acc_test_final))
    #print("{:.2f".format(Loss_local_each_worker))

    return loss_test_final, loss_train_final


if __name__ == '__main__':
    final_loss_test, final_loss_train = main()
    #print("main print: ", final_loss_train, final_loss_test)


