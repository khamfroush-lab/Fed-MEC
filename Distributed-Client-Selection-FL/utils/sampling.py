#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.Update import DatasetSplit
from random import seed
from random import randint


def create_shared_dataset(valid_ds, size):
    data_loader = DataLoader(valid_ds, batch_size=1)
    size_final_dataset = size
    each_label_size = size_final_dataset/10
    final_test_dataset_idx = np.zeros(size_final_dataset, dtype = int)
    # final_test_dataset_idx = {i: np.array([], dtype='int64') for i in range(10)}
    print(len(valid_ds))
    counter = np.zeros(10)
    counter_arr = 0
    for idx, (data, target) in enumerate(data_loader):
        # print(idx, ' ', ' ', target, ' ')
        if target == 0:
            if counter[0] < each_label_size:
                counter[0] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 1:
            if counter[1] < each_label_size:
                counter[1] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 2:
            if counter[2] < each_label_size:
                counter[2] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 3:
            if counter[3] < each_label_size:
                counter[3] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 4:
            if counter[4] < each_label_size:
                counter[4] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 5:
            if counter[5] < each_label_size:
                counter[5] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 6:
            if counter[6] < each_label_size:
                counter[6] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 7:
            if counter[7] < each_label_size:
                counter[7] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 8:
            if counter[8] < each_label_size:
                counter[8] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1
        elif target == 9:
            if counter[9] < each_label_size:
                counter[9] += 1
                final_test_dataset_idx[counter_arr] = int(idx)
                counter_arr += 1

    ds = DatasetSplit(valid_ds, final_test_dataset_idx)
    return ds

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    num_shards = 200
    num_imgs = int(60000/num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]


    dict_labels = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_labels_counter = np.zeros((10, num_users))

    chunks_number = int(num_shards/num_users)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chunks_number, replace=False))

        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:

            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            dict_labels[i] = np.concatenate((dict_labels[i], idxs_labels[1, idxs[rand * num_imgs:(rand + 1) * num_imgs]]), axis=0)


    for j in range(num_users):
        for i in range(len(dict_users[0])):
            if str(dict_labels[j][i]) == '0':
                dict_labels_counter[0][j] += 1
            elif str(dict_labels[j][i]) == '1':
                dict_labels_counter[1, j] += 1
            elif str(dict_labels[j][i]) == '2':
                dict_labels_counter[2, j] += 1
            elif str(dict_labels[j][i]) == '3':
                dict_labels_counter[3, j] += 1
            elif str(dict_labels[j][i]) == '4':
                dict_labels_counter[4, j] += 1
            elif str(dict_labels[j][i]) == '5':
                dict_labels_counter[5, j] += 1
            elif str(dict_labels[j][i]) == '6':
                dict_labels_counter[6, j] += 1
            elif str(dict_labels[j][i]) == '7':
                dict_labels_counter[7, j] += 1
            elif str(dict_labels[j][i]) == '8':
                dict_labels_counter[8, j] += 1
            elif str(dict_labels[j][i]) == '9':
                dict_labels_counter[9, j] += 1

    return dict_users, dict_labels_counter


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dict_labels = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_labels_counter = np.zeros((10, num_users))

    chunks_number = int(num_shards/num_users)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chunks_number, replace=False))

        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            # print(rand)
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_labels[i] = np.concatenate(
                (dict_labels[i], idxs_labels[1, idxs[rand * num_imgs:(rand + 1) * num_imgs]]), axis=0)


    for j in range(num_users):
        for i in range(len(dict_users[0])):
            if str(dict_labels[j][i]) == '0':
                dict_labels_counter[0][j] += 1
            elif str(dict_labels[j][i]) == '1':
                dict_labels_counter[1, j] += 1
            elif str(dict_labels[j][i]) == '2':
                dict_labels_counter[2, j] += 1
            elif str(dict_labels[j][i]) == '3':
                dict_labels_counter[3, j] += 1
            elif str(dict_labels[j][i]) == '4':
                dict_labels_counter[4, j] += 1
            elif str(dict_labels[j][i]) == '5':
                dict_labels_counter[5, j] += 1
            elif str(dict_labels[j][i]) == '6':
                dict_labels_counter[6, j] += 1
            elif str(dict_labels[j][i]) == '7':
                dict_labels_counter[7, j] += 1
            elif str(dict_labels[j][i]) == '8':
                dict_labels_counter[8, j] += 1
            elif str(dict_labels[j][i]) == '9':
                dict_labels_counter[9, j] += 1

    return dict_users, dict_labels_counter



def sampleFromClass(ds, k):
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in ds:
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_data.append(data)
            train_label.append(torch.unsqueeze(label, 0))
        else:
            test_data.append(data)
            test_label.append(torch.unsqueeze(label, 0))
    train_data = torch.cat(train_data)
    for ll in train_label:
        print(ll)
    train_label = torch.cat(train_label)
    test_data = torch.cat(test_data)
    test_label = torch.cat(test_label)

    return (TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label))


def mnist_noniid_unequal(dataset, num_users):

    dict_labels = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_labels_counter = np.zeros((10, num_users))
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
                dict_labels[i] = np.concatenate(
                    (dict_labels[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
                dict_labels[i] = np.concatenate(
                    (dict_labels[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
                dict_labels[i] = np.concatenate(
                    (dict_labels[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
                dict_labels[k] = np.concatenate(
                    (dict_labels[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users, dict_labels


# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
