#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np

from torch import nn
from typing import List


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


#
def FedAvg_2(w, n_k):
    # print("n_k is ", n_k)
    w_avg = copy.deepcopy(w[0])
    coeff = np.zeros(len(w))
    for k in w_avg.keys():
        # print("k is ", k)
        for i in range(1, len(w)):
            # print(f"n_k[{i}] is ", n_k[i])
            # print("sum of n_k is ", sum(n_k))
            coeff[i] = n_k[i]/sum(n_k)
            # w_temp = torch.mul(w[i][k], coeff[i])
            # print(type(w[i][k]))
            w_avg[k] += (torch.mul(w[i][k], coeff[i]))
            # w_avg[k] += (torch.mul(w[i][k], n_k[i]).type(torch.LongTensor))
        # w_avg[k] = torch.div(w_avg[k], sum(n_k))
    return w_avg


def fed_avg(models: List[nn.Module], data_sizes: List[int]):
    __model = models[0]
    new_weights = {layer: torch.zeros(size=params.shape)
                   for layer, params in __model.items()}

    coeffs = np.array([size for size in data_sizes])
    coeffs = coeffs / sum(coeffs)

    with torch.no_grad():
        for k, model in enumerate(models):
            for layer, params in model.items():
                new_weights[layer] += coeffs[k] * params

    return new_weights
