#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import defaultdict
import numpy as np
import torch

def merge(dict1, dict2):
	# print(len(dict1))
	# print(len(dict2))
	for k, v in dict1.items():
		for m, n in dict2.items():
			if k == m:
				dict1[k] = dict1[k] + dict2[k]

	return dict1


def pnorm(x, p):
    return (np.abs(x)**p).sum() ** (1./p)

def pnorm_2(model, p):
	total_norm = 0
	for param in model.parameters():
		print(param.grad.data)
		param_norm = param.grad.data.norm(p)
		total_norm += param_norm.item() ** p
		total_norm = total_norm ** (1. / p)


def compare_models(model_1, model_2):
	models_differ = 0
	for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
		if torch.equal(key_item_1[1], key_item_2[1]):
			pass
		else:
			models_differ += 1
			if (key_item_1[0] == key_item_2[0]):
				print('Mismtach found at', key_item_1[0])
			else:
				raise Exception
	if models_differ == 0:
		print('Models match perfectly! :)')