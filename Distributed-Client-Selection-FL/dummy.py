#!/usr/bin/env python3


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math

import random
from mpl_toolkits import mplot3d
import json
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
import argparse
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

from collections import OrderedDict
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


args = args_parser()
args.gpu = -1
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#dataset_train = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=trans_fmnist)
#dict_users, dict_labels_counter = mnist_noniid(dataset_train, args.num_users)
net_glob = CNNMnist(args=args).to(args.device)
#print(net_glob)

m = net_glob
m.train()

for p1, p2 in zip(m.parameters(), net_glob.parameters()):
	if p1.data.ne(p2.data).sum() > 0:
		print(False)
print(True)


#local_mainFL = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[5])
#w_mainFL, loss_mainFL=local_mainFL.train(net=copy.deepcopy(net_glob_mainFL).to(args.device))
#compare_models(net_glob, w_mainFL)

def pnorm(model, p):
#	for layer in model.keys():
	total_norm = 0
#	norm_weights = OrderedDict()
	norm_weights = []
	with torch.no_grad():
		for param in model.parameters():
			norm_weights.append(param.data)
		norm_weights = torch.stack(norm_weights)
		norm_weights = torch.norm(norm_weights)
#		for param, weights in model.state_dict().items():
#			norm_weights[param] = torch.norm(weights.data, p=p)
	return norm_weights
#		if param.grad is not None:
#			print(type(param))
#			param_norm = param.grad.data.norm(p)
#			total_norm += param_norm.item() ** p
#		total_norm = total_norm ** (1. / p)
#	return total_norm


def norm1(x, p):
	"First-pass implementation of p-norm."
	return (abs(x)**p).sum() ** (1./p)

#print(norm1(net_glob.state_dict(),1))
#net_glob.load_state_dict()
m = net_glob.train()
net_glob.eval()
#print(type(net_glob.state_dict()))

#print(net_glob.state_dict().keys())
#data = list(net_glob.state_dict().items())
#an_array = np.array(data)

#for layer in net_glob.ordered_layers:
#	norm_grad = layer.weight.grad.norm()
#	tone = f + ((norm_grad.numpy()) * 100.0)
best_state = copy.deepcopy(net_glob.state_dict())
#print(net_glob.grad)
print(pnorm(net_glob,2))
#print(torch.norm(net_glob, dim=None, p=2))
#print(net_glob.state_dict().grad.norm(1))

#print(torch.nn.utils.clip_grad_norm(net_glob, 1, 1))
