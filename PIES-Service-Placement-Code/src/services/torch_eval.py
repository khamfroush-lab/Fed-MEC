# -- coding: future_fstrings --
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision
import warnings

from .torch_models import *

from collections import defaultdict
from torch import nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from typing import Any, List, Tuple

warnings.filterwarnings("ignore")

device = torch.device("cpu")
IMAGENET_DIR = "~/Development/torch_datasets/imagenet-2012/"
COMMONVOICE_DIR = "~/Development/torch_datasets/common-voice/"
COMMONVOICE_VER = "cv-corpus-1-2019-02-25"
COMMONVOICE_VER_6 = "cv-corpus-6.1-2020-12-11"
MNIST_DIR = "~/Development/torch_datasets/mnist/"

class Services:

    IMAGE_CLASSIFICATION = 0
    SPEECH_TO_TEXT = 1
    SERVICES = {
        "IMAGE_CLASSIFICATION": 0,
        "SPEECH_TO_TEXT": 1,
    }
    MODELS = {
        (0, 0): {
            "name": "AlexNet",
            "func": alexnet,
        },
        (0, 1): {
            "name": "DenseNet",
            "func": densenet,
        },
        (0, 2): {
            "name": "GoogLeNet",
            "func": googlenet,
        },
        (0, 3): {
            "name": "MobileNet",
            "func": mobilenet,
        },
        (0, 4): {
            "name": "ResNet",
            "func": resnet,
        },
        (0, 5): {
            "name": "SqueezeNet",
            "func": squeezenet,
        },
    }

    @classmethod
    def get_models(cls, service_id: int) -> List[int]:
            return [m for (s, m), v in cls.MODELS.items() if s == service_id]

    @classmethod
    def get_models_from_str(cls, service_name: str) -> List[str]:
        assert service_name in cls.SERVICES
        return [v["name"] for (s, m), v in cls.MODELS.items() if s == service_name]

    @classmethod
    def get_model_name(cls, service_id: int, model_id: int) -> str:
        return cls.MODELS[service_id, model_id]["name"]

    @classmethod
    def get_service_model(cls, service_id: int, model_id: int) -> torch.nn.Module:
        return cls.MODELS[service_id, model_id]["func"]()

    @classmethod
    def get_service_model_from_str(cls, model_name: str) -> torch.nn.Module:
        model_fn = None
        for values in cls.MODELS.values():
            if values["name"] == model_name:
                model_fn = values["func"]
        if model_fn is None:
            raise ValueError(f"Provided `model_name` ('{model_name}') invalid.")

        return model_fn()

def get_preprocessor() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def imagenet_data(
    imagenet_dir: str=IMAGENET_DIR,
    do_preprocessing: bool=True
) -> torchvision.datasets.ImageNet:
    imagenet_dir = IMAGENET_DIR if imagenet_dir is None else imagenet_dir
    preprocess = get_preprocessor() if do_preprocessing else None
    return torchvision.datasets.ImageNet(imagenet_dir, split="val",
                                         transform=preprocess, download=False)


def imagenet_evaluate(
    model,
    first_n: int=None,
    pbar_desc: str=None
) -> Tuple[float, float, float]:
    from hwcounter import count, count_end
    data_loader = imagenet_data()
    correct, total = 0, 0
    cpu_cycles = []
    delays = []

    if first_n is None:
        pbar = tqdm(total=len(data_loader), desc=pbar_desc)
    else:
        assert first_n > 0
        pbar = tqdm(total=first_n, desc=pbar_desc)

    model.eval()
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            start = time.time()
            cpu_start = count()
            pred = torch.argmax(model(x.unsqueeze(0)))
            cpu_end = count_end() - cpu_start
            delay = time.time() - start
            total += 1
            correct += (pred == y).sum().item()
            cpu_cycles.append(cpu_end)
            delays.append(delay)
        pbar.update()

        if first_n is not None:
            if i >= first_n-1:
                break

    return correct/total, np.mean(cpu_cycles), np.mean(delay)