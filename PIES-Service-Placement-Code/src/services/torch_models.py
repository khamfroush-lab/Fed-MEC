import torch
import torchvision

"""
IMAGE CLASSIFICATION MODELS
"""
def alexnet():
    return torchvision.models.alexnet(pretrained=True, progress=False)
    
def densenet():
    return torchvision.models.densenet161(pretrained=True, progress=False)
    
def googlenet():
    return torchvision.models.googlenet(pretrained=True, progress=False)
    
def mobilenet():
    return torchvision.models.mobilenet_v2(pretrained=True, progress=False)
    
def resnet():
    return torchvision.models.resnet18(pretrained=True, progress=False)
    
def squeezenet():
    return torchvision.models.squeezenet1_0(pretrained=True, progress=False)