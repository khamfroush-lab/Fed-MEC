# Service Models
For this, we are ditching TensorFlow in lieu of PyTorch. There are many different models implemented in PyTorch that are pretrained that we will use. The decision for using PyTorch is the excessive challenge in using TensorFlow for loading in data.

## Audio (Mozilla's `CommonVoice` data)
For audio tasks, we will use the `CommonVoice` dataset provided by Mozilla and consider the following pre-trained models:
* Silero Speech-to-Text

## Image Classification (`ImageNet 2012` data)
We will perform image classification on the `ImageNet 2012` dataset with the following pre-trained models:
* AlexNet
* DenseNet
* GoogLeNet
* MobileNet
* ResNet
* SqueezeNet