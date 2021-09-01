import numpy as np
import torch

from torchvision import transforms
from typing import Any, Callable

class Model:
    """This class wraps up the process of performing inference using one of the selected
       service models from `src.services`. This will only be used for serving real-world
       requests by connected devices in a client-server environment.
    """
    def __init__(self, model_fn: Callable, device: str="device"):
        self._model_fn = model_fn
        self.device = device

    def __call__(self, data) -> Any:
        self.model.eval()
        with torch.no_grad():
            data = data.unsqueeze(0)
            pred = torch.argmax(self.model(data))
        return pred

    
class ImageClassificationModel(Model):
    """Model wrapper for image classification models implemented in PyTorch. For more
       information about the kind of model this class is based on, refer to:
       https://pytorch.org/hub/pytorch_vision_googlenet/
    """
    def __init__(self, model_fn: Callable, device: str="device"):
        super().__init__(model_fn, device)
        self.model = self._model_fn()
        self.preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, data) -> Any:
        self.model.eval()
        with torch.no_grad():
            # data = self.preprocessor(data) ## NOTE: We're doing client-side preprocessing.
            data = data.unsqueeze(0)
            pred = torch.argmax(self.model(data))
        return pred.item()


class SpeechToTextModel(Model):
    """Model wrapper for speech-to-text (STT) models implemented in PyTorch. For 
       information about the model this was based on, refer to:
       https://pytorch.org/hub/snakers4_silero-models_stt/
    """
    def __init__(self, model_fn: Callable, device: str="device"):
        super().__init__(model_fn, device)
        self.model, self.decoder, utils = self._model_fn()
        (self.read_batch, self.split_into_batches,
         self.read_audio, self.prepare_model_input) = utils

    def __call__(self, data) -> Any:
        self.model.eval()
        batches = self.split_into_batches(data, batch_size=10)
        input = self.prepare_model_input(self.read_batch(batches[0]),
                                         device=self.device)
        output = self.model(input)
        return self.decoder(output.cpu())