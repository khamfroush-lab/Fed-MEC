# -- coding: future_fstrings --
from typing import List, Number

class ServiceModel:

    def __init__(
        self,
        service_idx: int,
        model_idx: int,
        comm: Number, 
        comp: Number, 
        stor: Number
    ) -> None:
        super().__init__()
        self.service_idx = service_idx
        self.model_idx = model_idx
        self.comm = comm
        self.comp = comp
        self.stor = stor

    def __repr__(self) -> str:
        return f"SxM[{self.service_idx}, {self.model_idx}]"


class Service:

    def __init__(
        self, 
        idx: int,
        max_models: int,
        models: List[ServiceModel]=None
    ) -> None:
        if models is not None:
            assert len(models) <= max_models
        self.idx = idx
        self.max_models = max_models
        self.models = models

    def __getitem__(self, index: int) -> ServiceModel:
        return self.models[index]

    def __len__(self) -> int:
        return len(self.models)