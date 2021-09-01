from typing import List, Number

from .request import User
from .service import ServiceModel

class EdgeCloud:

    def __init__(
        self, 
        idx: int, 
        comm: Number=None, 
        comp: Number=None, 
        stor: Number=None, 
        users: List[User]=None, 
        models: List[ServiceModel]=None
    ) -> None:
        self.idx = idx
        self.comm = comm
        self.comp = comp
        self.stor = stor
        self.users = users
        self.models = models