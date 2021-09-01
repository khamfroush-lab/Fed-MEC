# -- coding: future_fstrings --
import pickle

from sys import getsizeof
from typing import Any

from ..strategies.knapsack import knapsack
from ..strategies.optimal import optimal
from ..strategies.proposed_v2 import greedy_v2
from ..strategies.simple_greedy import simple_greedy
from ..strategies.random import random_greedy, random_random

READY = b"ready"
DONE = b"quit"
DROPPED = b"dropped"

IMAGE_CLASSIFICATION = 0
# DELAY_MAX = 5
DELAY_MAX = 1.0 # 0.8

STRATEGIES = {
    "OPT": lambda env: optimal(env, msg=False),
    "EGP": greedy_v2,
    "AGP": simple_greedy,
    "SCK": knapsack,
    "RND": random_random,
}
STRATEGIES["OPT"].__name__ = "OPT"

class Request(object):

    def __init__(
        self,
        data=None,
        req_service=None,
        req_accuracy=None,
        req_delay=None,
        true_ans=None,
        idx: int=None
    ) -> None:
        self.data = data
        self.req_service = req_service
        self.req_accuracy = req_accuracy
        self.req_delay = req_delay
        self.true_ans = true_ans
        self.idx = idx

    def copy(self):
        return Request(data=self.data,
                       req_service=self.req_service,
                       req_accuracy=self.req_accuracy,
                       req_delay=self.req_delay,
                       true_ans=self.true_ans,
                       idx=self.idx)

    def __repr__(self):
        shorten = lambda string: string if len(string) <= 10 else f"{string[:7]}..."
        return f"<data: '{shorten(self.data)}', serv: {self.req_service}, " \
               f"acc: {self.req_accuracy}, del: {self.req_delay}>"


def strip_data(req: Request):
    stripped_request = Request(
        req_service=req.req_service,
        req_accuracy=req.req_accuracy,
        req_delay=req.req_delay,
        idx=req.idx
    )
    return stripped_request