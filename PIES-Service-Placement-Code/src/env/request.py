from typing import List

class Request:

    def __init__(self, service: int, accuracy_thresh: float, delay_thresh: float) -> None:
        assert 0.0 <= accuracy_thresh <= 1.0
        assert 0.0 <= delay_thresh <= 1.0

        self.service = service
        self.accuracy_thresh = accuracy_thresh
        self.delay_thresh = delay_thresh

class User:

    def __init__(self, requests: List[Request]) -> None:
        self.requests = requests
        self.covering_edge = None