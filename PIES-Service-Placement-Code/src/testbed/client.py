# -- coding: future_fstrings --
import numpy as np
import numpy.random as rand
import pickle
import random
import socket
import src.testbed.protocol as protocol
import time

from collections import defaultdict
from PIL import Image
from time import ctime
from typing import List

from src.services.torch_eval import imagenet_data, get_preprocessor
from src.testbed.client_profiles import *
from src.testbed.const import *
from src.testbed.protocol import Request
from src.testbed.util import *

LOCAL_MIN_DELAY = 0.0
LOCAL_MAX_DELAY = 1.0
MIN_DELAY = 0.0
MAX_DELAY = protocol.DELAY_MAX

# Accuracy functions.
def high_acc_fn() -> float:
    return 1 - np.clip(rand.exponential(scale=0.0625), 0, 1)
def standard_acc_fn() -> float:
    return 1 - np.clip(rand.exponential(scale=0.0625), 0, 1)

# Non-local delay functions.
def low_delay_fn() -> float:
    return np.clip(rand.normal(loc=0.25, scale=0.125), 0, MAX_DELAY)
def standard_delay_fn() -> float:
    return np.clip(rand.normal(loc=0.25, scale=0.125), 0, MAX_DELAY)

# Local delay functions.
def local_low_delay_fn() -> float:
    return np.clip(rand.normal(loc=0.125, scale=0.09375), 0, LOCAL_MAX_DELAY)
def local_standard_delay_fn() -> float:
    return np.clip(rand.normal(loc=0.5, scale=0.1875), 0, LOCAL_MAX_DELAY)

class Client:

    def __init__(
        self,
        image_dir: str=None,
        requests: List[Request]=None,
        hostname: str=DEFAULT_HOST,
        portno: int=DEFAULT_PORT,
        high_acc: bool=False,
        low_delay: bool=False,
        random_profile: bool=False,
        start_index: int=None,
        end_index: int=None,
    ) -> None:
        self.conn = None
        self.image_dir = image_dir
        self.requests = requests
        self.hostname = hostname
        self.portno = portno
        self.dataset = imagenet_data(imagenet_dir=image_dir, do_preprocessing=False)
        self.preprocessor = get_preprocessor()
        self.local = (self.hostname == "localhost")
        self.correct_answers = None
        self.num_mc_runs = 1

        self.random_profile = random_profile
        self.high_acc = high_acc
        self.low_delay = low_delay
        self.update_profile()

        self.start_index = start_index
        self.end_index = end_index
        self.data = None

    def update_profile(self) -> None:
        if self.random_profile:
            choices = [False, True]
            self.high_acc = random.choice(choices)
            self.low_delay = random.choice(choices)
        self.set_acc_fns()
        self.set_delay_fns()

    def set_acc_fns(self) -> None:
        if self.high_acc:
            self.accuracy_fn = high_acc_fn
        else:
            self.accuracy_fn = standard_acc_fn

    def set_delay_fns(self) -> None:
        if self.low_delay and self.local:
            self.delay_fn = local_low_delay_fn
        elif self.low_delay and not self.local:
            self.delay_fn = low_delay_fn
        elif not self.low_delay and self.local:
            self.delay_fn = local_standard_delay_fn
        elif not self.low_delay and not self.local:
            self.delay_fn = standard_delay_fn

    def run(self, n_requests: int=None) -> None:
        """The sequence of methods called to handle requests submitted to the server. Note
           that
        """
        self.connect()

        # Initialize a dictionary object that will record information pertaining to
        # the responses provided by the server to individual requests. This will be
        # used for evaluation.
        self.data = defaultdict(list)

        for mc_run in range(self.num_mc_runs):
            self.update_profile()
            if n_requests is not None:
                self.generate_requests(n_requests)
            self.inform()
            self.submit_requests(mc_run)

        # Time to send the data recorded by this client with regard to its QoS back to
        # the server for real-world evaluation.
        recv_msg(self.conn)
        send_msg(self.conn, pickle.dumps(dict(self.data)))
        Client.log("Results data sent to the server!")


    def connect(self) -> None:
        """Establish a connection with the (edge) server."""
        Client.log("Preparing to connect...")
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.hostname, self.portno))
        Client.log(f"Established connection with ({self.hostname}, {self.portno}).")
        send_msg(self.conn, pickle.dumps(socket.gethostname()))
        self.num_mc_runs = pickle.loads(recv_msg(self.conn))
        Client.log(f"Number of MC-Runs: {self.num_mc_runs}.")


    def random_request(self, image, label) -> Request:
        """Returns a random request using the client's delay and accuracy generator.

        Args:
            image (PIL.Image): Image data submitted with the request for classification.
            label (int): The integer corresponding with the correct label for the image.

        Returns:
            Request: Instance of a Request with all the needed data.
        """
        return Request(data=image,
                       req_service=protocol.IMAGE_CLASSIFICATION,
                       req_accuracy=self.accuracy_fn(),
                       req_delay=self.delay_fn(),
                       true_ans=label)


    def generate_requests(self, n_requests: int) -> None:
        """Creates a random set of requests with ImageNet data and accuracy/delay
           thresholds for QoS.

        Args:
            n_requests (int): Number of requests to generate.
        """
        low_index = 0 if self.start_index is None else self.start_index
        high_index = len(self.dataset) if self.end_index is None else self.end_index

        assert 0 <= high_index <= len(self.dataset)
        assert 0 <= low_index < high_index
        assert high_index - low_index >= n_requests
        assert isinstance(low_index, int) and isinstance(high_index, int)

        requests = []
        indices = list(range(low_index, high_index))
        random.shuffle(indices)
        for i in indices[:n_requests]:
            image, label = self.dataset[i]
            requests.append(self.random_request(image, label))
        self.requests = requests


    def inform(self) -> None:
        """Informing the (edge) server of our request thresholds (no data attached)."""
        send_msg(self.conn, protocol.READY)
        recv_msg(self.conn)
        Client.log("Sending requests to inform server for placement/scheduling.")
        for request in self.requests:
            temp_request = protocol.strip_data(request)
            send_msg(self.conn, pickle.dumps(temp_request))
            response = recv_msg(self.conn)
            req_id = pickle.loads(response)
            request.idx = req_id
        send_msg(self.conn, protocol.DONE)


    def submit_requests(self, mc_run: int) -> None:
        """Wait for a READY flag from the server after it finishes placing/scheduling.
           Then, sequentially submits requests in real-time and receives responses for
           each algorithm.
        """
        for _ in protocol.STRATEGIES:
            algorithm_label = pickle.loads(recv_msg(self.conn))
            for request in self.requests:
                start_time = time.time()
                req_copy = request.copy()
                req_copy.data = self.preprocessor(req_copy.data)
                send_msg(self.conn, pickle.dumps(req_copy))
                response = pickle.loads(recv_msg(self.conn))
                served_delay = time.time() - start_time

                pred, served_model = response["prediction"], response["served_model"]
                Client.log(f"Response from server for request {request.idx}: {pred} "
                           f"(right answer: {request.true_ans})")

                self.data["req_accuracy"].append(request.req_accuracy)
                self.data["req_delay"].append(request.req_delay)
                self.data["req_id"].append(request.idx)
                self.data["served_delay"].append(served_delay)
                self.data["served_model"].append(served_model)
                self.data["served_pred"].append(pred)
                self.data["true_answer"].append(request.true_ans)
                self.data["high_acc"].append(self.high_acc)
                self.data["low_delay"].append(self.low_delay)
                self.data["algorithm"].append(algorithm_label)
                self.data["mc_run"].append(mc_run)
            send_msg(self.conn, protocol.DONE)
        Client.log("Finished with service requests.")

    @staticmethod
    def log(string) -> None:
        prefix = f"$ [{ctime()}]"
        print(f"{prefix} {string}")