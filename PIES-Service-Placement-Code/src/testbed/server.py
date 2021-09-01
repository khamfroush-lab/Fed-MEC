# -- coding: future_fstrings --
import numpy as np
import os
import pandas as pd
import pickle
import socket
import struct
import subprocess
import time

from collections import defaultdict
from datetime import datetime
from threading import Thread
from time import ctime
from typing import Any, Dict

from . import protocol
from .const import *
from .model import ImageClassificationModel, SpeechToTextModel
from .protocol import IMAGE_CLASSIFICATION, Request
from .util import *
from ..env.environment import Environment
from ..env.qos import *
from ..services.torch_eval import Services

EDGE_ID = 0
OUT_DIR = os.path.join("out", "real-world")

SERVER_ENV_CONFIG = {
    # NOTE: comp = 3.2e9 -> 3.2e6
    "edges": {EDGE_ID: dict(comm=10*1500000, comp=3.2e6, stor=1,
                            requests=set(), services=set())},
    "requests": {},
    "services": SERVICES,
    "max_delay": protocol.DELAY_MAX, #10*1000,
}

class Server:

    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        buffsiz=DEFAULT_BUFF
    ) -> None:
        self.host = host
        self.port = port
        self.buffsiz = buffsiz
        self.addr = (self.host, self.port)
        self.socketserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = {}

        self.env = None
        self.placement_decisions = None
        self.scheduling_decisions = None
        self.torch_models = dict()
        self.session_data = defaultdict(list)

    # ================================================================================== #

    def run(self, num_clients: int, num_mc_runs: int=1):
        qos_pred_data_df = pd.DataFrame()
        results_df = pd.DataFrame()
        self.listen(num_clients, num_mc_runs)
        for run in range(num_mc_runs):
            df = self.inform()
            qos_pred_data_df = pd.concat([qos_pred_data_df, df])
            Server.log(f"Running Monte-Carlo run ({run+1}/{num_mc_runs})")
            for alg_label, algorithm in protocol.STRATEGIES.items():
                self.place(algorithm)
                self.serve(alg_label)

        df = self.collect_results_data()
        results_df = pd.concat([results_df, df])

        date = datetime.now()

        # Save the results data from the experiment that was provided by the clients.
        results_dir = os.path.join(OUT_DIR, "results", str(date.date()))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        path = os.path.join(results_dir, f"{date.time()}.csv")
        results_df = results_df.reset_index()
        results_df.to_csv(path)
        Server.log(f"Results saved to '{path}'.")

        # Save the prediction data that can be used for sensitivity analysis w.r.t. QoS.
        qos_pred_dir = os.path.join(OUT_DIR, "qos_pred", str(date.date()))
        if not os.path.exists(qos_pred_dir):
            os.makedirs(qos_pred_dir)
        path = os.path.join(qos_pred_dir, f"{date.time()}.csv")
        qos_pred_data_df = qos_pred_data_df.reset_index()
        qos_pred_data_df.to_csv(path)
        Server.log(f"QoS predictions saved to '{path}'.")

    # ================================================================================== #

    def listen(self, num_clients: int, num_mc_runs: int=1, **kwargs):
        self.clients, idx = {}, 0
        self.socketserver.bind(self.addr)
        self.socketserver.listen(5)
        Server.log(f"Listening for connections from {num_clients} clients...")
        while len(self.clients) < num_clients:
            conn, addr = self.socketserver.accept()
            name = pickle.loads(recv_msg(conn))
            send_msg(conn, pickle.dumps(num_mc_runs))
            Server.log(f"...Accepted connection from '{name}'.")
            self.clients[idx] = {
                "conn": conn,
                "addr": addr,
                "name": name,
            }
            idx += 1

    # ================================================================================== #

    def inform(self, n_packets=10, wait_sec=1) -> pd.DataFrame:
        """Collect requests from clients (a priori) to instantiate an Environment
           representation of the client-server ecosystem. This will be used to run the
           placement/scheduling algorithm.

        Args:
            n_packets (int, optional): Number of ping packets to measure communication
                signal strength. Defaults to 10.
            wait_sec (int, optional): Number of seconds to wait for ping packets.
                Defaults to 1.
        """
        def ping_thread(client_id: int) -> None:
            """Helper function to gauge the communication strength to a client using ping.

            Args:
                client_id (int): The int id associated with the client in `self.clients`.
            """
            # TODO: Come up with a way to measure the SIZE of the Ping packets so we can
            #       come up with a bits-per-second metric.
            addr = self.clients[client_id]["addr"][0]
            cmd = f"ping -c {n_packets} -W {wait_sec} {addr}".split()
            try:
                output = subprocess.check_output(cmd).decode().strip()
                lines = output.split("\n")
                n_received = int(lines[-2].split(',')[1].split()[0])
                times = lines[-1].split()[3].split("/")
                min_t, avg_t, max_t, dev_t = (float(t) for t in times)
                self.clients[client_id]["rtt"] = {
                    "min": min_t,
                    "avg": avg_t,
                    "max": max_t,
                    "stddev": dev_t,
                    "received_packed": n_received,
                    "sent_packets": n_packets,
                }
            except Exception as e:
                print(e)

        threads = [Thread(target=ping_thread, args=(client_id,), daemon=False)
                   for client_id in self.clients]

        for t in threads: t.start()
        for t in threads: t.join()
        req_2_client = {}

        def collect_requests() -> Dict[int, Request]:
            """Collects the requests from the users (represented as a dict) and then
               creates a representation of the client-server environment using the
               Environment class used for the PIES problem. This is to run the
               placement/scheduling algorithms w.r.t. user/client requests.

            Returns:
                Dict[int, Request]: Repository of requests retrieved from clients.
            """
            requests, idx = dict(), 0
            for client_id in self.clients:
                sock = self.clients[client_id]["conn"]
                recv_msg(sock)
                send_msg(sock, protocol.READY)
                response = recv_msg(sock)
                while response != protocol.DONE:
                    req: Request = pickle.loads(response)
                    requests[idx] = {
                        "covered_by":  EDGE_ID,
                        "req_accuracy": req.req_accuracy,
                        "req_delay":    req.req_service,
                        "req_service":  req.req_service
                    }
                    # Let the client know the request's ID, iterate the ID number, and
                    # receive the next response for the current client.
                    send_msg(sock, pickle.dumps(idx))
                    req_2_client[idx] = client_id
                    idx += 1
                    response = recv_msg(sock)
            return requests

        # Collect requests from all of the users (a priori) to inform the server as to
        # how it should place services. Information will be store in an `Environment`
        # object. This is to use the implemented strategies/algorithms for the
        # placement and scheduling problems.
        Server.log("Collecting requests to inform placement/scheduling.")
        requests = collect_requests()
        config = SERVER_ENV_CONFIG.copy()
        config["edges"][EDGE_ID]["requests"] = set(requests.keys())
        config["requests"] = requests

        # NOTE: This works fine for when each client submits 10 requests, but acts bizarre
        # when each client submits 100... We need to fix this somehow, but I'm unsure of
        # how to fix it...
        config["edges"][EDGE_ID]["comp"] *= 1/3 # Maybe try (1/3)?

        # With all of the requests collected and the environment setup, create a dataset
        # that shows the predictions for Quality-of-Service, Quality-of-Accuracy, and
        # Quality-of-Delay for each request under each model. This can give us a sense
        # of how the algorithms view each model and provide insight into how delay is
        # being computed against the actual delay.
        self.env = Environment(config=config)
        qos_pred_data = defaultdict(list)
        for u in self.env.requests:
            s = self.env.req_service(u)
            for m in self.env.models_for_service(s):
                qos_pred_data["QoS"].append(QoS_coeff(u, s, m, self.env))
                qos_pred_data["QoA"].append(quality_of_accuracy(u, s, m, self.env))
                qos_pred_data["QoD"].append(quality_of_delay(u, s, m, self.env))
                qos_pred_data["delay"].append(delay_fn(u, s, m, self.env))
                qos_pred_data["delay_comm"].append(delay_tran(u, s, m, self.env))
                qos_pred_data["delay_comp"].append(delay_comp(u, s, m, self.env))
                qos_pred_data["model_id"].append(m)
                qos_pred_data["service_id"].append(s)
                qos_pred_data["request_id"].append(u)
                qos_pred_data["client_id"].append(self.clients[req_2_client[u]]["name"])

        return pd.DataFrame.from_dict(qos_pred_data)

    # ================================================================================== #

    def place(self, strategy):
        """Use the designated strategy to perform placement and scheduling.

        Args:
            strategy (function): The placement/scheduling algorithm.
        """
        Server.log(f"Performing placement/scheduling with `{strategy.__name__}`.")
        self.placement_decisions = None
        self.scheduling_decisions = None
        self.placement_decisions, self.scheduling_decisions = strategy(self.env)

        # Place the PyTorch models.
        for (e, s, m) in self.placement_decisions:
            if self.placement_decisions.get((e, s, m), 0) == 1:
                model_fn = Services.MODELS[s, m]["func"]
                if s == protocol.IMAGE_CLASSIFICATION:
                    self.torch_models[s, m] = ImageClassificationModel(model_fn)
                else:
                    Server.log(f"WARNING: Illegal service identifier (s={s}).")
        Server.log("Placement/scheduling finished!")

    # ================================================================================== #

    def serve(self, algorithm_label: str="N/A") -> None:

        def process_request(request: Request, client_id: int) -> Any:
            data = request.data
            req_id = request.idx
            req_service = request.req_service
            models = {m: self.scheduling_decisions.get((req_id, m), 0)
                      for m in self.env.models_for_service(req_service)}
            scheduled_model = max(models, key=models.get)
            client_name = self.clients[client_id]["name"]
            if models[scheduled_model] == 0:
                Server.log(f"Dropped request {req_id}  (from {client_name}).")
                return protocol.DROPPED, None
            else:
                Server.log(f"Processed request {req_id} (from {client_name}).")
                return self.torch_models[req_service, scheduled_model](data), scheduled_model

        def serve_thread(client_id: int):
            client_name = self.clients[client_id]["name"]
            Server.log(f"Serving {client_name}'s requests.")
            sock = self.clients[client_id]["conn"]

            # send_msg(sock, protocol.READY)
            send_msg(sock, pickle.dumps(algorithm_label))

            response = recv_msg(sock)
            while response != protocol.DONE:
                r: Request = pickle.loads(response)
                output = process_request(r, client_id)
                message = {"prediction": output[0], "served_model": output[1]}
                send_msg(sock, pickle.dumps(message))
                response = recv_msg(sock)
                # TODO: store data about serving the request.
            Server.log(f"Served all of {client_name}'s requests.")

        threads = [Thread(target=serve_thread, args=(client_id,), daemon=False)
                   for client_id in self.clients]
        for t in threads: t.start()
        for t in threads: t.join()
        Server.log("Served all requests!")

    # ================================================================================== #

    def collect_results_data(self) -> pd.DataFrame():
        data = {}

        def collect(client_id: int) -> Dict[str, Any]:
            Server.log(f"Collect data from client {self.clients[client_id]['name']}.")
            sock = self.clients[client_id]["conn"]
            send_msg(sock, protocol.READY)
            response = recv_msg(sock)
            data[client_id] = pickle.loads(response)

        def compute_qos(req_accuracy, accuracy, req_delay, delay):
            qoa = 1.0 if (accuracy >= req_accuracy) else \
                  max(0, 1 - (req_accuracy-accuracy))
            qod = 1.0 if (delay <= req_delay) else \
                  max(0, 1 - min(delay-req_delay, self.env.max_delay)/self.env.max_delay)
            return 0.5 * (qoa + qod)

        threads = [Thread(target=collect, args=(client_id,), daemon=False)
                   for client_id in self.clients]
        for t in threads: t.start()
        for t in threads: t.join()
        Server.log("Merging results data collected from clients.")

        # Merge all of the collected results data from the clients into one dictionary.
        # Note that this dictionary does not include the *real* provided accuracy (at
        # first).
        merged_results_data = defaultdict(list)
        for client_id in data:
            for var, val in data[client_id].items():
                merged_results_data[var].extend(val)
            value = next(iter(data[client_id].values()))
            name = self.clients[client_id]["name"]
            merged_results_data["client_id"].extend([name] * len(value))

        # Compute the *real* provided accuracy. We do this by simply dividing the number
        # of predictions each model got right over the number of predictions each model
        # made.
        model_rights = defaultdict(int)
        model_totals = defaultdict(int)
        for i in range(len(merged_results_data["served_model"])):
            model_id = merged_results_data["served_model"][i]
            pred = merged_results_data["served_pred"][i]
            real = merged_results_data["true_answer"][i]
            if pred == real:
                model_rights[model_id] += 1
            model_totals[model_id] += 1
        real_model_accuracies = {key: model_rights[key]/model_totals[key]
                                 for key in model_totals}

        for i in range(len(merged_results_data["served_model"])):
            model_id = merged_results_data["served_model"][i]
            merged_results_data["served_model_name"].append(ID_2_MODEL[model_id])
            merged_results_data["served_pre_accuracy"].append(self.env.accuracy(IMAGE_CLASSIFICATION, model_id))

        for i in range(len(merged_results_data["served_model"])):
            req_accuracy = merged_results_data["req_accuracy"][i]
            req_delay = merged_results_data["req_delay"][i]
            delay = merged_results_data["served_delay"][i]

            u = merged_results_data["req_id"][i]
            s = self.env.req_service(u)
            m = merged_results_data["served_model"][i]
            merged_results_data["QoS"].append(QoS_coeff(u, s, m, self.env))

            model_id = merged_results_data["served_model"][i]
            accuracy = real_model_accuracies[model_id]
            merged_results_data["served_accuracy"].append(accuracy)
            qos = compute_qos(req_accuracy, accuracy, req_delay, delay)
            merged_results_data["post-QoS"].append(qos)

        return pd.DataFrame.from_dict(merged_results_data)

    # ================================================================================== #

    @staticmethod
    def log(string):
        prefix = f"$ [{ctime()}]"
        print(f"{prefix} {string}")