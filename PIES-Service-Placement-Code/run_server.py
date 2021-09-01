import argparse
import socket

from src.testbed.server import *
from src.testbed.protocol import *

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mc_runs", default=1, type=int, help="Number of trials.")
    parser.add_argument("-n", "--n_clients", default=1, type=int, 
                        help="Number of clients.")
    parser.add_argument("--local", action="store_true")    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    host = "localhost" if args.local else socket.gethostbyname(socket.gethostname())
    server = Server(host=host)
    server.run(args.n_clients, args.mc_runs)