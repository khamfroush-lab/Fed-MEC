import argparse
import random
import os

from PIL import Image
from socket import *
from src.testbed.client import *
from src.testbed.const import *
from src.testbed.protocol import Request
from src.testbed.util import *
from time import ctime
from typing import List

DEFAULT_DIR = os.path.join("~", "Development", "torch_datasets", "imagenet-2012")
DEFAULT_DIR = os.path.expanduser(DEFAULT_DIR)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default=DEFAULT_DIR, type=str, 
                        help="Image directory.")
    parser.add_argument("-n", "--n_requests", default=100, type=int, 
                        help="Number of requests.")
    parser.add_argument("--host", default=DEFAULT_HOST, type=str, 
                        help="Server's IP address.")
    parser.add_argument("-p", "--portno", default=DEFAULT_PORT, type=str,
                        help="Port number for the host.")
    parser.add_argument("--high_acc", action="store_true")
    parser.add_argument("--low_delay", action="store_true")
    parser.add_argument("--random_profile", action="store_true")
    parser.add_argument("--start_index", default=0, type=int, help="Start index for data.")
    parser.add_argument("--end_index", default=50000, type=int, help="End index for data.")
    return parser.parse_args()
        

if __name__ == "__main__":
    args = get_args()
    client = Client(image_dir=args.dir, 
                    hostname=args.host, 
                    portno=args.portno,
                    high_acc=args.high_acc,
                    low_delay=args.low_delay,
                    random_profile=args.random_profile,
                    start_index=args.start_index,
                    end_index=args.end_index)
    client.run(n_requests=args.n_requests)