"""Parses commmand line arguments for generate.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="the name of the dataset (e.g. train, val, test)")
parser.add_argument("--size", type=int, help="size of the dataset to generate")
parser.add_argument("--max_val", type=int, help="the maximum value in an array")
parser.add_argument("--min_length", type=int, help="the min size of array")
parser.add_argument("--max_length", type=int, help="the max size of array")
parser.add_argument("--ewc", action="store_true", help="generate sequence of datasets for ewc")


def get_config():
    """
    Get command line arguments.
    """
    return parser.parse_known_args()
