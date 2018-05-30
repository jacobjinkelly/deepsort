"""Parses commmand line arguments for generate.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, help="size of the dataset to generate")
parser.add_argument("--max_val", type=int, help="the maximum value in an array")
parser.add_argument("--min_length", type=int, help="the min size of array")
parser.add_argument("--max_length", type=int, help="the max size of array")

def get_config():
    return parser.parse_known_args()
