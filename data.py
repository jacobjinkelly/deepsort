"""For downloading and processing data.
"""
import torch
from utils import device, EOS_token

def str_to_array(lst):
    temp = lst[1:-1].split(",")
    return [int(i) for i in temp]

def read_data():
    print("Reading data...")

    # Read the file and split into lines
    lines = open("data/data.txt").read().split('\n')

    size, max_val, max_length = [int(i) for i in lines[0].split("|")]

    # Split every line into input/target pairs
    pairs = [[str_to_array(lst) for lst in l.split("|")] for l in lines[1:-1]]

    print("Found %s examples" % size)

    return max_val, max_length, pairs

def tensor_from_list(lst):
    # lst.append(EOS_token)
    return torch.tensor(lst, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_list(pair[0])
    target_tensor = tensor_from_list(pair[1])
    return (input_tensor, target_tensor)
