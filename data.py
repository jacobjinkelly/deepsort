"""For downloading and processing data.
"""
import torch

from utils import device, EOS_token


def str_to_array(lst):
    """
    Converts string representation of array read in from file to Python array.
    """
    temp = lst[1:-1].split(",")
    return [int(i) for i in temp]


def read_data():
    """
    Read in data from file.
    """
    print("Reading data...")

    # Read the file and split into lines
    lines = open("data/train.txt").read().split('\n')

    size, max_val, max_length = [int(i) for i in lines[0].split("|")]

    # Split every line into input/target pairs
    pairs = [[str_to_array(lst) for lst in l.split("|")] for l in lines[1:-1]]

    print("Found %s examples" % size)

    return max_val, max_length, pairs


def tensor_from_list(lst):
    """
    Converts Python array to PyTorch tensor.
    """
    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor(lst + [EOS_token], dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair):
    """Returns Tensor (input, output) pair from Python list (input, output) pair.
    """
    return tuple(map(tensor_from_list, pair))
