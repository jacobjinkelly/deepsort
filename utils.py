"""Miscellaneous utility functions.
"""
import math
import os
import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 16


def set_max_length(max_length):
    """
    Set the max length of an input sequence.
    """
    global MAX_LENGTH
    MAX_LENGTH = max_length


def as_minutes(s):
    """
    Returns the number of seconds in minutes.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    """
    Return time since.
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def save_checkpoint(state, iteration, save_every):
    """
    Save a checkpoint of the current model weights, deleting old ones.
    """
    old_file_name = "checkpoints/" + str(iteration - 6 * save_every) + ".ckpt"
    current_file_name = "checkpoints/" + str(iteration) + ".ckpt"
    if os.path.isfile(old_file_name):
        os.remove(old_file_name)
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(state, current_file_name)


def load_checkpoint():
    """
    Load the most recent (i.e. greatest number of iterations) checkpoint file.
    """
    if os.path.isdir("checkpoints"):
        max_iter = -1
        max_iter_file = ""
        for file_name in os.listdir("checkpoints"):
            try:
                iteration = int(file_name.split(".")[0])
                if file_name.endswith("ckpt") and iteration > max_iter:
                    max_iter = iteration
                    max_iter_file = file_name
            except ValueError:
                print("A file other than a checkpoint appears to be in the " +
                      "<checkpoints> folder; please remove it")
        if max_iter > 0:
            print("Loading checkpoint file...")
            return torch.load("checkpoints/" + max_iter_file)
