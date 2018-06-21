"""
Miscellaneous utility functions.
as_minutes(), time_since() modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
save_checkpoint(), load_checkpoint() modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import math
import os
import time
import re

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 16

RANDOM_SEED = 0


def set_max_length(max_length):
    """
    Set the max length of an input sequence.
    """
    global MAX_LENGTH
    MAX_LENGTH = max_length


def as_minutes(s):
    """
    Returns <s> seconds in (hours, minutes, seconds) format.
    """
    h, m = math.floor(s / 3600), math.floor(s / 60)
    m, s = m - h * 60, s - m * 60
    return '%dh %dm %ds' % (h, m, s)


def time_since(since, percent):
    """
    Return time since.
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def save_checkpoint(state, model):
    """
    Save a checkpoint of the current model weights and optimizer state.
    """
    epoch, iteration = state["epoch"], state["iter"]
    root_dir = os.path.join("checkpoints", model)
    file_name = os.path.join(root_dir, "e" + str(epoch) + "i" + str(iteration) + ".ckpt")
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    torch.save(state, file_name)


def load_checkpoint(model):
    """
    Load the most recent (i.e. greatest number of iterations) checkpoint file.
    """
    root_dir = os.path.join("checkpoints", model)
    if os.path.isdir(root_dir):
        max_epoch, max_iteration = -1, -1
        argmax_file = ""
        for file_name in os.listdir(root_dir):
            try:
                pattern = re.compile(r"""e(?P<epoch>[\d]*)
                                         i(?P<iter>[\d]*)
                                         \.ckpt""", re.VERBOSE)
                match = pattern.match(file_name)
                epoch, iteration = int(match.group("epoch")), int(match.group("iter"))
                if epoch > max_epoch and iteration > max_iteration:
                    max_epoch, max_iteration = epoch, iteration
                    argmax_file = file_name
            except (ValueError, AttributeError):
                print("A file other than a checkpoint appears to be in the checkpoints folder; please remove it")
        if max_epoch >= 0 and max_iteration >= 0:
            print("Loading checkpoint file...")
            return torch.load(os.path.join(root_dir, argmax_file))
