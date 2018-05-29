"""Miscellaneous utility functions.
"""
import torch
import time
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 16

def set_max_length(max_length):
    MAX_LENGTH = max_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def save_checkpoint(state, iter, save_every):
    old_file_name = "checkpoints/" + str(iter - 6 * save_every) + ".ckpt"
    current_file_name = "checkpoints/" + str(iter) + ".ckpt"
    if os.path.isfile(old_file_name):
        os.remove(old_file_name)
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(state, current_file_name)

def load_checkpoint():
    if os.path.isdir("checkpoints"):
        max_iter = -1
        max_iter_file = ""
        for file_name in os.listdir("checkpoints"):
            try:
                iter = int(file_name.split(".")[0])
            except ValueError:
                print("A file other than a checkpoint appears to be in the " +
                        "<checkpoints> folder; please remove it")
            if file_name.endswith("ckpt") and iter > max_iter:
                max_iter = iter
                max_iter_file = file_name
        if max_iter > 0:
            print("Loading checkpoint file...")
            return torch.load("checkpoints/" + max_iter_file)
