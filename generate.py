"""This module generates the data for the sorting task.
"""
import os
import numpy as np

from config.generate import get_config


def generate(size, max_val, min_length, max_length):
    """Generates <size> samples for the dataset,
    """
    if not os.path.isdir("data"):
        os.mkdir("data")
    with open("data/train.txt", mode='w') as file:
        file.write("|".join([str(size), str(max_val), str(max_length)]) + "\n")
        for j in range(size):
            if j % 10000 == 0:
                print(j)
            lst = list(np.random.randint(2, max_val, np.random.randint(min_length, max_length)))
            srt = sorted(lst)
            file.write(str(lst) + "|" + str(srt) + "\n")


if __name__ == "__main__":
    args, unparsed = get_config()
    generate(args.size, args.max_val, args.min_length, args.max_length)
