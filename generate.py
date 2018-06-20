"""This module generates the data for the sorting task.
"""
import os
import numpy as np

from config.generate import get_config


def generate(name, size, max_val, min_length, max_length, ewc):
    """Generates <size> samples for the dataset,
    """
    if not os.path.isdir("data"):
        os.mkdir("data")
    with open("data/" + name + ".txt", mode='w') as file:
        file.write("|".join([str(size), str(max_val), str(max_length)]) + "\n")
        if ewc:
            for length in range(2, max_length + 1):
                for i in range(size):
                    if i % 10000 == 0:
                        print(length, i)
                    lst = list(np.random.randint(2, max_val, length))
                    srt = sorted(lst)
                    file.write(str(lst) + "|" + str(srt) + "\n")
        for i in range(size):
            if i % 10000 == 0:
                print(i)
            lst = list(np.random.randint(2, max_val, np.random.randint(min_length, max_length)))
            srt = sorted(lst)
            file.write(str(lst) + "|" + str(srt) + "\n")


if __name__ == "__main__":
    args, unparsed = get_config()
    generate(args.name, args.size, args.max_val, args.min_length, args.max_length, args.ewc)
