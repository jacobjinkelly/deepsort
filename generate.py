"""This module generates the data for the sorting task.
"""
import os
import random
from config.generate import get_config

def generate(size, max_val, min_length, max_length):
    """Generates <size> samples for the dataset,
    """
    if not os.path.isdir("data"):
        os.mkdir("data")
    with open("data/data.txt", mode='w') as file:
        file.write("|".join([str(size), str(max_val), str(max_length)]) + "\n")
        for i in range(size):
            length = random.randint(min_length, max_length)
            lst = [0 for _ in range(length)]
            for i in range(length):
                lst[i] = random.randint(2, max_val)
            srt = sorted(lst)
            file.write(str(lst) + "|" + str(srt) + "\n")

if __name__ == "__main__":
    args, unparsed = get_config()
    generate(args.size, args.max_val, args.min_length, args.max_length)
