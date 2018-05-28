"""This module generates the data for the sorting task.
"""
import os
import random

def generate(size, max_val, min_length, max_length):
    """Generates <size> samples for the dataset,
    """
    with open("data.txt", mode='w') as file:
        file.write("|".join([str(size), str(max_val), str(max_length)]) + "\n")
        for i in range(size):
            length = random.randint(min_length, max_length)
            lst = [0 for _ in range(length)]
            for i in range(length):
                lst[i] = random.randint(2, max_val)
            srt = sorted(lst)
            file.write(str(lst) + "|" + str(srt) + "\n")

# TODO: refactor using command line arguments
