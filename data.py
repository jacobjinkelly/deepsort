"""For downloading and processing data.
"""
import torch
from utils import device, EOS_token

from collections import deque

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

def ind_from_pair(input_lst, output_lst):
    """Converts items of <output_lst> to correspond to indices of <input_lst>.
    If input_lst has duplicates, indices will appear in same order.
    >>> input_lst = [-1, -3, -2]
    >>> output_lst = [-3, -2, -1]
    >>> ind_from_pair(input_lst, output_lst)
    >>> output_lst
    [1, 2, 0]
    >>> input_lst = ['a', 'a', 'b', 'c']
    >>> output_lst = ['a', 'a', 'b', 'c']
    >>> ind_from_pair(input_lst, output_lst)
    >>> output_lst
    [0, 1, 2, 3]
    """
    tbl = {}
    for i in range(len(input_lst)):
        if tbl.get(input_lst[i]):
            tbl[input_lst[i]].append(i)
        else:
            q = deque()
            q.append(i)
            tbl[input_lst[i]] = q
    for i in range(len(output_lst)):
        output_lst[i] = tbl[output_lst[i]].popleft()

def tensor_from_list(lst):
    # lst.append(EOS_token)
    return torch.tensor(lst, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair, ptr=False):
    """Returns tensor (input, output) pair. If <ptr>, output_tensor is returned
    in pointer form, as required for models.prt_decoder_rnn
    """
    input_tensor = tensor_from_list(pair[0])
    if ptr:
        ind_from_pair(pair)
    target_tensor = tensor_from_list(pair[1])
    return (input_tensor, target_tensor)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
