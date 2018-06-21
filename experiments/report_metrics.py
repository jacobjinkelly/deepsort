"""
Script for reproducing results.
"""
import numpy as np

from data import read_data, tensors_from_pair
from evaluate import evaluate
from models.attn_decoder import AttnDecoder
from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from utils import load_checkpoint, device, RANDOM_SEED
from metrics import is_permutation, nondecreasing


def run():
    """
    Run the experiment.
    """
    is_ptr = False
    np.random.seed(RANDOM_SEED)
    max_val, max_length, pairs = read_data(name="test")
    np.random.shuffle(pairs)
    training_pairs = [tensors_from_pair(pair) for pair in pairs]

    data_dim = max_val + 1
    hidden_dim = embedding_dim = 256

    encoder = Encoder(input_dim=data_dim,
                      embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim).to(device)
    if is_ptr:
        decoder = PtrDecoder(output_dim=data_dim,
                             embedding_dim=embedding_dim,
                             hidden_dim=hidden_dim).to(device)
    else:
        decoder = AttnDecoder(output_dim=data_dim,
                              embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim).to(device)

    checkpoint = load_checkpoint("ptr" if is_ptr else "vanilla")
    if checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    else:
        print("Count not find checkpoint file.")

    permutation_count, nondecreasing_count = 0, 0
    for i in range(len(training_pairs)):
        input_tensor, target_tensor = training_pairs[i]
        output_tensor = evaluate(encoder=encoder,
                                 decoder=decoder,
                                 input_tensor=training_pairs[i][0],
                                 is_ptr=is_ptr)
        target, output = list(np.asarray(input_tensor.data).squeeze()), output_tensor[:-1]
        if is_permutation(target, output):
            permutation_count += 1
        if nondecreasing(output) == 0:
            nondecreasing_count += 1
    print("Permutation: %s" % (permutation_count / len(training_pairs)))
    print("Nondecreasing: %s" % (nondecreasing_count / len(training_pairs)))
