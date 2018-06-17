"""The example experiment ran in the original tutorial.
"""

import random

from models.attn_decoder import AttnDecoder
from models.encoder import Encoder
from torch import optim

from data import read_data, tensors_from_pair
from train import train_iters
from utils import device, set_max_length

max_val, max_length, pairs = read_data()
n_iters = 3000
learning_rate = 0.01
data_dim = max_val + 1
set_max_length(max_length)
training_pairs = [tensors_from_pair(random.choice(pairs))
                  for _ in range(n_iters)]

hidden_dim = embedding_dim = 256
encoder = Encoder(input_dim=data_dim,
                  embedding_dim=embedding_dim,
                  hidden_dim=hidden_dim).to(device)
decoder = AttnDecoder(output_dim=data_dim,
                      embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim).to(device)
encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)


def run():
    train_iters(encoder, decoder, encoder_optim, decoder_optim, training_pairs,
                n_iters)
