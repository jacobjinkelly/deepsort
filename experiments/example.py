"""The example experiment ran in the original tutorial.
"""

from models.encoder_rnn import EncoderRNN
from models.attn_decoder_rnn import AttnDecoderRNN
from data import read_data, tensors_from_pair
from utils import device, set_max_length
from train import train_iters
import random

max_val, max_length, pairs = read_data()
n_iters = 1000
data_dim = max_val + 1
set_max_length(max_length)
training_pairs = [tensors_from_pair(random.choice(pairs))
                  for _ in range(n_iters)]

hidden_size = 256
encoder = EncoderRNN(data_dim, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, data_dim, dropout_p=0.1).to(device)
encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)
def run():
    train_iters(encoder, attn_decoder, encoder_optim, decoder_optim,
                    training_pairs, n_iters, print_every=1000, plot_every=10)
