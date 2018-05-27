"""The example experiment ran in the original tutorial.
"""

from models.encoder_rnn import EncoderRNN
from models.attn_decoder_rnn import AttnDecoderRNN
from data import prepare_data, tensors_from_pair
from utils import device
from train import train_iters
import random

input_lang, output_lang, pairs = prepare_data("eng", "fra", True)
n_iters = 75000
training_pairs = \
            [tensors_from_pair(input_lang, output_lang, random.choice(pairs))
             for _ in range(n_iters)]

hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                                    dropout_p=0.1).to(device)
def run():
    train_iters(encoder, attn_decoder, training_pairs, n_iters, print_every=1000,
                                                                plot_every=1000)
