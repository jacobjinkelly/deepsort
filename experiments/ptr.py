"""Simple experiment for ptr networks.
"""

from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from torch import optim
from data import read_data, tensors_from_pair
from utils import device, set_max_length
from train import train_iters

max_val, max_length, pairs = read_data()
n_epochs = 2
n_iters = len(pairs) * n_epochs
learning_rate = 0.01
data_dim = max_val + 1
set_max_length(max_length)
training_pairs = [tensors_from_pair(pair) for pair in pairs]

hidden_dim = embedding_dim = 256
encoder = Encoder(input_dim=data_dim,
                  embedding_dim=embedding_dim,
                  hidden_dim=hidden_dim).to(device)
decoder = PtrDecoder(output_dim=data_dim,
                     embedding_dim=embedding_dim,
                     hidden_dim=hidden_dim).to(device)
encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)


def run():
    """
    Run the experiment.
    """
    train_iters(encoder=encoder,
                decoder=decoder,
                encoder_optim=encoder_optim,
                decoder_optim=decoder_optim,
                is_ptr=True,
                training_pairs=training_pairs,
                n_iters=n_iters,
                print_every=50,
                plot_every=50,
                save_every=50)


if __name__ == "__main__":
    run()
