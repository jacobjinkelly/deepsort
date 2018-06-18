"""Simple experiment for ptr networks.
"""

from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from torch import optim
import torch.nn as nn
from data import read_data, tensors_from_pair
from utils import device, set_max_length
from train import train_iters

max_val, max_length, pairs = read_data()
n_epochs = 2
batch_size = 3
learning_rate = 0.01
grad_clip = 2
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
optimizer = optim.SGD
optimizer_params = {"lr": learning_rate}


def weight_init(module):
    """
    Initialize weights of <module>. Applied recursivly over model weights via .apply()
    """
    for parameter in module.parameters():
        nn.init.uniform_(parameter, -0.08, 0.08)


def run():
    """
    Run the experiment.
    """
    train_iters(encoder=encoder,
                decoder=decoder,
                optim=optimizer,
                optim_params=optimizer_params,
                weight_init=weight_init,
                grad_clip=grad_clip,
                is_ptr=True,
                training_pairs=training_pairs,
                n_epochs=n_epochs,
                batch_size=batch_size,
                print_every=50,
                plot_every=50,
                save_every=50)


if __name__ == "__main__":
    run()
