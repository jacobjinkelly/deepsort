"""Simple experiment for ptr networks.
"""

import torch.nn as nn
from torch import optim

from data import read_data, tensors_from_pair
from models.attn_decoder import AttnDecoder
from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from train import train
from utils import device, set_max_length


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
    name = "train"
    is_ptr = True
    hidden_dim = embedding_dim = 256
    n_epochs = 1
    grad_clip = 2
    teacher_force_ratio = 0.5
    optimizer = optim.Adam
    optimizer_params = {}

    max_val, max_length, pairs = read_data(name)

    set_max_length(max_length)
    training_pairs = [tensors_from_pair(pair) for pair in pairs]

    data_dim = max_val + 1
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

    train(encoder=encoder,
          decoder=decoder,
          optim=optimizer,
          optim_params=optimizer_params,
          weight_init=weight_init,
          grad_clip=grad_clip,
          is_ptr=True,
          training_pairs=training_pairs,
          n_epochs=n_epochs,
          teacher_force_ratio=teacher_force_ratio,
          print_every=50,
          plot_every=50,
          save_every=100)
