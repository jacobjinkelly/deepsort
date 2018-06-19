"""
Evaluate the model.
"""
import numpy as np

from data import read_data, tensors_from_pair
from utils import load_checkpoint, device
from models.ptr_decoder import PtrDecoder
from models.encoder import Encoder
from evaluate import evaluate

max_val, max_length, pairs = read_data()
np.random.shuffle(pairs)
training_pairs = [tensors_from_pair(pair) for pair in pairs]

data_dim = max_val + 1

hidden_dim = embedding_dim = 256

encoder = Encoder(input_dim=data_dim,
                  embedding_dim=embedding_dim,
                  hidden_dim=hidden_dim).to(device)
decoder = PtrDecoder(output_dim=data_dim,
                     embedding_dim=embedding_dim,
                     hidden_dim=hidden_dim).to(device)


def run():
    """
    Run the experiment.
    """
    checkpoint = load_checkpoint()
    if checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    else:
        print("Count not find checkpoint file.")

    for i in range(10):
        input_tensor, target_tensor = training_pairs[i]
        output_tensor, _ = evaluate(encoder=encoder,
                                    decoder=decoder,
                                    input_tensor=training_pairs[i][0],
                                    is_ptr=True)
        print(list(np.asarray(input_tensor.data).squeeze()), output_tensor[:-1])
