"""
A simple encoder in the seq2seq model using a LSTM.
"""
import torch
import torch.nn as nn

from utils import device


# ignore that parameter <input> for <forward()> shadows built-in keyword input
# noinspection PyShadowingBuiltins
class Encoder(nn.Module):
    """
    A simple encoder in the seq2seq model using a LSTM.
    """
    def __init__(self, input_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers)

    def forward(self, input, hidden):
        """
        The forward pass of the encoder.
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the weights of the encoder.
        """
        return (torch.zeros(1, 1, self.hidden_dim, device=device), torch.zeros(1, 1, self.hidden_dim, device=device))
