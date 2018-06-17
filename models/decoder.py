"""A simple decoder in the seq2seq model using a gated recurrent unit (GRU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device


# noinspection PyShadowingBuiltins
class Decoder(nn.Module):
    def __init__(self, output_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=output_dim,
                                      embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout)
        self.out = nn.Linear(in_features=hidden_dim,
                             out_features=output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(input=F.relu(embedded),
                                   h_0=hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
