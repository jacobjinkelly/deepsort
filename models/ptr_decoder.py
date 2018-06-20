"""
A decoer in seq2seq model using pointer attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device


# ignore unresolved reference for torch.stack, torch.cat (bug in PyTorch)
# ignore that parameter <input> for <forward()> shadows built-in keyword input
# noinspection PyUnresolvedReferences,PyShadowingBuiltins
class PtrDecoder(nn.Module):
    """A decoder in seq2seq model using pointer attention.
    """
    def __init__(self, output_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0.1):
        super(PtrDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=self.output_dim,
                                      embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers)
        self.attn = nn.Linear(in_features=self.hidden_dim * 2,
                              out_features=self.hidden_dim)
        self.out = nn.Linear(in_features=self.hidden_dim,
                             out_features=1)

    def forward(self, input, hidden, encoder_outputs, encoder_inputs):
        """
        The forward pass of the decoder.
        """
        length = encoder_outputs.shape[0]  # the length of the input sequence

        # feed input through embedding, dropout, and LSTM cell to get new hidden state
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)

        # concatenate to get context vector (this can probably be done more efficiently)
        # use hidden[0] = hidden state, in contrast to hidden[1] = cell state
        stacked_hidden = torch.squeeze(torch.stack([hidden[0] for _ in range(length)]))
        context = torch.cat((encoder_outputs, stacked_hidden), dim=1)

        attn_weights = F.log_softmax(self.out(F.tanh(self.attn(context))), dim=0)

        # convert log probability vector over inputs to log probability vector over all possible outputs
        # outputs not present in output have probability 0, so log probability of -inf
        output = torch.full((self.output_dim,), float("-inf"))
        for i in range(len(attn_weights)):
            output[encoder_inputs[i]] = attn_weights[i]

        return torch.unsqueeze(output, 0), hidden, attn_weights

    def init_hidden(self):
        """
        Initialize the hidden state of the decoder.
        """
        return torch.zeros(1, 1, self.hidden_dim, device=device)
