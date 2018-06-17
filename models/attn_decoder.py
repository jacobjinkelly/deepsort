import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device, MAX_LENGTH


# noinspection PyUnresolvedReferences,PyShadowingBuiltins
class AttnDecoder(nn.Module):
    """A decoder in seq2seq model using a gated recurrent unit (GRU) and attention.
    """

    def __init__(self, output_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(num_embeddings=self.output_dim,
                                      embedding_dim=embedding_dim)
        self.attn = nn.Linear(in_features=self.hidden_dim + embedding_dim,
                              out_features=self.max_length)
        self.attn_combine = nn.Linear(in_features=self.hidden_dim + embedding_dim,
                                      out_features=self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout)
        self.out = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        concat = torch.cat((embedded[0], attn_applied[0]), 1)
        attn_combine = self.attn_combine(concat).unsqueeze(0)

        output, hidden = self.lstm(input=F.relu(attn_combine),
                                   h_0=hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
