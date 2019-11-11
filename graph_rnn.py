import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence, pad_packed_sequence


class GraphRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 num_layers, has_input=True, has_output=False, output_size=None):
        super(GraphRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=0.15)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=0.15)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.Tanh(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # initialize
        self.hidden_emb = nn.Sequential(
            nn.Linear(1, self.hidden_size)
        )
        self.hidden = None  # need initialize before forward run

    def init_hidden(self, input, batch_size):
        hidden_emb = torch.cat([
            self.hidden_emb(input).view(1, batch_size, self.hidden_size),
            torch.zeros(self.num_layers - 1, batch_size, self.hidden_size).to(input)
        ])
        return hidden_emb

    def forward(self, input_raw, pack=False, input_len=None):
        # input_raw = [batch_size, seq_length, input_size]
        output_raw_emb, output_raw, output_len = None, None, None

        if self.has_input:
            input = self.input(input_raw)
            input = self.tanh(input)
        else:
            input = input_raw
        if pack:
            pass  # input = pack_sequence(input)

        # output_raw_emb = [batch_size, seq_length, output_size]
        output_raw_emb, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw_emb, output_len = pad_packed_sequence(output_raw_emb, batch_first=True)

        if self.has_output:
            output_raw = self.output(output_raw_emb)

        if pack:
            output_raw_packed = pack_padded_sequence(output_raw, lengths=output_len, batch_first=True)
            return output_raw_emb, output_raw, output_len

        # return hidden state at each time step
        return output_raw_emb, output_raw, output_len


