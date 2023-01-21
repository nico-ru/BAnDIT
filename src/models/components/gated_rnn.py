import torch
import torch.nn as nn


class GatedRNN(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(GatedRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.h2o = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.h2o(output)
        return output, hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size)
