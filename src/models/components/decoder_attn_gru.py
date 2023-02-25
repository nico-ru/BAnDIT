import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnDecoderGRU(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, max_length=20):
        super(AttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.view(1, 1, -1)
        input = self.dropout(input)

        attn_weights = F.softmax(self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float)
