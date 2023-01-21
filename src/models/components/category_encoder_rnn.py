import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryEncoderRNN(nn.Module):
    def __init__(self, category_size, input_size, hidden_size):
        super(CategoryEncoderRNN, self).__init__()
        self.category_size = category_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(
            self.category_size + self.hidden_size, self.hidden_size, batch_first=True
        )

    def forward(self, category, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        categories = category.repeat(input.size(1), 1).unsqueeze(0)
        combined = torch.cat((categories, embedded), dim=2)
        output, hidden = self.gru(combined, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float)
