from typing import OrderedDict
import torch
import torch.nn as nn


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        vector_size,
        embedded_size,
        layers,
    ):
        super(DenseNetClassifier, self).__init__()

        self.input_size = input_size
        self.vector_size = vector_size
        self.embedded_size = embedded_size

        sequence = OrderedDict()
        in_size = self.input_size * self.embedded_size
        for i, size in enumerate(layers):
            sequence[f"layer_{i}"] = nn.Linear(in_size, size)
            if i < (len(layers) - 1):
                sequence[f"relu_{i}"] = nn.ReLU()
                in_size = size

        self.embedding = nn.Embedding(self.vector_size, self.embedded_size)
        self.sequential = nn.Sequential(sequence)

    def forward(self, input):
        embedded = self.embedding(input).view(
            -1, (self.input_size * self.embedded_size)
        )
        output = self.sequential(embedded)
        return torch.log_softmax(output, dim=0)
