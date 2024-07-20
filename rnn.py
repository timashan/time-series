import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # [185, 128]
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # [185, 18]
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        combined = torch.cat(
            (input_tensor, hidden_tensor), 1
        )  # [1, 57] + [1, 128] = [1, 185]

        hidden = self.i2h(combined)  # [1, 128]
        output = self.i2o(combined)  # [1, 18]
        # output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)  # [1, 57]
