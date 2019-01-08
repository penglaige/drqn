import torch
import torch.nn as nn
import numpy as np

class Qnetwork(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim, trace):
        super(Dnetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.hidden_dim = hidden_dim
        self.trace = trace
        self.input_dim = 7 * 7 * 64

        self.lstm = nn.LSTM(self.input_dim, hidden_dim)

        self.hidden2qvalue = nn.Linear(hidden_dim, num_actions)
        self.hidden = self.init_hidden()

        self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, x):
        # x-size: (10, batch, 1, h, w)
        batch_size = x[0].size()[0]
        trace_list = []
        for i in range(self.trace):
            frame = x[i]
            h = self.relu(self.conv1(frame))
            h = self.relu(self.conv2(h))
            h = self.relu(self.conv2(h))
            h = h.view(h.size(0), -1)
            trace_list.append(h)
        # input size: (seq_len = 10, batch, input_size)
        input_seq = torch.cat(trace_list).view(self.trace, batch_size, -1)
        lstm_out, self.hidden = self.lstm(input_seq, self.hidden)

        x = lstm_out[-1]
        x = self.hidden2qvalue(x)

        return x
