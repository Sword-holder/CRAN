# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, config, bias=True):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=config.word_embedding_dimension, 
                            hidden_size=config.rnn_hidden_size, 
                            num_layers=config.rnn_layer_size)
        self.config = config
        self.layer_size = config.rnn_layer_size
        self.hidden_size = config.rnn_hidden_size

        self.lin = nn.Linear(self.hidden_size, config.label_num * config.cnn_output_size)

    def forward(self, x):

        h0 = Variable(torch.randn(self.layer_size, x.size(1), self.hidden_size), requires_grad=True)
        c0 = Variable(torch.randn(self.layer_size, x.size(1), self.hidden_size), requires_grad=True)

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))

        out = self.lin(final_hidden_state[-1,-1])
        # print(out.size())
        out = out.view(self.config.label_num, self.config.cnn_output_size)

        return out