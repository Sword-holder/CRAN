# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from CNN import CNN
from RNN import RNN
from attention import Attention

class CRAN(nn.Module):

    def __init__(self, config):
        self.CNN = CNN(config)
        self.RNN = RNN()
        self.attention = Attention()

    def forward(self, x):
        cnn_out = self.CNN(x)
        rnn_out = self.RNN(x)
        x = self.attention(rnn_out, cnn_out)
        x = nn.softmax(x)
        return x
