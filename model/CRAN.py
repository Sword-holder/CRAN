# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.CNN import CNN
from model.RNN import RNN
from model.attention import Attention
import torch.nn.functional as F

class CRAN(nn.Module):

    def __init__(self, config):
        super(CRAN, self).__init__()
        self.config = config
        self.CNN = CNN(config)
        self.RNN = RNN(config)
        self.attention = Attention(config)

    def forward(self, x):
        # 填充输入
        padding = torch.zeros(self.config.sentence_max_size - x.size(0), self.config.word_embedding_dimension)
        x = torch.cat([x, padding], dim=0)
        
        cnn_out = self.CNN(x.unsqueeze(0).unsqueeze(0))
        rnn_out = self.RNN(x.unsqueeze(0))
        x = self.attention(rnn_out, cnn_out)
        x = F.softmax(x, dim=0)
        x = x.view(1, 4)
        return x
