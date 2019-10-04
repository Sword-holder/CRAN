# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, config):
        self.config = config

    def forward(self, x, context):
        u = nn.tanh(x) * context
        u = nn.softmax(u)
        x = nn.softmax(x)
        x = u * x
        return x