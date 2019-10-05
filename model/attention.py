# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, config):
        super(Attention, self).__init__()
        
        self.config = config

    def forward(self, input, context):
        out = []
        
        for x in input:
            u = F.tanh(x) * context
            u = F.softmax(u, dim=0)
            x = F.softmax(x, dim=0)
            x = u * x
            x = x.sum()
            out.append(x)
        out = torch.FloatTensor(out)
        return out