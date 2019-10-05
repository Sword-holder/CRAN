# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 1, (3, config.word_embedding_dimension))
        self.conv2 = nn.Conv2d(1, 1, (4, config.word_embedding_dimension))
        self.conv3 = nn.Conv2d(1, 1, (5, config.word_embedding_dimension))
        self.max_pool1 = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.max_pool2 = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.max_pool3 = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, config.cnn_output_size)

    def forward(self, x):

        padding = torch.zeros(self.config.sentence_max_size - x.size(0), self.config.word_embedding_dimension)
        x = torch.cat([x, padding], dim=0)
        x = x.unsqueeze(0).unsqueeze(0)

        # 卷积层
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        # max pooling层
        x1 = self.max_pool1(x1)
        x2 = self.max_pool2(x2)
        x3 = self.max_pool3(x3)

        # 拼接
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(1, -1)

        # 全连接层
        x = self.linear1(x)
        x = x.view(1, self.config.cnn_output_size)

        return x

