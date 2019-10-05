import os
from gensim.models import Word2Vec
from data_helper import load_data, get_sentences
from nltk.tokenize import word_tokenize
import torch.optim as optim
import torch.nn as nn
import torch
from model.CRAN import CRAN
from model.CNN import CNN
from config import Config

word_embedding_file = 'data/mini_ag_news/mini_word2vec.model'
train_file = 'data/mini_ag_news/mini_train.csv'
test_file = 'data/mini_ag_news/mini_test.csv'

x_train, y_train, x_test, y_test = load_data(train_file, test_file)

# 词嵌入
if not os.path.exists(word_embedding_file):
    sentences = get_sentences(train_file)
    corpus = [word_tokenize(s) for s in sentences]
    model = Word2Vec(corpus, min_count=1,size=5,workers=3, window=3, sg=1)
    model.save(word_embedding_file)
else:
    model = Word2Vec.load(word_embedding_file)

config = Config()

net = CNN(config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    i = 1
    for inputs, labels in zip(x_train, y_train):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        tokens = word_tokenize(inputs)
        input_matrix = []
        for t in tokens:
            try:
                input_matrix.append(model[t])
            except KeyError:
                pass
        
        inputs = torch.FloatTensor(input_matrix)
        outputs = net(inputs)
        labels = torch.tensor([labels - 1]).long()
        # 计算误差
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        i += 1

print('Finished Training')