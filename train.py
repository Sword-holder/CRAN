import os
from gensim.models import Word2Vec
from data_helper import load_data, get_sentences
from nltk.tokenize import word_tokenize

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

