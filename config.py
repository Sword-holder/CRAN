
class Config(object):
    # 词嵌入的维度
    word_embedding_dimension = 5
    # 句子的最大长度
    sentence_max_size = 40
    # 文本类别数量
    cnn_output_size = 4

    rnn_hidden_size = 10
    rnn_layer_size = 2

    label_num = 4