import pandas as pd

# 读取类别映射
class_map = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

def load_data_file(file_name):
    data = pd.read_csv(file_name, header=None)
    data[3] = data[1] + '.' + data[2]
    x = data[3].tolist()
    y = data[0].tolist()
    return x, y
    

def load_data(train_file, test_file):
    x_train, y_train = load_data_file(train_file)
    x_test, y_test = load_data_file(test_file)
    return x_train, y_train, x_test, y_test

def get_sentences(file_name):
    data = pd.read_csv(file_name, header=None)
    sentences = data[1].tolist()
    for s in data[2]:
        sentences.append(s)
    return sentences


if __name__ == '__main__':
    train_file = 'data/mini_ag_news/mini_test.csv'
    test_file = 'data/mini_ag_news/mini_test.csv'
    x_train, y_train, x_test, y_test = load_data(train_file, test_file)
    print(get_sentences(train_file))