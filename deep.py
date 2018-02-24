from shallow.shallow_data_processor import ShallowDataProcessor
from deep.word_representation import Word2Vec
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np

def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    
    wtv = Word2Vec()
    
    train_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[1:]])
    train_y = np.array([row[2] for row in dataset[1:]])
    print(train_x)
    print(train_y)
    max_length = len(train_x[2])
    train_x = pad_sequences(train_x, max_length, value=0.0)
    train_y = to_categorical(train_y, nb_classes=2)

    net = tflearn.input_data([None, max_length])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(train_x, train_y, show_metric=True, batch_size=32)

if __name__ == '__main__':
    main()