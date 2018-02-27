from shallow.shallow_data_processor import ShallowDataProcessor
from deep.word_representation import Word2Vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    
    wtv = Word2Vec()
    
    train_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[1:5]])
    train_y = np.array([row[2] for row in dataset[1:5]])

    test_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[6:10]])
    test_y = np.array([row[2] for row in dataset[6:10]])
    max_length = len(train_x[2])    
    train_x = pad_sequences(train_x, max_length, value=0.0)
    train_y = to_categorical(train_y, num_classes=2)
    print(train_x)
    print(train_y)

    test_x = pad_sequences(train_x, max_length, value=0.0)
    test_y = to_categorical(train_y, num_classes=2)

    model = Sequential()
    model.add(Embedding(input_dim=max_length, output_dim=256))
    model.add(LSTM(input_dim=256, units=128))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10)

if __name__ == '__main__':
    main()