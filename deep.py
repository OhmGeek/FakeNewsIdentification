from shallow.shallow_data_processor import ShallowDataProcessor
from deep.word_representation import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    dataset = dp.process()
    pivot = int(0.6 * len(dataset))
    wtv = Word2Vec()
    print("Start training set x")
    train_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[1:pivot]])
    print("Now training set y")
    train_y = np.array([row[2] for row in dataset[1:pivot]])
    print("Now test set x")
    test_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[pivot+1:]])
    print("Now test set y")
    test_y = np.array([row[2] for row in dataset[pivot+1:]])

    print("Train set X: ")


    max_length = 300

    train_x = sequence.pad_sequences(train_x, maxlen=max_length)
    train_y = to_categorical(train_y, num_classes=2)

    test_x = sequence.pad_sequences(test_x, maxlen=max_length)
    test_y = to_categorical(test_y, num_classes=2)

    print(train_x.shape)
    print(test_x.shape)

    model = Sequential()
    model.add(Embedding(input_dim=max_length, output_dim=1000, input_length=300))
    model.add(Dropout(0.1))
    model.add(LSTM(120))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # print(model.summary())
    model.fit(train_x, train_y, epochs=100, batch_size=32)
    score = model.evaluate(test_x, test_y, verbose=1)
    print("Final Accuracy is: " + str(score))
if __name__ == '__main__':
    main()
