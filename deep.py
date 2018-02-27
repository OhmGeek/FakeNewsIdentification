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
    
    wtv = Word2Vec()
    
    train_x = np.array([np.array(wtv.corpus_to_vec_list(row[1])) for row in dataset[1:100]])
    train_y = np.array([row[2] for row in dataset[1:100]])

    max_length = 300
    train_x = sequence.pad_sequences(train_x, maxlen=max_length)
    train_y = to_categorical(train_y, num_classes=2)

    model = Sequential()
    model.add(Embedding(input_dim=max_length, output_dim=1000, input_length=300))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, epochs=30, batch_size=64)
   
if __name__ == '__main__':
    main()