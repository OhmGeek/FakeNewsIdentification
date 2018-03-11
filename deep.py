from shallow.shallow_data_processor import ShallowDataProcessor
from deep.word_representation import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import Adam
def main():
    dp = ShallowDataProcessor()
    filename = 'dataset.csv'
    dp.read_dataset_from_file(filename)
    stopword_list = ["I" , "a" , "about" ,"an" ,"are" ,"as" ,"at" ,"be" ,"by" ,"com" ,"for" ,"from","how","in" ,"is" ,"it" ,"of" ,"on" ,"or","that","the" ,"this","to" ,"was" ,"what" ,"when","where","who" ,"will","with","the","www"]
    dataset = dp.process(stopwords=stopword_list)
    max_data=len(dataset)
    pivot = int(0.6 * max_data)


    # First, we are going to tokenize the dataset.
    # Then convert to sequences
    # Finally we will create an embedding from the word2vec method.
    tokenizer = Tokenizer(num_words=15000)
    tokenizer.fit_on_texts([data[1] for data in dataset])
    sequences = tokenizer.texts_to_sequences([data[1] for data in dataset[1:]])
    word_index = tokenizer.word_index
    print("Found " + str(word_index) + " unique tokens")

    data_x = sequence.pad_sequences(sequences, maxlen=1000)

    labels = [int(doc[2]) for doc in dataset[1:]]
    data_y = to_categorical(labels) # Labels

    train_x = data_x[:pivot]
    train_y = data_y[:pivot]

    test_x = data_x[pivot+1:]
    test_y = data_y[pivot+1:]
    # Now deal with embeddings, using word2vec
    word2vec = Word2Vec()
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in word2vec.vec_model:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = word2vec.vec_model[word]

    print(train_x.shape)
    print(train_y.shape)


    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=1000,
                            trainable=False))
    model.add(Dropout(0.1))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, epochs=8, batch_size=32, validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=1)
    print("Final Accuracy is: " + str(score))
    model.save('deep-model.h5')
if __name__ == '__main__':
    main()
