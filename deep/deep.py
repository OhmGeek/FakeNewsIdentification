import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, load_csv

# First, load the dataset
data, labels = load_csv('/home/ryan/Documents/Programming/FakeNewsIdentification/dataset.csv',
                        target_column=2, categorical_labels=True, n_classes=2)


net = tflearn.input_data(shape=[None, 1000])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(data, labels, n_epoch=21, validation_set=0.1, show_metrix=True, batch_size=32)



# Preprocessing
    # Sequence padding
    # Use the tflearn pad_sequences function (pads at zero)

# convert labels to binary vectors (2 classes: fake news, or not)

# Then define the network:
    ## input layer,
    ## embedding layer
    ## lstm layer (dropout can be 0.8 to avoid overfitting)
    ## fully connected layer - can learn non-linear combinations. use softmax. Yields probabilities
    ## Regression layer: adam optimizer, categorical crossentropy

