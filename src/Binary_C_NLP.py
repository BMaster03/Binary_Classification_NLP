# Program for binary classification 

import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import imdb # type: ignore
from keras import models, layers
from tensorflow.keras.utils import plot_model #type: ignore

def binary_classification():

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print(train_data[0])

    word_index = imdb.get_word_index()

    list(word_index.items())[:5]


    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    list(reverse_word_index.items())[:5]

    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

    print(decoded_review)

    print(train_labels[0])


    def vectorize_sequences(sequences, dimension=10000):

        results = np.zeros((len(sequences), dimension))
        
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.

        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    print(x_train[0])

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=3,
                        batch_size=256,
                        validation_data=(x_val, y_val))


    history_dict = history.history
    history_dict.keys()

    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    model.evaluate(x_test, y_test)
    results = model.evaluate(x_test, y_test)
    print(results)
    model.predict(x_test[0:2])

    plt.show()


