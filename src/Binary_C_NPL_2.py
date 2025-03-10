# Binary Classification Model for Movie Reviews using Dense Neural Network and Deep Learning

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb  # type: ignore
from keras import models, layers
from tensorflow.keras.utils import plot_model  # type: ignore

class BinaryClassification:
    
    def __init__(self):
        """
            Load the IMDB dataset and initialize the word index.
        """
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words=10000)
        self.word_index = imdb.get_word_index()
        self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])
        self.model = self.create_model()

    def create_model(self):
        """
            Create a sequential model for binary classification.
        """
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def vectorize_sequences(self, sequences, dimension=10000):
        """
            Convert sequences of integers to binary matrix representation.
        """
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def decode_review(self, review):
        """
            Decode a review from the dataset using the reverse word index.
        """
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in review])

    def prepare_data(self):
        """
            Prepare and vectorize training and test data.
        """
        x_train = self.vectorize_sequences(self.train_data)
        x_test = self.vectorize_sequences(self.test_data)
        y_train = np.asarray(self.train_labels).astype('float32')
        y_test = np.asarray(self.test_labels).astype('float32')
        return x_train, y_train, x_test, y_test

    def train_model(self, x_train, y_train):
        """
            Train the model with the training data.
        """
        x_val = x_train[:10000]
        partial_x_train = x_train[10000:]
        y_val = y_train[:10000]
        partial_y_train = y_train[10000:]

        history = self.model.fit(partial_x_train,
                                 partial_y_train,
                                 epochs=3,
                                 batch_size=256,
                                 validation_data=(x_val, y_val))
        return history

    def plot_history(self, history):
        """
            Plot the training and validation loss and accuracy.
        """
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
        plt.show()

    def evaluate_model(self, x_test, y_test):
        """
            Evaluate the model on the test data and return predictions.
        """
        results = self.model.evaluate(x_test, y_test)
        print(results)
        predictions = self.model.predict(x_test[0:2])
        return predictions

    def run(self):
        """
            Execute the workflow for binary classification.
        """
        x_train, y_train, x_test, y_test = self.prepare_data()
        history = self.train_model(x_train, y_train)
        self.plot_history(history)
        self.evaluate_model(x_test, y_test)