import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb # type: ignore
from keras import models, layers
from tensorflow.keras.utils import plot_model # type: ignore

# Cargar los datos
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Vectorizar las secuencias
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Corregido: usa una tupla para la forma
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Crear el modelo
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Dividir los datos en entrenamiento y validación
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Entrenar el modelo
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=3,
                    batch_size=256,
                    validation_data=(x_val, y_val))

# Obtener los datos del historial de entrenamiento
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

# Crear una figura con dos subgráficos
plt.figure(figsize=(12, 5))

# Subgráfico 1: Pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Subgráfico 2: Precisión
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Mostrar la figura
plt.tight_layout()
plt.show()

# Evaluar el modelo en el conjunto de prueba
results = model.evaluate(x_test, y_test)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Realizar una predicción
print("Predicción para las dos primeras muestras del conjunto de prueba:")
print(model.predict(x_test[0:2]))