# Zrobić walidację krzyżową 5 na 2
# Dla każdego foldu podzielić treningowy na train i validation.
# Potem zrobić trenowanie
# Na końcu ogarnąć predicta na reszcie

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from models_utils import get_model

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")
x_train, x_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.3, random_state=0)

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)
print(images[0].shape)

model = get_model(1)

with tf.device('/device:GPU:0'):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.adam(), metrics=['accuracy'])
    history =  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=5)
    np.save("history_1.npy", history.history)

model = get_model(2)

with tf.device('/device:GPU:0'):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.adam(), metrics=['accuracy'])
    history =  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=5)
    np.save("history_2.npy", history.history)

model = get_model(3)

with tf.device('/device:GPU:0'):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.adam(), metrics=['accuracy'])
    history =  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=5)
    np.save("history_3.npy", history.history)

model = get_model(4)

with tf.device('/device:GPU:0'):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.adam(), metrics=['accuracy'])
    history =  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=5)
    np.save("history_4.npy", history.history)

model = get_model(5)

with tf.device('/device:GPU:0'):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.adam(), metrics=['accuracy'])
    history =  model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=5)
    np.save("history_5.npy", history.history)
   


# # Open the file
# with open('network_result.txt','w') as fh:
#     fh.write(f"\nModel summary:\n")
#     # Pass the file handle in as a lambda function to make it callable
#     model.summary(print_fn=lambda x: fh.write(x + '\n'))

# model.save('model.h5')