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

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")
x_train, x_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.3, random_state=0)

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)
print(images[0].shape)
for _ in range(3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build()
    model.summary()

# model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=50, epochs=1)

# # Open the file
# with open('network_result.txt','w') as fh:
#     fh.write(f"\nModel summary:\n")
#     # Pass the file handle in as a lambda function to make it callable
#     model.summary(print_fn=lambda x: fh.write(x + '\n'))

# model.save('model.h5')