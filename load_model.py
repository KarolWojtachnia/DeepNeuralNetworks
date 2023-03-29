import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_images_from_path(path):
    images = []
    labels = []
    for file in os.listdir(path):
        print(f"Loading {file}")
        # if file[0] == "2":
        #     break
        images.append(tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        label = int(file[0])
        labels.append(label)
    return images, labels

model = tf.keras.models.load_model('model.h5')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy())
model.summary()

test_images, test_labels = load_images_from_path("Spectrograms/")

# Convert to np array to fix this issue:
# Failed to find data adapter that can handle input: 
# (<class 'list'> containing values of types {"<class 'numpy.ndarray'>"}), (<class 'list'> containing values of types {"<class 'int'>"})
# test_images = np.array(test_images)
# test_labels = np.array(test_labels)

# res = model.predict(test_images)
# print(res)
# results = model.evaluate(test_images, test_labels, batch_size=128)
# print('Results: {} {}'.format(results))

# Not working