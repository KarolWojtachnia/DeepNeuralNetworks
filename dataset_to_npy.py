import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def load_images_from_path(path):
    images = []
    labels = []
    for file in os.listdir(path):
        print(f"Loading {file}")
        images.append(tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        label = int(file[0])
        labels.append(label)
    return images, labels

images, labels = load_images_from_path("Spectrograms")
print("All the file were loaded")

images_normalised = np.array(images) / 255

with open("data/images.npy", 'wb') as file:
    np.save(file, images_normalised, allow_pickle=True)

with open("data/labels.npy", 'wb') as file:
    np.save(file, labels, allow_pickle=True)
