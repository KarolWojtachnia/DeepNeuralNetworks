import json
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from models_utils import get_model
import json

## 2 wykresy, po 2 linie

filenames = ["results/history/history_1.npy", 
             "results/history/history_2.npy", 
             "results/history/history_3.npy", 
             "results/history/history_4.npy", 
             "results/history/history_5.npy"]

for idx, filename in enumerate(filenames):
    history = np.load(filename, allow_pickle=True)
    res = json.loads(str(history).replace("'", "\""))
    accuracy = res['accuracy']
    val_accuracy = res['val_accuracy']
    loss = res['loss']
    val_loss = res['val_loss']

    # summarize history for accuracy
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.xticks([0, 1, 2, 3, 4])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"results/plots/accuracy_plot_{idx+1}_layers.png")
    plt.clf()

    # summarize history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xticks([0, 1, 2, 3, 4])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"results/plots/loss_plot_{idx+1}_layers.png")
    plt.clf()