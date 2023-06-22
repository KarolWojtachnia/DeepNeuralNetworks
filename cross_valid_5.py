import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from models_utils import get_model
import tensorflow as tf

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((n_splits * n_repeats))

X = np.load("data/images.npy")
y = np.load("data/labels.npy")

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    print(fold_id)
    x_train, x_valid, y_train, y_valid = train_test_split(X[train], y[train], stratify=y[train], test_size=0.3, random_state=0)
    model = get_model(5)
    y_train_encoded = tf.keras.utils.to_categorical(y_train)
    y_valid_encoded = tf.keras.utils.to_categorical(y_valid)
    with tf.device('/device:GPU:0'):
        model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.fit(x_train, y_train_encoded, validation_data=(x_valid, y_valid_encoded), batch_size=50, epochs=5)
        y_pred = model.predict(X[test])
        scores[fold_id] = accuracy_score(y[test], y_pred)

np.save("cross_valid_5.npy")