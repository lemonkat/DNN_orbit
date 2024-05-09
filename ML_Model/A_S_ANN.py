# Changes from ANN_05:
# optimized data generator

# importing data generator
from data_generation.full_sim import generate

from load_utils import norm, inv_norm, custom_loss

# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras, optimizers
from tensorflow.keras import layers, losses, callbacks, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# problem parameters
num_sats = 1000000  # num satellites to simulate
dims = 3  # how many dimensions
tgt_time = -1  # how far ahead
generate_new_data = True

# model config parameters
mid_layers = 4  # number of hidden layers
mid_layer_width = 256  # width of each hidden layer

# training parameters
train_frac = 0.8  # test-train-split ratio
epochs = 10  # enough, don't change
batch_size = 16
vel_priority = 100

# LR parameters
start_alpha = 0.0001
patience = 2
factor = 0.5
min_alpha = 1e-10

if tgt_time == -1:
    tgt_time = int(input("Prediction target time: "))

# generate data
if generate_new_data:
    all_data = generate(num_sats, tgt_time, dims=dims)
    np.save("data/ANN_06_temp.npy", all_data)

else:
    all_data = np.load("data/ANN_06_temp.npy")

x_full = all_data[0]
y_full = all_data[1]

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

print("Data generated -100 epochs")

# main model
# a lot of reshaping and normalization surrounding the dense core
model = Sequential(
    [
        layers.InputLayer(input_shape=(2, dims)),

        layers.Flatten(),
        norm,
        *[layers.Dense(mid_layer_width, activation="relu") for i in range(mid_layers)],
        layers.Dense(2 * dims),
        inv_norm,
        layers.Reshape((2, dims)),

    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=start_alpha),
    loss=custom_loss
)

print("Model ready...")

print(model.summary())
print("Training begins!")


# printed-out accuracy
def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("", epoch, [float(x) for x in [
            *accuracy(y_train[:1000], self.model.predict(x_train[:1000])),
            *accuracy(y_test[:1000], self.model.predict(x_test[:1000])),
            *accuracy(y_full[:1000], self.model.predict(x_full[:1000]))
        ]])


result = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[
        CustomCallback(),
        callbacks.ReduceLROnPlateau(
            min_lr=min_alpha,
            factor=factor,
            patience=patience
        ),
    ],
    verbose=1
)

model.save(f'saved_models/ANN_06_{tgt_time}')

print(result.history)