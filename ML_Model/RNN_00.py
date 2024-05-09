# RNN attempt 1

# importing data generators
from load_utils import norm, inv_norm, custom_loss


# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
#from tensorflow.data import Dataset
#from numpy.lib.stride_tricks import sliding_window_view as window

print("Libraries loaded")

# parameters

dims = 2
alpha = 0.001

run_up = 5

output_shape = 2, dims
input_shape = run_up, *output_shape

test_train_split = 0.8

mid_layers = 10
mid_layer_width = 256

batch_size = 256

epochs = 256

# load and organize data

# all_data = [
#     np.lib.format.open_memmap(
#         f"../big_data/RNN_00/{i}.npy", 
#         mode="r", 
#         dtype=np.float32
#     ) 
#     for i in range(100)
# ]

# x_full = [arr[:, :-1] for arr in all_data]
# y_full = [arr[:, -1] for arr in all_data]

all_data = np.lib.format.open_memmap(
    "../big_data/RNN_00_windowed.npy",
    mode="r",
    dtype=np.float32
)


all_data = all_data[::100]

x_full = all_data[:, :-1]
y_full = all_data[:, -1]

split_index = int(len(all_data) * test_train_split)

x_train = x_full[:split_index]
x_test = x_full[split_index:]
y_train = y_full[:split_index]
y_test = y_full[split_index:]

print("Data loaded")

model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=(run_up, 2, dims)),
        layers.Reshape((run_up))
        norm,
        layers.Flatten(),
        *[layers.Dense(mid_layer_width, activation="relu") for i in range(mid_layers)],
        layers.Dense(2 * dims), 
        layers.Reshape(output_shape), 
        inv_norm
    ]
)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=alpha),
    loss="mean_squared_error"
)

def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            epoch,
                [
                float(x) for x in [
                    *accuracy(
                        y_train[::10000], 
                        self.model(x_train[::10000])
                    ), 
                    *accuracy(
                        y_test[::10000], 
                        self.model(x_test[::10000])
                    )
                ]
            ]
        )

print(model.summary())

print("Training begins!")

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[CustomCallback()],
    verbose=1
)