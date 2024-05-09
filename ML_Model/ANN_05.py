# Changes from ANN_04:
# includes sun and moon
# now in 3D

# importing data generators
from data_generation.advanced import simulate_RK45
from data_generation.orbit_generator import generate
from data_generation.partial import simulate_RK45_partial


# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras, optimizers
from tensorflow.keras import layers, losses, callbacks, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# problem parameters
num_sats = int(1e6)  # num satellites to simulate - REDUCE
dims = 3 # how many dimensions
tgt_time = 60 # how far ahead

# model config parameters
mid_layers = 20 # number of hidden layers
mid_layer_width = 256 # width of each hidden layer

# training parameters
train_frac = 0.8 # test-train-split ratio
epochs = 1000
batch_size = 16
vel_priority = 100

# LR parameters
start_alpha = 0.0004
patience = 3
factor = 0.5
min_alpha = 1e-10

# generate data
# uncomment S1 for re-generation, uncomment S2 for loading from file

# S1
# warning: simulations are very memory-intensive
planet_selection = np.array([0, 1, 2])

masses = np.load("data/sol_masses.npy")[planet_selection]

planet_solution = simulate_RK45(
    np.load("data/sol_state_vectors.npy")[planet_selection, ..., :dims],
    tgt_time,
    masses=masses,
    dt=1,
    lock=0
)

print("Planetary trajectories ready")

sat_solution = simulate_RK45_partial(
    generate(num_sats, dims=dims), 
    tgt_time,
    planet_solution,
    masses=masses,
    dt=1
)

print("Satellite trajectories ready")

all_data = np.empty((2, num_sats, 2, dims))

x_full = all_data[0]
y_full = all_data[1]


x_full[...] = sat_solution(0)
y_full[...] = sat_solution(tgt_time)

# free memory, these are not needed anymore
del planet_solution
del sat_solution

np.save("data/ANN_05_temp.npy", all_data)

# S2
all_data = np.load("data/ANN_05_temp.npy")
x_full = all_data[0]
y_full = all_data[1]


x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

print("Data generated")

scale_matrix = tf.convert_to_tensor(
    np.array(
        [
            [np.std(all_data[:, :, 0, :]) * dims], 
            [np.std(all_data[:, :, 1, :]) * dims]
        ], 
        dtype=np.float32
    )
)

# does all normalization work
# might be simpler way with preset weights, but this works
class CustomNormalizer(layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        return inputs * self.factor

normalizer = CustomNormalizer(1 / scale_matrix)
inv_normalizer = CustomNormalizer(scale_matrix)


# main model
# a lot of reshaping and normalization surrounding the dense core
model = Sequential(
    [
        layers.InputLayer(input_shape=(2, dims)),
        normalizer,
        layers.Flatten(),
        *[layers.Dense(mid_layer_width, activation="relu") for i in range(mid_layers)],
        layers.Dense(2 * dims),
        layers.Reshape((2, dims)),
        inv_normalizer
    ]
)

# printed-out accuracy
def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# reference model - assumes r' constant
def reference_model(x):
    y = np.copy(x)
    y[:, 0] += x[:, 1] * tgt_time
    return y


print(accuracy(y_full, reference_model(x_full)))


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("", epoch,[float(x) for x in [
            *accuracy(y_train[:1000], self.model(x_train[:1000])), 
            *accuracy(y_test[:1000], self.model(x_test[:1000]))
        ]])


# prioritize velocity, via magnification
loss_scale = tf.convert_to_tensor(
    np.array(
        [[
            [1] * dims, 
            [vel_priority] * dims
        ]], 
        dtype=np.float32
    )
)

def custom_loss(y_true, y_pred):
    return losses.mean_squared_error(
        y_true * loss_scale, 
        y_pred * loss_scale
    )

model.compile(
    optimizer=optimizers.Adam(learning_rate=start_alpha),
    loss=custom_loss
)

print("Model ready...")

print(model.summary())

print("Training begins!")

model.fit(
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