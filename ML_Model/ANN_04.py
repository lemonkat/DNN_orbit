# some experiments
# also not final at all

# importing data generators
from data_generation.orbit_generator import generate
from data_generation.partial import simulate_RK45_partial
from data_generation.sim import GRAV_CONSTANT

# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# constants
num_sats = 100000  # num satellites to simulate

dims = 2
train_frac = 0.8 # test-train-split ratio
#alpha = 0.000001
alpha = 0.0001
epochs = 20000

tgt_time = 60 # how far ahead

mid_layers = 20 # number of hidden layers
mid_layer_width = 256 # width of each hidden layer

#pos_vel_balance = tf.convert_to_tensor([0.8, 0.2]) # how much to weight position vs velocity

batch_size = 16 # reduce this number if you run out of memory

# generate data
# uncomment S1 for re-generation, uncomment S2 for loading from file

# S1
earth_mass = np.load("data/sol_masses.npy")[0]
initial_state = generate(num_sats, dims=dims)

sat_solution = simulate_RK45_partial(
    initial_state, 
    tgt_time,
    lambda t: np.zeros((1, 2, dims)),
    masses = np.array([earth_mass]),
    dt=1
)

print("Sim complete")

all_data = np.empty((2, num_sats, 2, dims))

x_full = all_data[0]
y_full = all_data[1]


x_full[...] = sat_solution(0)
y_full[...] = sat_solution(tgt_time)

np.save("data/ANN_04_temp.npy", all_data)

# # S2
#x_full, y_full = np.load("data/ANN_04_temp.npy")

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

model = tf.keras.Sequential(
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


def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# reference model - assumes r' constant
def reference_model(x):
    y = np.copy(x)
    y[:, 0] += x[:, 1] * tgt_time
    return y


print(accuracy(y_full, reference_model(x_full)))


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("", epoch,[float(x) for x in [
            *accuracy(y_train[:100], self.model(x_train[:100])), 
            *accuracy(y_test[:100], self.model(x_test[:100]))
        ]])

loss_scale = tf.convert_to_tensor(np.array([[[1] * dims, [50] * dims]], dtype=np.float32))

def custom_loss(y_true, y_pred):
    # errors = K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2))
    # return K.mean(errors[0] + errors[1] * 10)
    return keras.losses.mean_squared_error(y_true * loss_scale, y_pred * loss_scale)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=alpha),
    #loss="mean_squared_error"
    loss=custom_loss
)

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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            min_lr = 1e-10,
            factor=0.5,
            patience=2
        ),
    ],
    verbose=1
)

print("Training done, ploting...")