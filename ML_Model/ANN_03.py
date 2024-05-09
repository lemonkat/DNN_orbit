# some experiments
# also not final at all

# importing data generators
from data_generation.orbit_generator import generate
from data_generation.partial import simulate_RK45_partial


# other libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# constants
num_sats = 5000  # num satellites to simulate
sample_num = 1  # num recordings per satellite
sample_interval = 60  # interval between recordings

dims = 2
train_frac = 0.8 # test-train-split ratio
alpha = 0.0001
epochs = 512

tgt_time = 60 # how far ahead

mid_layers = 16 # number of hidden layers
mid_layer_width = 256 # width of each hidden layer

#pos_vel_balance = tf.convert_to_tensor([0.8, 0.2]) # how much to weight position vs velocity

batch_size = 512 # reduce this number if you run out of memory

# generate data
# uncomment S1 for re-generation, uncomment S2 for loading from file

# S1
earth_mass = np.load("data/sol_masses.npy")[0]
# initial_state = generate(num_sats, dims=dims)

# sat_solution = simulate_RK45_partial(
#     initial_state, 
#     tgt_time + sample_num * sample_interval,
#     lambda t: np.zeros((1, 2, dims)),
#     masses = np.array([earth_mass])
# )

# print("Sim complete")

# all_data = np.empty((2, sample_num * num_sats, 2, dims))

# x_full = all_data[0]
# y_full = all_data[1]

# for i in range(sample_num):
#     idx = i * num_sats
#     t = sample_interval * i
#     x_full[idx : idx + num_sats] = sat_solution(t)
#     y_full[idx : idx + num_sats] = sat_solution(t + tgt_time)

# np.save("data/ANN_03_temp.npy", all_data)

# # S2
x_full, y_full = np.load("data/ANN_03_temp.npy")

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

print("Data generated")

# normalization
def norm(train, test):
    std = np.std(train, axis=0)
    train /= std
    test /= std

    return std

x_std = norm(x_train, x_test)
y_std = norm(y_train, y_test)

# model goal: given state vector at time T, return state vector at time T + 1 min
model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=(2, dims)),
        layers.Reshape([2 * dims]),
        *[layers.Dense(mid_layer_width, activation="relu") for i in range(mid_layers)],
        layers.Dense(2 * dims), 
        layers.Reshape((2, dims))
    ]
)

def loss(y_true, y_pred):
    #return accuracy(y_true, y_pred) * pos_vel_balance
    return K.mean((y_true - y_pred) ** 2, axis=(0, 1, 2))

def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# so that the model prints out seperate accuracies for position and velocity
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            epoch,
                [
                float(x) for x in [
                    *accuracy(
                        y_train[::1000] * y_std, 
                        self.model(x_train[::1000]) * y_std
                    ), 
                    *accuracy(
                        y_test * y_std, 
                        self.model(x_test) * y_std
                    )
                ]
            ]
        )


model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=alpha),
    loss="mean_squared_error"
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

print("Training done, ploting...")

def predict(model, x):
    x /= x_std
    y = model(x)
    return y * y_std

def plot_accuracy(num_steps, sol):
    result = np.empty((2, num_steps))

    sim_state = sat_solution(0)

    for i in range(num_steps):
        sim_state = predict(model, sim_state)
        pos_acc, vel_acc = accuracy(
            sim_state, 
            sat_solution(tgt_time * i)
        )

        result[0, i] = pos_acc
        result[1, i] = vel_acc

    return result

import matplotlib.pyplot as plt

test_num_steps = 100

test_solution = simulate_RK45_partial(
    generate(100, dims=dims), 
    tgt_time * test_num_steps,
    lambda t: np.zeros((1, 2, dims)),
    masses = np.array([earth_mass])
)

acc_log = plot_accuracy(test_num_steps, test_solution)

plt.plot(acc_log[0], color=[0, 0, 1])
plt.plot(acc_log[1], color=[1, 0, 0])
plt.show()