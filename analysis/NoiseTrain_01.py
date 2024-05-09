from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.full_sim import generate
from load_utils import load_model
from noise_01 import add_noise
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


epochs = 500
batch_size = 16

rng = np.random.default_rng()


update_time = 60

all_data = generate(5000, update_time)


model = load_model(f"saved_models/ANN_06_{update_time}")


x_full = all_data[0]
y_full = all_data[1]

x_full *= rng.normal(1, 0.002, x_full.shape)

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)


# printed-out accuracy
def accuracy(y_true, y_pred):
    return K.mean(K.sqrt(K.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


# so that the model prints out separate accuracies for position and velocity
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("", epoch,[float(x) for x in [
            *accuracy(y_train[:1000], self.model.predict(x_train[:1000])),
            *accuracy(y_test[:1000], self.model.predict(x_test[:1000])),
            *accuracy(y_full[:1000], self.model.predict(x_full[:1000]))
        ]])


model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[CustomCallback()],
)