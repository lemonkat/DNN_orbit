import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import convert_to_tensor as ctf
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import backend as K

data = np.load("../data_generation/data/1_min_predictions_flat.npy").astype(np.float32)[:, :4]

# GOAL: Given position and velocity on Jan 1, 2023, 12:00, 
# return position and velocity on Jan 1, 2023, 12:01

print(data.shape)

x_full = data[0]
y_full = data[1]

print(data)


# train-test split
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, train_size = 0.5)

# normalize data
def normalize(train, test):
	pos_scale = np.mean(np.sqrt(np.sum(train[:, :2] ** 2, axis=1)))
	vel_scale = np.mean(np.sqrt(np.sum(train[:, 2:] ** 2, axis=1)))

	train[:, :2] /= pos_scale
	train[:, 2:] /= vel_scale

	test[:, :2] /= pos_scale
	test[:, 2:] /= vel_scale

	return ctf(pos_scale), ctf(vel_scale)

	#return 1, 1

x_pos_scale, x_vel_scale = normalize(x_train, x_test)
y_pos_scale, y_vel_scale = normalize(y_train, y_test)

# create model
model = keras.Sequential(
    [
        layers.Dense(4, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(10, activation="sigmoid"),
        layers.Dense(4),
    ]
)

def predict(model, x):
	x[:, :2] /= x_pos_scale
	x[:, 2:] /= x_vel_scale
	y = np.array(model(x))
	y[:, :2] *= y_pos_scale
	y[:, 2:] *= y_vel_scale
	return y

# mean magnitude of difference
def loss(y_true, y_pred):
	pos_diff = (y_true[:, :2] - y_pred[:, :2])
	pos_mag = K.sqrt(K.sum(pos_diff ** 2, axis=1)) * y_pos_scale

	vel_diff = (y_true[:, 2:] - y_pred[:, 2:])
	vel_mag = K.sqrt(K.sum(vel_diff ** 2, axis=1)) * y_vel_scale
	
	return K.mean(pos_mag) + K.mean(vel_mag)

model.compile(
	optimizer=tf.optimizers.Adam(learning_rate=0.001),
	loss=loss
	#loss="mean_squared_error"
)

model.fit(
	x_train, 
	y_train, 
	batch_size=64,
	epochs=1000,
	validation_data=(x_test, y_test),
)

pred = predict(model, x_train[:1])
real = y_train[:1] * np.array((y_pos_scale, y_pos_scale, y_vel_scale, y_vel_scale))
print(pred, real, pred - real, np.sqrt(np.sum((pred - real) ** 2, axis=1)))




