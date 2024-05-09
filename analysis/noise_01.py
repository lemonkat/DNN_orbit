from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.full_sim import generate
from load_utils import load_model
import numpy as np
import tensorflow as tf
from noise_utils import add_noise
from config import FACTOR

rng = np.random.default_rng()


update_time = 60

data = generate(1000, update_time)


model = load_model(f"saved_models/ANN_06_{update_time}B")


noise_amounts = np.linspace(0.0, 0.01, 11)
print(noise_amounts)
print("model ready")


def accuracy(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2)), axis=0)


for i, amount in enumerate(noise_amounts):
    inp = add_noise(data[0], amount, amount)
    guess = np.array(model.predict(inp, verbose=0))
    #print(i, amount)
    #print(accuracy(data[0], inp))
    acc = accuracy(data[1], guess)
    acc[1] *= FACTOR

    #print(round(acc[0] * 1000) / 1000, round(acc[1] * 1000) / 1000)
    print(acc[1])

    #print()