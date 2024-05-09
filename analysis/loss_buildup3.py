from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors
from load_utils import load_model, accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import sys
from noise_utils import add_noise

generate_new_data = False

if generate_new_data:
    initial_state = generate_state_vectors(100, dims=3, flat=True)
    np.save("data/A_S_UpdateTemp.npy", initial_state)
else:
    initial_state = np.load("data/A_S_UpdateTemp.npy")

planet_selection = np.array([0, 1, 2])
masses = np.load("data/sol_masses.npy")[planet_selection]
planet_state = np.load("data/sol_state_vectors.npy")[planet_selection]

from config import FACTOR

update_time = -2

if update_time == -1:
    update_time = int(input("Prediction target time: "))

elif update_time == -2:
    update_time = int(sys.argv[1])


cycles = 100

future = cycles * update_time

planet_solution = simulate_RK45(
    planet_state,
    future,
    masses=masses,
    dt=1,
    lock=0
)

sat_solution = simulate_RK45_partial(
    initial_state,
    future,
    planet_solution,
    dt=1,
    masses=masses
)


model = load_model(f"saved_models/ANN_06_{update_time}B")

state = initial_state

errors = []

for i in tqdm(range(cycles)):
    time = i * update_time
    state = model.predict(state, verbose=0)
    real = sat_solution(time)
    errors.append(accuracy(real, state))

errors = np.array(errors)
print(errors)

plt.rcParams.update({'font.size': 20})

fig, (axp, axv) = plt.subplots(1, 2, figsize=(10, 5))
x_values = np.arange(cycles) * update_time / 60

axp.set_xlabel("Time, min")
axv.set_xlabel("Time, min")

axp.set_ylabel("Position error, km")
axv.set_ylabel("Velocity error, km/s")

axp.set_title("Position Error")
axv.set_title("Velocity Error")

#axp.set_yscale("log")
#axv.set_yscale("log")

color = [0, 0, 1]

axp.plot(x_values, errors[:, 0], color=color)
axv.plot(x_values, errors[:, 1] * FACTOR, color=color)

fig.tight_layout()
#plt.show()

plt.savefig("figures/loss_buildup_normal_6.png", format="png")