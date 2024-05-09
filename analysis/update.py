from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors
from load_utils import load_model

import tensorflow as tf
import numpy as np
import pygame as pg

update_time = 600

dims = 2

initial_state = generate_state_vectors(100, dims=dims)

planet_selection = np.array([0, 1, 2])
masses = np.load("data/sol_masses.npy")[planet_selection]
planet_state = np.load("data/sol_state_vectors.npy")[planet_selection][..., :dims]

planet_solution = simulate_RK45(
    planet_state,
    3600 * 24,
    masses=masses,
    dt=1,
    lock=0
)

sat_solution = simulate_RK45_partial(
    initial_state,
    3600 * 24,
    planet_solution,
    dt=1,
    masses=masses
)



model = load_model(f"saved_models/ANN_06_{update_time}_FINAL")


# model
# any random satellite and put it in the model 60 seconds
# position and velocity after 60 seconds
# optional - cross-check on simulation to see error
# ideal number fed back into the model, repeat 1000 times

state = sat_solution(0)[:1]
coordinates = []

for i in range(3600 * 24 // update_time):
    time = i * update_time
    guess = model.predict(state, verbose=0)[..., :dims]
    # resetting
    state = sat_solution(time)[:1]
    x = guess[0, 0, 0]
    y = guess[0, 0, 1]
    coordinates.append((i, time, float(x), float(y)))

print("hi")
print(coordinates)
'''
sat_icon = pg.transform.scale(pg.image.load("data/sat_icon.png"), (50, 50))
pg.font.init()
font = pg.font.Font(pg.font.get_default_font(), 15)
'''