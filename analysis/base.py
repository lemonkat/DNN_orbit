from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors
from load_utils import load_model

import tensorflow as tf
import numpy as np

update_time = 600

model = load_model(f"saved_models/ANN_06_{update_time}")


initial_state = generate_state_vectors(100)

planet_selection = np.array([0, 1, 2])
masses = np.load("data/sol_masses.npy")[planet_selection]
planet_state = np.load("data/sol_state_vectors.npy")[planet_selection]

planet_solution = simulate_RK45(
    planet_state,
    3600 * 24, # 1 day
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

