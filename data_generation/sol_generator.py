from data_generation.advanced import simulate_RK45
import numpy as np
import pickle

# only need to run once

state = np.load("data/sol_state_vectors.npy")
masses = np.load("data/sol_masses.npy")

# only earth, moon, sun, and jupiter
selection = np.array([0, 1, 2, 6])
state = state[selection]
masses = masses[selection]

time = 1 * 365 * 24 * 60 * 60

solution = simulate_RK45(state, time, masses=masses, dt=60)

with open("data/sol_solution_2023_0126_1Y_64.p", "wb") as file:
	pickle.dump(solution, file)
