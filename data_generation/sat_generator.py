import numpy as np
import pickle
from data_generation.partial import simulate_RK45_partial
from data_generation.horizons_loader import request_state_vector
from data_generation.orbit_generator import generate

# keep this the same
# file: year 2023, planet indices 0, 1, 2, 6, 1 year duration, 64-bit floats
with open("data/sol_solution_2023_0126_1Y_64.p", "rb") as file:
	sol_solution = pickle.load(file)

sol_masses = np.load("data/sol_masses.npy")
selection = np.array([0]) # earth, moon, sun, jupiter
sol_masses = sol_masses[selection]

# To simulate some satellites:
# set sat_ids to a list of the Horizons IDs of each satellite
# set the file opened at the end to where you want the simulation to be stored
# when you un-pickle that file, you will get a function F such that
# F(t)[i] is the state of satellite i (index in sat_ids, not Horizions ID)
# at time t (seconds past Jan 1, 2023, 12:00)

# sat_ids = [-48, -125544]
# satellites = np.array([request_state_vector(sat_id) for sat_id in sat_ids])

satellites = generate(1000) + sol_solution(0)[0]

sat_solution = simulate_RK45_partial(
	satellites, 
	365 * 24 * 60, 
	lambda t: sol_solution(t)[selection], 
	masses=sol_masses, 
	dt=60
)

with open("data/1000_randoms_1y_flat.p", "wb") as file:
	pickle.dump(sat_solution, file)