from data_generation.orbit_generator import generate
from data_generation.partial import simulate_RK45_partial
import numpy as np
from tqdm import tqdm

dims = 2
num_sats = int(1e7)
batch_size = int(1e3)

num_steps = 60
step_size = 60

window_size = 5

shape = (num_sats, num_steps, 2, dims)

masses = np.load("data/sol_masses.npy")[None, 0]

sol = lambda t: np.zeros((1, 2, 2))

trajectories = np.empty(shape, dtype=np.float32)

filepath = "../big_data/RNN_00_raw.npy"

for i in tqdm(range(0, num_sats, batch_size)):
    sat_solution = simulate_RK45_partial(
        generate(batch_size, dims=2), 
        num_steps * step_size,
        sol,
        masses=masses,
        dt=10
    )

    for t in range(num_steps):
        trajectories[i : i + batch_size, t] = sat_solution(t * step_size)

np.save(filepath, trajectories)