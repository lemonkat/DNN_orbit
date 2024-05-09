from data_generation.orbit_generator import generate
from data_generation.partial import simulate_RK45_partial
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view as window
from numpy.lib.format import open_memmap as memmap

rng = np.random.default_rng()

dims = 2
num_sats = int(1e7)
batch_size = int(1e4)

window_size = 6

num_steps = 60
step_size = 60

windows_per_sat = (num_steps - window_size + 1)


masses = np.load("data/sol_masses.npy")[None, 0]

sol = lambda t: np.zeros((1, 2, 2))

outfile_true = memmap(
    "../big_data/RNN_00_windowed.npy", 
    mode="w+", 
    dtype=np.float32, 
    shape=(
        num_sats * windows_per_sat, 
        window_size, 
        2, 
        dims
    )
)

outfile = np.reshape(outfile_true, (num_sats // batch_size, batch_size * windows_per_sat, window_size, 2, dims))

# initialize beforehand
raw_data = np.empty((num_steps, batch_size, 2, dims), dtype=np.float32)

# raw_data[i, j] is sat J of the current batch, at time I

for i in tqdm(range(num_sats // batch_size)):
    sat_solution = simulate_RK45_partial(
        generate(batch_size, dims=2), 
        num_steps * step_size,
        sol,
        masses=masses,
        dt=10
    )

    for t in range(num_steps):
        raw_data[t] = sat_solution(t * step_size)

    windows = np.reshape(
        window(raw_data, window_size, axis=0), 
        (-1, window_size, 2, dims)
    )
    rng.shuffle(windows)

    outfile[i] = windows

# write to disk
outfile_true.flush()

print(outfile_true[0])

# RESULT:
# A single file
# Contains a huge number of examples
# each example is the state vector at 6 consecutive minutes