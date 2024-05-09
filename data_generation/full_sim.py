# the final simulator
# everything you need to generate single or multi iteration training data

import numpy as np
from scipy.integrate import solve_ivp
from data_generation.sim import get_accel_vectors
from data_generation.partial import get_accel_vectors_partial
from data_generation.orbit_generator import generate_state_vectors
from data_generation.advanced import OutputReshaper, OutputLocker
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


def generate(num_sats, tgt_time=60, body_selection=np.array([0, 1, 2]), dims=3, start_state=None):
    # Step 1: get planet solution

    if start_state is None:
        start_state = generate_state_vectors(num_sats, dims=dims)

        
    planet_masses = np.load("data/sol_masses.npy")[body_selection]
    planet_state_vectors = np.load("data/sol_state_vectors.npy")[body_selection][..., :dims]

    planet_shape = planet_state_vectors.shape

    def planet_sim_func(t, flat):
        state = np.reshape(flat, planet_shape)
        result = np.empty_like(state)
        result[:, 0] = state[:, 1]
        result[:, 1] = get_accel_vectors(state[:, 0], masses=planet_masses)
        return result.flatten()

    planet_result = solve_ivp(
        planet_sim_func,
        (0, tgt_time),
        planet_state_vectors.flatten(),
        method="RK45",
        dense_output=True,
        max_step=1,
    )

    planet_solution = OutputLocker(OutputReshaper(planet_result.sol, planet_shape), 0)

    # Step 2: simulate satellites

    full_result = np.empty((2, num_sats, 2, dims))

    full_result[0] = start_state

    max_size = 10000

    split_num = math.ceil(num_sats / max_size)

    state_split = np.array_split(full_result[0], split_num, axis=0)
    result_split = np.array_split(full_result[1], split_num, axis=0)

    for i in tqdm(range(split_num)):
        shape = state_split[i].shape

        def sat_sim_func(t, flat):
            mass_state = planet_solution(t)
            tgt_state = np.reshape(flat, shape)

            result = np.empty_like(tgt_state)
            result[:, 0] = tgt_state[:, 1]
            result[:, 1] = get_accel_vectors_partial(
                mass_state[:, 0], tgt_state[:, 0], masses=planet_masses
            )

            return result.flatten()

        sat_result = solve_ivp(
            sat_sim_func,
            (0, tgt_time),
            state_split[i].flatten(),
            method="RK45",
            max_step=1,
        ).y[..., -1]

        result_split[i][...] = np.reshape(sat_result, shape)

    return full_result[..., :dims]

