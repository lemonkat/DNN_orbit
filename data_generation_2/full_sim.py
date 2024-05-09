# the final simulator
# everything you need to generate training data

import numpy as np
from scipy.integrate import solve_ivp
from data_generation_2.orbit_generator import generate_state_vectors
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

GRAV_CONSTANT = 6.674 * (10**-20)

# given the positions of things, determine their accelerations
def get_accel_vectors(positions, masses=None):
    num_planets = len(positions)

    # masses defaults to all 1kg
    if masses is None:
        masses = np.ones(num_planets)

    each = np.arange(num_planets)

    # get distances all at once
    p1s, p2s = np.meshgrid(each, each, indexing="ij")

    # distance_vectors[i, j] = the vector from planet i to planet j
    distance_vectors = positions[p2s] - positions[p1s]

    # dist_squareds[i, j] = the square of the distance from planet i to planet j
    dist_squareds = np.sum(distance_vectors ** 2, axis=2)

    ignore = (
        dist_squareds == 0
    )  # things pulling on themselves break things, but they should be zero

    dist_squareds[ignore] = 1

    # calculate pull strengths
    # pulls[i, j] = the magnitude of the
    # free-fall acceleration of planet i caused by planet j
    pulls = masses[p2s] * GRAV_CONSTANT / dist_squareds

    # get pull vectors and sum them for total accel
    # pull_vectors[i, j] is the vector of the free-fall
    # acceleration of planet i caused by planet j
    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    pull_vectors[ignore] = 0

    # sum up pulls from each other planet
    # accel_vectors[i] = the combined free-fall acceleration of planet j
    accel_vectors = np.sum(pull_vectors, axis=1)

    return accel_vectors

# given the position of some particles and masses, determine the acceleration of the particles
def get_accel_vectors_partial(mass_positions, tgt_positions, masses=None):
    if masses is None:
        masses = np.ones(len(mass_positions))

    mass_idx, tgt_idx = np.meshgrid(
        np.arange(len(mass_positions)), 
        np.arange(len(tgt_positions)), 
        indexing="ij"
    )

    distance_vectors = mass_positions[mass_idx] - tgt_positions[tgt_idx]
    dist_squareds = np.sum(distance_vectors**2, axis=2)

    pulls = masses[mass_idx] * GRAV_CONSTANT / dist_squareds

    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    return np.sum(pull_vectors, axis=0)

# solution to not being able to pickle lambdas
# given a function, return a function which calls it and reshapes the output
class OutputReshaper:
    def __init__(self, func, shape):
        self.func = func
        self.shape = shape

    def __call__(self, *args):
        return np.reshape(self.func(*args), self.shape)

# given a function, return a function which calls it and shifts the output
class OutputLocker:
    def __init__(self, func, lock_idx=0):
        self.func = func
        self.lock_idx = lock_idx

    def __call__(self, *args):
        result = self.func(*args)
        return result - result[self.lock_idx]

# main simulator
def generate(num_sats, tgt_time=60, planet_selection=np.array([0, 1, 2]), dims=3):
    # Step 1: get planet solution
    planet_masses = np.load("data/sol_masses.npy")[planet_selection]
    planet_state_vectors = np.load("data/sol_state_vectors.npy")[planet_selection][..., :dims]

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


    full_result[0] = generate_state_vectors(num_sats, dims=dims)

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