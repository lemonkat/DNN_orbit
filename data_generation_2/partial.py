from data_generation_2.sim import get_accel_vectors, GRAV_CONSTANT
from data_generation_2.advanced import OutputReshaper
import numpy as np
import scipy as sp

try:
    from tqdm import tqdm

except ImportError:
    def tqdm(x):
        return x


# This module is for simulators that are given
# the pre-computed states of the solar system
# by simulating only the satellites, we gain a lot of speed


# If you have the solar system trajectories pre-computed, use this
# should be quite a bit faster


def get_accel_vectors_partial(mass_positions, tgt_positions, masses=None):
    if masses is None:
        masses = np.ones(len(mass_positions))

    mass_idx, tgt_idx = np.meshgrid(
        np.arange(len(mass_positions)), np.arange(len(tgt_positions)), indexing="ij"
    )

    distance_vectors = mass_positions[mass_idx] - tgt_positions[tgt_idx]
    dist_squareds = np.sum(distance_vectors**2, axis=2)

    pulls = masses[mass_idx] * GRAV_CONSTANT / dist_squareds

    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    return np.sum(pull_vectors, axis=0)


def simulate_RK45_partial(state, time, sol, dt = 60, masses=None):

    shape = state.shape

    if masses is None:
        masses = np.ones(shape[0])

    max_size = 10000

    if shape[0] > max_size:
        # break problem into chunks and solve each chunk
        print("Splitting simulation...")
        solutions = []
        state_split = np.array_split(state, shape[0] // max_size, axis=0)
        for i in tqdm(range(shape[0] // max_size)):
            solutions.append(
                simulate_RK45_partial(
                    state_split[i],
                    time,
                    sol,
                    dt=dt,
                    masses=masses
                )
            )


        return combine_solutions(*solutions)


    def sim_func(t, flat):
        mass_state = sol(t)
        tgt_state = np.reshape(flat, shape)

        result = np.empty_like(tgt_state)
        result[:, 0] = tgt_state[:, 1]
        result[:, 1] = get_accel_vectors_partial(
            mass_state[:, 0], tgt_state[:, 0], masses=masses
        )

        return result.flatten()

    result = sp.integrate.solve_ivp(
        sim_func,
        (0, time),
        state.flatten(),
        dense_output=True,
        method="RK45",
        max_step = dt
    )

    solution = result.sol

    return OutputReshaper(solution, shape)

# use this to "combine" multiple solutions for simultaneous rendering
class combine_solutions:
    def __init__(self, *solutions):
        self.solutions = solutions

    def __call__(self, t):
        return np.concatenate([sol(t) for sol in self.solutions])