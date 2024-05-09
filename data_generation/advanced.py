import numpy as np
import scipy as sp
from data_generation.sim import get_accel_vectors

# solution to not being able to pickle lambdas
class OutputReshaper:
    def __init__(self, func, shape):
        self.func = func
        self.shape = shape

    def __call__(self, *args):
        return np.reshape(self.func(*args), self.shape)

class OutputLocker:
    def __init__(self, func, lock_idx = 0):
        self.func = func
        self.lock_idx = lock_idx

    def __call__(self, *args):
        result = self.func(*args)
        return result - result[self.lock_idx]


# Runga-Kutta advanced simulation
# note: dt is maximum dt, use it to ensure accuracy
def simulate_RK45(state, time, masses=None, dt=3600, lock=None):
    shape = state.shape

    def sim_func(t, flat):
        state = np.reshape(flat, shape)
        result = np.empty_like(state)
        result[:, 0] = state[:, 1]
        result[:, 1] = get_accel_vectors(state[:, 0], masses=masses)
        return result.flatten()


    result = sp.integrate.solve_ivp(
        sim_func,
        (0, time),
        state.flatten(),
        dense_output=True,
        method="RK45",
        max_step=dt,
    )

    assert result.status == 0

    if lock is not None:
        return OutputLocker(OutputReshaper(result.sol, shape), lock)

    return OutputReshaper(result.sol, shape)