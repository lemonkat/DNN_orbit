import numpy as np

rng = np.random.default_rng()

def add_noise(data, pos, vel):
    std = np.std(data, axis=(0, 2))
    result = np.copy(data)

    result[:, 0] += rng.normal(0, std[0] * pos, result[:, 0].shape)
    result[:, 1] += rng.normal(0, std[1] * vel, result[:, 0].shape)

    return result
