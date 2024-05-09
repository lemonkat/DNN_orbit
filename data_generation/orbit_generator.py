import numpy as np
from scipy.spatial.transform import Rotation
from math import pi

rng = np.random.default_rng()

# generates low-orbit state vectors

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


def generate_state_vectors(num, dims=3, flat=False):
    k = 398451.84
    params = np.empty((num, 6))
    params[:, 0] = rng.uniform(100, 1000, num) + 6378.14
    max_ecc = 1 - 100 / params[:, 0]
    max_ecc[max_ecc > 0.25] = 0.25
    params[:, 1] = rng.random(num) * max_ecc

    if dims == 3 and not flat:
        params[:, 2] = rng.uniform(0, 2 * pi, num)
        params[:, 3] = rng.uniform(0, 2 * pi, num)

    else:
        params[:, 2] = 0
        params[:, 3] = 0

    params[:, 4] = rng.uniform(0, 2 * pi, num)
    params[:, 5] = rng.uniform(0, 2 * pi, num)

    result = np.empty((num, 2, 3))

    alt = params[:, 0] / (1 + params[:, 1] * np.cos(params[:, 5]))

    v = np.zeros((num, 3))
    v[:, 0] = np.cos(params[:, 5])
    v[:, 1] = np.sin(params[:, 5])
    pos = alt[..., None] * v

    v = np.zeros((num, 3))
    v[:, 0] = -np.sin(params[:, 5])
    v[:, 1] = params[:, 1] + np.cos(params[:, 5])

    vel = np.sqrt(k / params[:, 0])[..., None] * v

    for i in tqdm(range(num)):
        R = Rotation.from_euler("ZXZ", [-params[i, 4], -params[i, 2], -params[i, 3]]).as_matrix()

        result[i, 0] = pos[i] @ R
        result[i, 1] = vel[i] @ R

    return result[..., :dims]

if __name__ == "__main__":
    print(generate_state_vectors(10, dims=2))


