import numpy as np
from scipy.spatial.transform import Rotation
import random
from math import pi
from data_generation_2.core import GRAV_CONSTANT
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

deg_to_rad = pi / 180

rng = np.random.default_rng()


# defaults to earth
def get_state_vectors(elements, k=np.load("data/sol_masses.npy")[0] * GRAV_CONSTANT, dims=3):
	# SMA, ECC, INC, LAN, ARG_PE, TA

	if dims == 2:
		elements_2 = np.zeros((elements.shape[0], 6))
		elements_2[:, :2] = elements[:, :2]
		elements_2[:, 4:] = elements[:, 2:]
		elements = elements_2

	result = np.empty((len(elements), 2, 3))

	for i, params in enumerate(elements):
		SMA, ECC, INC, LAN, ARG_PE, TA = params

		alt = SMA / (1 + ECC * np.cos(TA))
		pos = alt * np.array((np.cos(TA), np.sin(TA), 0))

		vel = np.sqrt(k / SMA) * np.array((-np.sin(TA), ECC + np.cos(TA), 0))

		R = Rotation.from_euler("ZXZ", [-ARG_PE, -INC, -LAN]).as_matrix()

		result[i, 0] = pos @ R
		result[i, 1] = vel @ R

	return result[:, :, :dims]

def get_random_params(num, dims=3):
	if dims == 3:
		result = np.empty((num, 6))
		for row in result:

			row[0] = random.uniform(100, 1000) + 6378.14
			row[1] = random.random() * min((1 - 100/row[0]), 0.25) # don't hit the atmosphere

			# just angles
			row[2] = random.random() * 2 * pi
			row[3] = random.random() * 2 * pi
			row[4] = random.random() * 2 * pi
			row[5] = random.random() * 2 * pi

		return result

	result = np.empty((num, 4))
	for row in result:

		row[0] = random.uniform(100, 1000) + 6378.14
		row[1] = random.random() * min((1 - 100/row[0]), 0.25) # don't hit the atmosphere

		# just angles
		row[2] = random.random() * 2 * pi
		row[3] = random.random() * 2 * pi

	return result

def generate(num, k=398451.84, dims=3):
	params = get_random_params(num, dims=dims)
	return get_state_vectors(params, k=k, dims=dims)

def generate_v2(num, dims=3):
	k = 398451.84
	params = np.empty((num, 6))
	params[:, 0] = rng.uniform(100, 1000, num) + 6378.14
	max_ecc = 1 - 100 / params[:, 0]
	max_ecc[max_ecc > 0.25] = 0.25
	params[:, 1] = rng.random(num) * max_ecc

	if dims == 3:
		params[:, 2] = rng.uniform(0, 2 * pi, num)
		params[:, 3] = rng.uniform(0, 2 * pi, num)

	else:
		params[:, 2] = 0
		params[:, 3] = 0

	params[:, 4] = rng.uniform(0, 2 * pi, num)
	params[:, 5] = rng.uniform(0, 2 * pi, num)

	result = np.empty((num, 2, dims))

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

	return result




if __name__ == "__main__":
	print(get_state_vectors(np.array([[
		6.794275888738486E+03, 
		8.707205233195385E-04, 
		4.912300099720557E+01 * deg_to_rad, 
		9.352997424739738E+01 * deg_to_rad,
		1.153256926900435E+02 * deg_to_rad, 
		1.610249958522944E+02 * deg_to_rad
	]])))

