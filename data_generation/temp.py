import numpy as np
import pickle

with open("data/1000_randoms_1y_flat.p", "rb") as file:
	sol_raw = pickle.load(file)
	
shift = np.load("data/sol_state_vectors.npy")[0, :, :-1]

def sol(t):
	return sol_raw(t) - shift

data = np.empty((2, 1000, 4))
data[0] = np.reshape(sol(0), (1000, 4))
data[1] = np.reshape(sol(60), (1000, 4))

np.save("data/1_min_predictions_flat.npy", data)