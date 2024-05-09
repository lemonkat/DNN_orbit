from load_utils import load_model, accuracy
from data_generation.full_sim import generate
from noise_utils import add_noise
import sys
import numpy as np
from config import FACTOR

tgt_time = int(sys.argv[1])

if len(sys.argv) > 2:
    noise = float(sys.argv[2])
else:
    noise = 0

#model = load_model(f"saved_models/ANN_06_{tgt_time}_FINAL")
model = load_model(f"saved_models/ANN_06_{tgt_time}B")
#model = load_model(f"saved_models/ANN_07_{tgt_time}")
#model = load_model(f"saved_models/ANN_08_{tgt_time}")
data = generate(1000, tgt_time)
diff = model.predict(add_noise(data[0], noise, noise)) - data[1]
diff[1] *= FACTOR

mag = np.sqrt(np.sum(diff ** 2, axis=2))

#print(np.mean(mag, axis=0), np.std(mag, axis=0) * 3)

#breakpoint()
print(np.mean(mag, axis=0))