import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
import numpy as np
from config import FACTOR

tgt_time = -2

if tgt_time == -1:
    tgt_time = int(input("Prediction target time: "))

elif tgt_time == -2: # command-line argument
    tgt_time = int(sys.argv[1])

main_file = f"saved_models/ANN_07_{tgt_time}"

with open(main_file + "/history.txt", "r") as file:
    hist = eval(file.read())

pos_error = []
vel_error = []
with open(main_file + "/log.txt", "r") as file:
    for line in file:
        result = eval(line)
        pos_error.append(result[-2])
        vel_error.append(result[-1])

plt.rcParams.update({'font.size': 20})


# I'm never using matplotlib ever again...

fig, (axp, axv, axlr) = plt.subplots(1, 3, figsize=(15, 5))

axp.set_xlabel("Epoch")
axv.set_xlabel("Epoch")
axlr.set_xlabel("Epoch")

axp.set_ylabel("Position error, km")
axv.set_ylabel("Velocity error, km/s")
axlr.set_ylabel("Learning Rate")

axp.set_title("Position Test Error")
axv.set_title("Velocity Test Error")
axlr.set_title("Learning Rate")

axp.set_yscale("log")
axv.set_yscale("log")
axlr.set_yscale("log")

axp.yaxis.set_major_formatter(ticker.ScalarFormatter())
axp.set_yticks([5, 10, 20, 40])

axv.yaxis.set_major_formatter(ticker.ScalarFormatter())
axv.set_yticks([0.03125, 0.0625, 0.125, 0.25])

axp.yaxis.set_minor_formatter(ticker.NullFormatter())
axv.yaxis.set_minor_formatter(ticker.NullFormatter())

color = [0, 0, 1]

axp.plot(pos_error, color=color)
axv.plot(np.array(vel_error) * FACTOR, color=color)
axlr.plot(hist["lr"], color=color)

fig.tight_layout()
#plt.show()
plt.savefig("figures/noisetrain_loss.png", format="png")