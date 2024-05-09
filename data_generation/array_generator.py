import numpy as np
from data_generation.advanced import simulate_RK45
from data_generation.partial import simulate_RK45_partial

# relatively memory-efficient systems
# generate arrays and toss the large solution objects
# incomplete, don't use

def simulate_RK45_arr(state, time, masses=None, sim_dt=3600, lock=None, record_dt=10):
    result = np.empty((time // record_dt, *state.shape))
    solution = simulate_RK45(state, time, masses=masses, dt=sim_dt, lock=lock)

    for i, t in enumerate(range(0, time, record_dt)):
        result[i] = solution(t)

    del solution

    return result

def simulate_RK45_partial_arr(state, time, sol, sim_dt=60, masses=None, record_dt=10, out=None):
    if out is None:
        out = np.empty((time // record_dt, *state.shape))

    max_size = 10000

    if state.shape[0] > max_size:
        # break problem into chunks and solve each chunk
        print("Splitting simulation...")
        state_split = np.array_split(state, shape[0] // max_size, axis=0)
        out_split = np.array_split(out, shape[0] // max_size, axis=1)
        for i in tqdm(range(shape[0] // max_size)):
            simulate_RK45_partial_arr(
                state_split[i],
                time,
                sol,
                sim_dt=sim_dt,
                masses=masses,
                record_dt=record_dt,
                out=out_split[i]
            )

    else:
        solution = simulate_RK45_partial(
            state=state,
            time=time,
            sol=sol,
            dt=sim_dt,
            masses=masses
        )

        for i, t in enumerate(range(0, time, record_dt)):
            out[i] = solution(t)

        del solution

    return out