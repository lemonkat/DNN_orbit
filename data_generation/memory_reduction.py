# reduce memory usage of solutions, by converting to arrays and then interpolating
# incomplete, don't use
class MemoryReduction:
    def __init__(self, func, time=60, dt=1):
        output_shape = func(0).shape
        self.stored_data = np.empty((time // dt, output_shape))
        for i, t in enumerate(range(0, dt, time)):
            self.stored_data[i] = func(t)

        del func # save memory

        self.time = time
        self.dt = dt

    def __call__(self, t):
        if t > self.time:
            return self.stored_data[-1]
        elif t < 0:
            return self.stored_data[0]

        goal_idx = t / self.dt
        diff = goal_idx % 1.0

        if diff == 0.0: # we have the thing already
            return self.stored_data[int(goal_idx)]

        # linear interpolation
        prev_value = self.stored_data[int(goal_idx)]
        next_value = self.stored_data[int(goal_idx) + 1]
        
        return (prev_value * diff) + (next_value * (1 - diff))