import numpy as np


class checkPointing:
    def __init__(self, delta_t):
        self.delta_t = delta_t  # time interval at which data is checkpointed
        self.ichecked = []  # space for iteration number
        self.tchecked = []  # space for time
        self.uchecked = []  # space for the numerical solution
        self.time2check = 0.0  # time at which to checkpoint the data

    def add(self, iteration, t, u):
        print("Checkpointing data at t = {:5.2f} iterations.".format(t))
        self.ichecked.append(iteration)
        self.tchecked.append(t)
        self.uchecked.append(np.copy(u))
        self.time2check = self.time2check + self.delta_t
