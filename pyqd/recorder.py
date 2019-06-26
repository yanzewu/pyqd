
import copy
import numpy as np


class Recorder:

    def __init__(self):
        self.snapshots = []
        self.timestamps = []
        self.KEs = []
        self.PEs = []

    def collect(self, state, time):
        self.snapshots.append(copy.deepcopy(state))
        self.timestamps.append(time)

    def collect_energy(self, PE, KE):
        self.PEs.append(PE)
        self.KEs.append(KE)

    def get_data(self, name):
        
        if name.upper() == 'KE':
            return np.array(self.KEs)
        elif name.upper() == 'PE':
            return np.array(self.PEs)

        return np.array([getattr(s, name) for s in self.snapshots])

    def get_time(self):
        return np.array(self.timestamps)

        