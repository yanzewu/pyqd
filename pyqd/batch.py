
import copy
import numpy.random as random

from . import evaluator
from . import integrator


class FSSHTask:
    """ FSSH molecular dynamics
    """

    def __init__(self, nstep, box):
        """ nstep: int;
            box: N x 2 array
        """
        self.nstep = nstep
        self.box = box
        self.detect_step = 10

    def load(self, init_state, model, integrator):
        self.state = copy.deepcopy(init_state)
        self.integrator = integrator
        self.evaluator = evaluator.Evaluator(model)

    def run(self):
        self.evaluator.update_potential(self.state)
        self.integrator.initialize(self.state)   # Initialize cache

        for n in range(self.nstep):
            self.integrator.update_first_half(self.state)    # Verlet first half
            self.evaluator.update_potential(self.state)      # Load energy, force and drv coupling
            self.integrator.update_el_state(self.state)      # ES integration
            self.integrator.update_latter_half(self.state)   # Verlet second half
            self.integrator.try_hop(self.state)

            if n % self.detect_step == 0:
                if integrator.outside_box(self.state, box):
                    break

        self.realstep = n

    def is_normal_terminated(self):
        return self.realstep < self.nstep


def run_single(mdtask, seed=0):
    random.seed(seed)
    mdtask.run()
    return [mdtask.state, mdtask.is_normal_terminated()]

