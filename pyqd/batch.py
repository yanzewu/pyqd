
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

    def load(self, init_state, model, integrator, recorder=None):
        self.state = copy.deepcopy(init_state)
        self.integrator = integrator
        self.evaluator = evaluator.Evaluator(model)
        self.recorder = recorder

    def run(self):
        self.evaluator.update_potential_ss(self.state)
        self.integrator.initialize(self.state)   # Initialize cache

        if self.recorder:
            self.recorder.collect(self.state, 0.0)
            self.recorder.collect_energy(*self.integrator.get_energy_ss(self.state))

        for n in range(self.nstep):
            self.integrator.update_first_half(self.state)    # Verlet first half
            self.evaluator.update_potential_ss(self.state)      # Load energy, force and drv coupling
            self.integrator.update_el_state_sh(self.state)      # ES integration
            self.integrator.update_latter_half(self.state)   # Verlet second half
            self.integrator.try_hop(self.state)

            if n % self.detect_step == 0:
                if integrator.outside_box(self.state, self.box):
                    break
                if self.recorder:
                    self.recorder.collect(self.state, self.integrator.dt * n)
                    self.recorder.collect_energy(*self.integrator.get_energy_ss(self.state))

        self.realstep = n

    def is_normal_terminated(self):
        return self.realstep < self.nstep


class EhrenfestTask:
    """ Ehrenfest dynamics
    """

    def __init__(self, nstep, box):
        """ nstep: int;
            box: N x 2 array
        """
        self.nstep = nstep
        self.box = box
        self.detect_step = 10

    def load(self, init_state, model, integrator, recorder=None):
        self.state = copy.deepcopy(init_state)
        self.integrator = integrator
        self.evaluator = evaluator.Evaluator(model)
        self.recorder = recorder

    def run(self):
        self.evaluator.update_potential_ms(self.state)
        self.integrator.initialize(self.state, 'mf')   # Initialize cache

        if self.recorder:
            self.recorder.collect(self.state, 0.0)
            self.recorder.collect_energy(*self.integrator.get_energy_mf(self.state))

        for n in range(self.nstep):
            self.integrator.update_first_half(self.state)    # Verlet first half
            self.evaluator.update_potential_ms(self.state)      # Load energy, force and drv coupling
            self.integrator.update_el_state_mf(self.state)      # ES integration
            self.integrator.update_latter_half(self.state)   # Verlet second half

            if n % self.detect_step == 0:
                if integrator.outside_box(self.state, self.box):
                    break
                if self.recorder:
                    self.recorder.collect(self.state, self.integrator.dt * n)
                    self.recorder.collect_energy(*self.integrator.get_energy_mf(self.state))

        self.realstep = n

    def is_normal_terminated(self):
        return self.realstep < self.nstep    


def run_single(mdtask, seed=0):
    random.seed(seed)
    mdtask.run()
    return [mdtask.state, mdtask.is_normal_terminated()]

