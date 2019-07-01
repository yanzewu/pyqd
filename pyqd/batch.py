
import copy
import numpy.random as random

from . import evaluator
from . import integrator

class MDTask:

    def __init__(self, nstep, box, analyze_step=10):
        self.nstep = nstep
        self.box = box
        self.detect_step = 10
        self.analyze_step = analyze_step
        self.realstep = 0

    def load(self, init_state, model, integrator, recorder=None):
        self.state = copy.deepcopy(init_state)
        self.integrator = integrator
        self.evaluator = evaluator.Evaluator(model)
        self.recorder = recorder

    def is_normal_terminated(self):
        return self.realstep < self.nstep


class FSSHTask(MDTask):
    """ FSSH molecular dynamics
    """

    def __init__(self, nstep, box, analyze_step=10):
        """ nstep: int;
            box: N x 2 array
        """
        super().__init__(nstep, box, analyze_step)

    def run(self):
        self.evaluator.update_potential_ss(self.state)
        self.integrator.initialize(self.state)   # Initialize cache

        self.analyze(0)

        for n in range(self.nstep):
            self.integrator.update_first_half(self.state)    # Verlet first half
            self.evaluator.update_potential_ss(self.state)      # Load energy, force and drv coupling
            self.integrator.update_el_state_sh(self.state)      # ES integration
            self.integrator.update_latter_half(self.state)   # Verlet second half
            self.integrator.try_hop(self.state)

            if (n+1) % self.detect_step == 0:
                if integrator.outside_box(self.state, self.box):
                    break
            if (n+1) % self.analyze_step == 0:
                self.analyze(n+1)

        self.realstep = n+1

    def analyze(self, n):
        if self.recorder:
            self.recorder.collect(self.state, self.integrator.dt * (n+1))
            self.recorder.collect_energy(*self.integrator.get_energy_ss(self.state))


class EhrenfestTask(MDTask):
    """ Ehrenfest dynamics
    """

    def __init__(self, nstep, box, analyze_step=10):
        """ nstep: int;
            box: N x 2 array
        """
        super().__init__(nstep, box, analyze_step)

    def run(self):
        self.evaluator.update_potential_ms(self.state)
        self.integrator.initialize(self.state, 'mf')   # Initialize cache

        print('t\tPE\tEtot')
        self.analyze(0)

        for n in range(self.nstep):
            self.integrator.update_first_half(self.state)    # Verlet first half
            self.evaluator.update_potential_ms(self.state)      # Load energy, force and drv coupling
            self.integrator.update_el_state_mf(self.state)      # ES integration
            self.integrator.update_latter_half(self.state)   # Verlet second half

            if (n+1) % self.detect_step == 0:
                if integrator.outside_box(self.state, self.box):
                    break
            if (n+1) % self.analyze_step == 0:
                self.analyze(n+1)

        self.realstep = n+1

    def analyze(self, n):
        PE, KE = self.integrator.get_energy_mf(self.state)
        print('%g\t%4g\t%4g' % (self.integrator.dt * n, PE, KE+PE))
        if self.recorder:
            self.recorder.collect(self.state, self.integrator.dt * n)
            self.recorder.collect_energy(PE, KE)        


def run_single(mdtask, seed=0):
    random.seed(seed)
    mdtask.run()
    return [mdtask.state, mdtask.is_normal_terminated()]

