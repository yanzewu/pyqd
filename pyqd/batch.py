
import copy
import numpy as np
import numpy.random as random
import multiprocessing as mp

from . import evaluator
from . import integrator
from . import recorder

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
            self.recorder.collect(self.state, self.integrator.dt * n)
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
    return [mdtask.state, mdtask.is_normal_terminated(), mdtask.recorder]


def run_scatter_fssh(m_state, m_model, m_integrator, box, nstep, seed, nbatch, nproc):

    state_stat = []
        
    mdtask = FSSHTask(nstep, box)
    mdtask.load(m_state, m_model, m_integrator)

    if nproc > 1:

        pool = mp.Pool(nproc)
        ret = []
        for i in range(nbatch):
            ret.append(pool.apply_async(run_single, args=[copy.deepcopy(mdtask), seed+i], error_callback=err_callback))
            
        pool.close()
        pool.join()

        result = [r.get() for r in ret]

    else:
        result = [run_single(copy.deepcopy(mdtask), seed+i) for i in range(nbatch)]

    # statistics: wall (%), state (%)
    stat_matrix = np.zeros((box.shape[0]*2+1, m_model.el_dim))
    # ROW: outside wall, COL: el_state

    for r in result:
        w = integrator.outside_which_wall(r[0], box)
        e = r[0].el_state
        stat_matrix[w, e] += 1

    return stat_matrix/nbatch


def run_population_fssh(m_state, m_model, m_integrator, box, nstep, record_step, seed, nbatch, nproc):

    mdtask = FSSHTask(nstep, box, record_step)
    mdtask.load(m_state, m_model, m_integrator, recorder.Recorder())

    evtmp = evaluator.Evaluator(m_model)
    mdtask.state.rho_el = evtmp.to_adiabatic(mdtask.state.rho_el, mdtask.state.x)

    if nproc > 1:

        pool = mp.Pool(nproc)
        ret = []
        for i in range(nbatch):
            ret.append(pool.apply_async(run_single, args=[copy.deepcopy(mdtask), seed+i], error_callback=err_callback))
            
        pool.close()
        pool.join()

        result = [r.get() for r in ret]

    else:
        result = [run_single(copy.deepcopy(mdtask), seed+i) for i in range(nbatch)]

    t = result[0][2].get_time()
    sumpop = np.zeros((len(t), m_model.el_dim))

    from .state import create_pure_rho_el



    for r in result:
        for i, s in enumerate(r[2].snapshots):
            sumpop[i] += np.diag(evtmp.to_diabatic(s.el_state, s.x)).real
            s.rho_el = evtmp.to_diabatic(s.rho_el, s.x)
            
    return t, sumpop / nbatch


def run_scatter_ehrenfest(m_state, m_model, m_integrator, box, nstep, analyze_step):

    mdtask = EhrenfestTask(nstep, box, analyze_step)
    mdtask.load(m_state, m_model, m_integrator)
    result = run_single(mdtask)
    
    # statistics: wall (%), state (%)
    stat_matrix = np.zeros((box.shape[0]*2+1, m_model.el_dim))
    # ROW: outside wall, COL: el_state
     
    rho_ad = evaluator.Evaluator(m_model).to_adiabatic(result[0].rho_el, result[0].x)
    w = integrator.outside_which_wall(result[0], box)
    stat_matrix[w, :] = np.diag(result[0].rho_el.real)

    return stat_matrix


def run_population_ehrenfest(m_state, m_model, m_integrator, box, nstep, recorder_step):

    mdtask = EhrenfestTask(nstep, box, recorder_step)
    mdtask.load(m_state, m_model, m_integrator, recorder.Recorder())
    run_single(mdtask)
    
    return mdtask.recorder.get_time(), np.diagonal(mdtask.recorder.get_data('rho_el'), 0, 1, 2).real


def err_callback(e):
    print(e)

