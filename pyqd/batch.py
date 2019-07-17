""" Concurrent tasks
"""

import copy
import numpy as np

from . import task
from . import state
from . import evaluator
from . import recorder
from . import integrator


def run_scatter_fssh(x0list, v0list, m_model, m_integrator, box, nstep, record_step, seed, nbatch, nproc):
    """ Run a scattering task with FSSH. State is initiated on adiabatic surface.
    """

    Etot = []
    stat_matrices = []

    for x0, v0 in zip(x0list, v0list):
            
        m_state = state.State(x0, v0, state.create_pure_rho_el(m_model.el_dim))
        mdtask = task.FSSHTask(nstep, box, record_step)
        mdtask.load(m_state, m_model, m_integrator)
        KE, PE = m_integrator.get_energy_ss(m_state)
        Etot.append(KE+PE)

        tasks = [copy.deepcopy(mdtask) for i in range(nbatch)]
        results = run_batch(tasks, seed, nproc)

        stat_matrix = np.zeros((box.shape[0]*2+1, m_model.el_dim))
        # ROW: outside wall, COL: el_state

        for r in results:
            w = integrator.outside_which_wall(r[0], box)
            e = r[0].el_state
            stat_matrix[w, e] += 1

        stat_matrices.append(stat_matrix/nbatch)

    return Etot, stat_matrices


def run_scatter_ehrenfest(x0list, v0list, m_model, m_integrator, box, nstep, record_step):
    """ Run a scattering task with Ehrenfest dynamics. State is initiated on adiabatic surface.
    """
    
    m_evaluator = evaluator.Evaluator(m_model)

    Etot = []
    stat_matrices = []

    for x0, v0 in zip(x0list, v0list):

        m_state = state.State(x0, v0, state.create_pure_rho_el(m_model.el_dim))
        m_state.rho_el = m_evaluator.to_diabatic(m_state.rho_el, m_state.x) # to diabatic basis

        mdtask = task.EhrenfestTask(nstep, box, record_step)
        mdtask.load(m_state, m_model, m_integrator)

        final_state = run_single(mdtask)[0]
        KE, PE = m_integrator.get_energy_mf(final_state)
        Etot.append(KE + PE)

        stat_matrix = np.zeros((box.shape[0]*2+1, m_model.el_dim))
        # ROW: outside wall, COL: el_state
        
        final_state.rho_el = m_evaluator.to_adiabatic(final_state.rho_el, final_state.x)
        w = integrator.outside_which_wall(final_state, box)
        stat_matrix[w, :] = np.diag(final_state.rho_el.real)
        stat_matrices.append(stat_matrix)

    return Etot, stat_matrices


def run_population_fssh(x0list, v0list, m_model, m_integrator, nstep, record_step, seed, nproc):
    """ Run a population task with FSSH. State is initiated on diabatic surface.
    Returns time and population evolution.
    """

    m_evaluator = evaluator.Evaluator(m_model)
    tasks = []

    for x0, v0 in zip(x0list, v0list):

        m_state = state.State(x0, v0, state.create_pure_rho_el(m_model.el_dim))
        m_evaluator.sample_adiabatic_states(m_state)    # convert to adiabatic basis

        mdtask = task.FSSHTask(nstep, None, record_step)
        mdtask.load(m_state, m_model, m_integrator, recorder.Recorder())

        tasks.append(mdtask)

    results = run_batch(tasks, seed, nproc)

    t = results[0][1].get_time()
    sumpop = np.zeros((len(t), m_model.el_dim))

    for r in results:
        for i, s in enumerate(r[1].snapshots):
            sumpop[i] += m_evaluator.recover_diabatic_state(s)
            
    return t, sumpop / len(results)


def run_population_ehrenfest(x0list, v0list, m_model, m_integrator, nstep, recorder_step, seed, nproc):
    """ Run a scattering task with Ehrenfest dynamics. State is initiated on diabatic surface.
    Returns time and population evolution.
    """

    m_evaluator = evaluator.Evaluator(m_model)
    tasks = []

    for x0, v0 in zip(x0list, v0list):

        m_state = state.State(x0, v0, state.create_pure_rho_el(m_model.el_dim))

        mdtask = task.EhrenfestTask(nstep, None, recorder_step)
        mdtask.load(m_state, m_model, m_integrator, recorder.Recorder())
        tasks.append(mdtask)

    results = run_batch(tasks, seed, nproc)

    t = results[0][1].get_time()
    sumpop = np.zeros((len(t), m_model.el_dim))

    for r in results:
        sumpop += np.diagonal(mdtask.recorder.get_data('rho_el'), 0, 1, 2).real
            
    return t, sumpop / len(results)


def run_single(mdtask:task.MDTask, seed=0):
    """ Run a single md task.
    """
    np.random.seed(seed)
    mdtask.run()
    return [mdtask.state, mdtask.recorder]


def run_batch(tasks, seed, nproc):
    if nproc > 1:
        pool = mp.Pool(nproc)
        ret = []
        for i, mdtask in enumerate(tasks):
            ret.append(pool.apply_async(run_single, args=[m_mdtask, seed+i], error_callback=err_callback))
            
        pool.close()
        pool.join()
        results = [r.get() for r in ret]

    else:
        results = []
        for i, mdtask in enumerate(tasks):
            results.append(run_single(mdtask, seed+i))

    return results


def err_callback(e):
    print(e)
