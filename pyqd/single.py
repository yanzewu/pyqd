"""
Run a single session, with fixed x, p
"""

import numpy as np

from . import state
from . import batch
from . import recorder
from . import evaluator
from . import dump


def run_single_session(x0, v0, m_integrator, m_model, box, jobtype, jobobj, seed, output=['population']):

    np.random.seed(seed)
    m_recorder = recorder.Recorder()
    m_evaluator = evaluator.Evaluator(m_model)
    m_state = state.State(x0, v0, state.create_pure_rho_el(m_model.el_dim))
    population = []

    if jobobj == 'scatter':
        
        if jobtype == 'fssh':
            task = batch.FSSHTask(opt.nstep, box, opt.dstep)
            task.load(init_state, m_model, m_integrator, m_recorder)
            task.run()
            population = np.zeros((len(m_recorder.snapshots), m_model.el_dim))
            for i, s in enumerate(m_recorder.snapshots):
                population[i, s.el_state] = 1
            
        elif jobtype == 'ehrenfest':
            m_state.rho_el = m_evaluator.to_diabatic(m_state.rho_el, m_state.x)
            task = batch.EhrenfestTask(opt.nstep, box, opt.dstep)
            task.load(init_state, m_model, m_integrator, m_recorder)
            task.run()
            for s in m_recorder.snapshots:
                s.rho_el = m_evaluator.to_adiabatic(s.rho_el, s.x)
            population = np.diagonal(m_recorder.get_data('rho_el'), 0, 1, 2).real


    elif jobobj == 'population':

        if jobtype == 'fssh':
            m_evaluator.sample_adiabatic_states(init_state)
            task = batch.FSSHTask(opt.nstep, None, opt.dstep)
            task.load(init_state, m_model, m_integrator, m_recorder)
            task.run()
            for s in m_recorder.snapshots:
                population.append(m_evaluator.recover_diabatic_state())
            population = np.array(population)
            
        elif jobtype == 'ehrenfest':
            task = batch.EhrenfestTask(opt.nstep, None, opt.dstep)
            task.load(init_state, m_model, m_integrator, m_recorder)
            task.run()
            population = np.diagonal(m_recorder.get_data('rho_el'), 0, 1, 2).real
    
    dumper = dump.Dumper('-')
    
    for d in output:

        if d == 'population':
            dumper.write_data_with_time(population, m_recorder.get_time(), title_prefix='P')
        else:
            dumper.write_data(m_recorder.get_data(d), title_prefix=d)

    plot_md(jobtype, m_recorder, m_model, box)


def plot_md(jobtype, m_recorder:recorder.Recorder, m_model):

    import matplotlib.pyplot as plt

    m_x = m_recorder.get_data('x')
    m_t = m_recorder.get_time()
    m_ke = m_recorder.get_data('ke')
    m_pe = m_recorder.get_data('pe')
    m_rho = m_recorder.get_data('rho_el')

    plt.figure('E-t')
    plt.plot(m_t, m_ke+m_pe, 'k-', lw=1, label='Energy')
    plt.plot(m_t, m_pe, 'm-', lw=1, label='PE')
    plt.legend()

    plt.figure('x-t')
    for j in range(min(m_x.shape[1], 4)):
        plt.plot(m_t, m_x[:,j], lw=0.8, label='traj%d'%j)
    plt.legend()

    plt.figure('state-t')
    plt.plot(m_t, m_rho[:,0,0], lw=0.8, label='P0')
    plt.plot(m_t, m_rho[:,1,1], lw=0.8, label='P1')
    plt.legend()

    plt.figure('E-x')

    if m_model.kinetic_dim == 1:
        x = np.linspace(min(m_x[:,0]), max(m_x[:,0]), 200)[:,np.newaxis]
        ad_energy, drv_coupling = evaluator.Evaluator(m_model).evaluate(x)
        plt.plot(x, ad_energy[:,0], 'k--', lw=0.5, label='E0')
        plt.plot(x, ad_energy[:,1], 'b--', lw=0.5, label='E1')
    else:
        ad_energy = m_recorder.get_data('ad_energy')
        plt.plot(m_x[:,0], ad_energy[:,0], 'k--', lw=0.5, label='E0')
        plt.plot(m_x[:,0], ad_energy[:,1], 'b--', lw=0.5, label='E1')

    plt.plot(m_x[:,0], m_pe, 'm-', lw=1, label=jobtype)
    plt.legend()

    if jobtype == 'fssh':

        m_drv_coupling = m_recorder.get_data('drv_coupling')

        plt.figure('d-x')
        if m_model.kinetic_dim == 1:
            plt.plot(x, drv_coupling[:,0,1,0], 'k--', lw=0.5, label='d12')
        plt.plot(m_x[:,0], m_drv_coupling[:,0,1,0], 'm-', lw=1, label=jobtype)
        plt.legend()


    plt.show()