
import sys
import copy
import argparse
import numpy as np
import multiprocessing as mp

from . import batch
from . import integrator
from . import model
from . import state
from . import recorder
from . import evaluator


def plot_md(tasktype, recorder:recorder.Recorder, m_model, box):

    import matplotlib.pyplot as plt

    m_x = recorder.get_data('x')
    m_t = recorder.get_time()
    m_ke = recorder.get_data('ke')
    m_pe = recorder.get_data('pe')

    # if tasktype == 'ehrenfest':
    #     ev = evaluator.Evaluator(m_model)
    #     for state in recorder.snapshots:
    #         ev.to_adiabatic(state)
    ev = evaluator.Evaluator(m_model)
    if tasktype == 'fssh':
        for i, s in enumerate(recorder.snapshots):
            s.rho_el = ev.to_diabatic(s)

    m_rho = recorder.get_data('rho_el')

    print('t\tP0\tP1')
    for i in range(len(m_t)):
        print(m_t[i], m_rho[i,0,0], m_rho[i,1,1])

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
    if tasktype == 'fssh':
        plt.plot(m_t, recorder.get_data('el_state'), 'm-', lw=1, label='El-state')
    plt.legend()

    plt.figure('E-x')

    if m_model.kinetic_dim == 1:
        x = np.linspace(box[0,0], box[0,1], 200)[:,np.newaxis]
        ad_energy, drv_coupling = evaluator.Evaluator(m_model).evaluate(x)
        plt.plot(x, ad_energy[:,0], 'k--', lw=0.5, label='E0')
        plt.plot(x, ad_energy[:,1], 'b--', lw=0.5, label='E1')
    else:
        ad_energy = recorder.get_data('ad_energy')
        plt.plot(m_x[:,0], ad_energy[:,0], 'k--', lw=0.5, label='E0')
        plt.plot(m_x[:,0], ad_energy[:,1], 'b--', lw=0.5, label='E1')

    plt.plot(m_x[:,0], m_pe, 'm-', lw=1, label=tasktype)
    plt.legend()

    if tasktype == 'fssh':

        m_drv_coupling = recorder.get_data('drv_coupling')

        plt.figure('d-x')
        if m_model.kinetic_dim == 1:
            plt.plot(x, drv_coupling[:,0,1,0], 'k--', lw=0.5, label='d12')
        plt.plot(m_x[:,0], m_drv_coupling[:,0,1,0], 'm-', lw=1, label=tasktype)
        plt.legend()


    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSSH')
    parser.add_argument('--batch', default=0, type=int, help='Run batch')
    parser.add_argument('--np', default=1, type=int, help='Process')
    parser.add_argument('--task', default='fssh', type=str, help='Simulation type')
    parser.add_argument('--nstep', default=40000, type=int, help='Maximum step')
    parser.add_argument('--dt', default=0.5, type=float, help='Timestep')
    parser.add_argument('--box', default='-4,4', type=str, help='Simulation box')
    parser.add_argument('--dstep', default=10, type=int, help='Analysis step')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--k0', default='20', type=str, help='k to sample')
    parser.add_argument('--x0', default='-4', type=str, help='Start position of x')
    parser.add_argument('--m', default='2000', type=str, help='Mass')
    parser.add_argument('--model', default='sac', type=str, help='Model')
    parser.add_argument('--args', type=str, help='Additional args to model')
    parser.add_argument('--obj', default='scatter', type=str, help='Object')
    parser.add_argument('--output', default='-', type=str, help='Output name')

    opt = parser.parse_args()

    # For windows compatibility; Use $(< file) on *nix
    if opt.x0.startswith('@'):
        opt.x0 = open(opt.x0[1:], 'r').readlines()[0]
    if opt.k0.startswith('@'):
        opt.k0 = open(opt.k0[1:], 'r').readlines()[0]

    start_x = np.array(list(map(float, opt.x0.split(','))))
    klist = np.array([list(map(float, krow.split(','))) for krow in opt.k0.split(':')])
    m = np.array(list(map(float, opt.m.split(','))))
    box = np.array(list(map(float, opt.box.split(','))))
    print('x0', start_x, file=sys.stderr)
    print('klist', klist, file=sys.stderr)
    print('m', m, file=sys.stderr)
    print('box', box, file=sys.stderr)

    assert len(start_x) == klist.shape[1]
    assert len(m) == len(start_x) or len(m) == 1

    # Creating a cubic box
    if len(box) == 2:
        box = np.array([box]*len(start_x))
    else:
        box = box.reshape([len(start_x), 2])

    # Initialize random seed
    if opt.task == 'fssh':
        if opt.seed == 0:
            from time import time
            opt.seed = int(time())

        print('seed', opt.seed, file=sys.stderr)

    # Tully's 3 Models
    if opt.model == 'sac':
        m_model = model.SACModel(0.01, 1.6, 0.005, 1.0)
    elif opt.model == 'dac':
        m_model = model.DACModel(0.1, 0.28, 0.015, 0.06, 0.05)
    elif opt.model == 'ecr':
        m_model = model.ECRModel(6e-4, 0.1, 0.9)

    # Spin Boson Model
    elif opt.model == 'sbm':
        m_model = model.GenSBModel(opt.args)

    assert m_model.kinetic_dim == len(start_x)

    m_integrator = integrator.Integrator(opt.dt, m) 

    # Scattering task
    if opt.batch != 0 and opt.obj == 'scatter':

        if opt.output == '-':
            fp = sys.stdout
        else:
            fp = open(opt.output + '.txt', 'a')

        title = 'k\tE'
        for i in range(len(box)):
            for j in range(m_model.el_dim):
                title += '\t%dL:%d' % (i+1, j)
            for j in range(m_model.el_dim):
                title += '\t%dR:%d' % (i+1, j)
        for j in range(m_model.el_dim):
            title += '\tN:%d' % j

        print(title, file=fp)
        for start_k in klist:
            init_state = state.State(
                start_x,
                start_k/m,
                state.create_pure_rho_el(m_model.el_dim)
                )

            KE, PE = m_integrator.get_energy_ss(init_state) # WARNING: this could be incorrect for mixed state.

            if opt.task == 'fssh':
                stat_matrix = batch.run_scatter_fssh(
                    init_state,
                    m_model,
                    m_integrator,
                    box,
                    opt.nstep,
                    opt.seed,
                    opt.batch,
                    opt.np
                    )
            elif opt.task == 'ehrenfest':
                stat_matrix = batch.run_scatter_ehrenfest(init_state, m_model, m_integrator, box, opt.nstep, opt.dstep)

            print('%.4g\t%.4g\t%s' % (start_k, KE+PE,
                '\t'.join(('%.4g'%d for d in stat_matrix.flatten()))),
                file=fp)
            fp.flush()

        if fp != sys.stdout:
            fp.close()

    elif opt.batch != 0 and opt.obj == 'population':

        fp = sys.stdout if opt.output == '-' else open(opt.output + '.txt', 'w')
        print('t' + ''.join(('\tP%d' % i for i in range(m_model.el_dim))), file=fp)
        
        init_state = state.State(
            start_x,
            klist[0]/m,
            state.create_pure_rho_el(m_model.el_dim)
            )

        if opt.task == 'fssh':
            t, pop = batch.run_population_fssh(init_state, m_model, m_integrator, box, opt.nstep, opt.dstep, opt.seed, opt.batch, opt.np)
        elif opt.task == 'ehrenfest':
            t, pop = batch.run_population_ehrenfest(init_state, m_model, m_integrator, box, opt.nstep, opt.dstep)

        for t_, p_ in zip(t, pop):
            print('%4f' % t_ + ''.join(('\t%.6f' % p__ for p__ in p_)))

        fp.close()

    else:
        np.random.seed(opt.seed)
        init_state = state.State(start_x, klist[0]/m, state.create_pure_rho_el(m_model.el_dim))
        recorder = recorder.Recorder()

        if opt.task == 'fssh':
            task = batch.FSSHTask(opt.nstep, box, opt.dstep)
        elif opt.task == 'ehrenfest':
            task = batch.EhrenfestTask(opt.nstep, box, opt.dstep)

        task.load(init_state, m_model, m_integrator, recorder)
        task.run()
        
        plot_md(opt.task, recorder, m_model, box)

