
import sys
import argparse
import numpy as np

from . import batch
from . import integrator
from . import model
from . import state
from . import recorder
from . import evaluator


def err_callback(e):
    print(e)


def run_batch(state, model, integrator, box, nstep, seed, nbatch, nproc):

    state_stat = []
    mdtask = batch.FSSHTask(nstep, box)
    mdtask.load(state, model, integrator)

    if nproc > 1:

        pool = mp.Pool(nproc)
        ret = []
        for i in range(nbatch):
            ret.append(pool.apply_async(run_single, args=[copy.deepcopy(mdtask), seed+i], error_callback=err_callback))
            
        pool.close()
        pool.join()

        result = [r.get() for r in ret]

    else:
        for i in range(nbatch):
            result.append(run_single(copy.deepcopy(mdtask), seed+i))
    
    # statistics: wall (%), state (%)
    stat_matrix = np.zeros((box.shape[0]*2+1, model.el_dim))
    # ROW: outside wall, COL: el_state

    for r in result:
        w = integrator.outside_which_wall(r[0], box)
        e = el_state_stat[r[0].el_state]
        stat_matrix[w, e] += 1

    return stat_matrix


def plot_md(recorder:recorder.Recorder, model, box):

    import matplotlib.pyplot as plt

    m_x = recorder.get_data('x')
    m_ke = recorder.get_data('ke')
    m_pe = recorder.get_data('pe')
    m_drv_coupling = recorder.get_data('drv_coupling')

    x = np.linspace(box[0,0], box[0,1], 200)
    ad_energy, drv_coupling = evaluator.Evaluator(model).evaluate(x)

    plt.figure('E-x')
    plt.plot(x, ad_energy[:,0], 'k--', lw=0.5, label='E0')
    plt.plot(x, ad_energy[:,1], 'b--', lw=0.5, label='E1')
    plt.plot(m_x, m_pe, 'm-', lw=1, label='simulation')
    plt.legend()

    plt.figure('d-x')
    plt.plot(x, drv_coupling[:,0,1,0], 'k--', lw=0.5, label='d12')
    plt.plot(m_x, m_drv_coupling[:,0,1,0], 'm-', lw=1, label='simulation')
    plt.legend()

    plt.figure('E-t')
    plt.plot(recorder.get_time(), m_ke+m_pe, 'k-', lw=1, label='Energy')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSSH')
    parser.add_argument('--batch', default=0, type=int, help='Run batch')
    parser.add_argument('--np', default=1, type=int, help='Process')
    parser.add_argument('--nstep', default=40000, type=int, help='Maximum step')
    parser.add_argument('--dt', default=0.5, type=float, help='Timestep')
    parser.add_argument('--box', default='-4,4', type=str, help='Simulation box')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--k0', default='20', type=str, help='k to sample')
    parser.add_argument('--x0', default='-4', type=str, help='Start position of x')
    parser.add_argument('--m', default='2000', type=str, help='Mass')
    parser.add_argument('--model', default='sac', type=str, help='Model')
    parser.add_argument('--output', default='a', type=str, help='Output name')

    opt = parser.parse_args()

    start_x = np.array(list(map(float, opt.x0.split(','))))
    klist = np.array([list(map(float, krow.split(','))) for krow in opt.k0.split(':')])
    m = np.array(list(map(float, opt.m.split(','))))
    box = np.array(list(map(float, opt.box.split(',')))).reshape([len(start_x), 2])
    print('x0', start_x, file=sys.stderr)
    print('klist', klist, file=sys.stderr)
    print('m', m, file=sys.stderr)
    print('box', box, file=sys.stderr)

    assert len(start_x) == klist.shape[1]
    assert len(m) == len(start_x) or len(m) == 1


    if opt.seed == 0:
        from time import time
        opt.seed = int(time())

    print('seed', opt.seed, file=sys.stderr)

    # Tully's 3 Models
    if opt.model == 'sac':
        model = model.SACModel(0.01, 1.6, 0.005, 1.0)
    elif opt.model == 'dac':
        model = model.DACModel(0.1, 0.28, 0.015, 0.06, 0.05)
    elif opt.model == 'ecr':
        model = model.ECRModel(6e-4, 0.1, 0.9)

    assert model.kinetic_dim == len(start_x)

    integrator = integrator.Integrator(opt.dt, m) 

    if opt.batch != 0:

        fp = open(opt.output + '.txt', 'a')

        title = 'k\tE'
        for i in range(len(box)):
            for j in range(model.el_dim):
                title += '\t%dL:%d\t%dR:%d' % (i+1, j, i+1, j)
        for j in range(model.el_dim):
            title += '\tI:%d' % j

        print(title, file=fp)
        for start_k in klist:
            init_state = state.State(
                start_x,
                start_k/m,
                state.create_pure_rho_el(model.el_dim)
                )

            KE, PE = integrator.get_energy(init_state)
            stat_matrix = run_batch(
                state,
                model,
                integrator,
                box,
                opt.nstep,
                opt.seed,
                opt.batch,
                opt.np
                )

            print('%.4g\t%.4g\t%s' % (k, KE+PE,
                '\t'.join(('%.4g'%(d/opt.batch) for d in stat_matrix.flatten()))
                ))
            fp.flush()

        fp.close()

    else:
        np.random.seed(opt.seed)
        init_state = state.State(start_x, klist[0]/m, state.create_pure_rho_el(model.el_dim))
        recorder = recorder.Recorder()
        task = batch.FSSHTask(opt.nstep, box)
        task.load(init_state, model, integrator, recorder)
        task.run()
        
        plot_md(recorder, model, box)

