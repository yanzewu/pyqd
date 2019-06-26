
import argparse
import numpy as np

from . import batch
from . import integrator
from . import model
from . import state

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
    stat_matrix = np.zeros((box.shape[0]*2+1, model.dim()))
    # ROW: outside wall, COL: el_state

    for r in result:
        w = integrator.outside_which_wall(r[0], box)
        e = el_state_stat[r[0].el_state]
        stat_matrix[w, e] += 1

    return stat_matrix


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSSH')
    parser.add_argument('--batch', default=0, type=int, help='Run batch')
    parser.add_argument('--np', default=1, type=int, help='Process')
    parser.add_argument('--nstep', default=40000, type=int, help='Maximum step')
    parser.add_argument('--dt', default=0.5, type=float, help='Timestep')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--k0', default='20', type=str, help='k to sample')
    parser.add_argument('--x0', default='-4', type=str, help='Start position of x')
    parser.add_argument('--m', default='2000', type=str, help='Mass')
    parser.add_argument('--model', default='sac', type=str, help='Model')
    parser.add_argument('--output', default='a', type=str, help='Output name')

    opt = parser.parse_args()

    start_x = np.array(list(map(float, opt.x0.split(','))))
    start_k = np.array(list(map(float, opt.k0.split(','))))
    m = np.array(list(map(float, opt.m.split(','))))
    print(start_x)
    print(start_k)
    print(m)

    if opt.seed == 0:
        from time import time
        opt.seed = int(time())

    if opt.model == 'sac':
        model = model.SACModel(0.01, 1.6, 0.005, 1.0)
    elif opt.model == 'dac':
        model = model.DACModel(0.1, 0.28, 0.015, 0.06, 0.05)
    elif opt.model == 'ecr':
        model = model.ECRModel(6e-4, 0.1, 0.9)

    integrator = integrator.Integrator(opt.dt) 

    if opt.batch != 0:    
        print(klist, file=sys.stderr)
        tl = []
        rl = []
        th = []
        rh = []

        fp = open(opt.output + '.txt', 'a')

        title = 'k\tE'
        for i in range(len(box)):
            for j in range(model.dim()):
                title += '\t%dL:%d\t%dR:%d' % (i+1, j, i+1, j)
        for j in range(model.dim()):
            title += '\tI:%d' % j

        print(title, file=fp)
        for k in klist:
            init_state = state.State(
                start_x,
                start_k/m,
                state.create_pure_rho_el(model.dim())
                )

            energy = integrator.get_energy(init_state)
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

            print('%.4g\t%.4g\t%s' % (k, energy,
                '\t'.join(('%.4g'%(d/opt.batch) for d in stat_matrix.flatten()))
                ))
            fp.flush()

        fp.close()
