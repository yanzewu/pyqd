
import sys
import argparse
import numpy as np

from . import batch
from . import integrator
from . import model
from . import state
from . import single
from . import dump

def extend_array(arr, length):

    assert arr.shape[0] == 1 or arr.shape[0] == length
    if arr.shape[0] == length:
        return arr
    else:
        return np.array((arr[0] for i in range(length)))


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
    parser.add_argument('--m', default='2000', type=float, help='Mass')
    parser.add_argument('--model', default='sac', type=str, help='Model')
    parser.add_argument('--args', type=str, help='Additional args to model')
    parser.add_argument('--dump', default='population', type=str, help='Additional data to dump')
    parser.add_argument('--obj', default='scatter', type=str, help='Object')
    parser.add_argument('--output', default='-', type=str, help='Output name')

    opt = parser.parse_args()

    if opt.x0.startswith('@'):
        opt.x0 = ':'.join(open(opt.x0[1:], 'r').readlines())
    if opt.k0.startswith('@'):
        opt.k0 = ':'.join(open(opt.k0[1:], 'r').readlines())
    
    # Set up random seed
    if opt.seed == 0:
        from time import time
        opt.seed = int(time())

    x0list = np.array([list(map(float, row.split(','))) for row in opt.x0.split(':')])
    k0list = np.array([list(map(float, row.split(','))) for row in opt.k0.split(':')])
    box = np.array([list(map(float, row.split(','))) for row in opt.box.split(':')])

    print('x0', x0, file=sys.stderr)
    print('klist', klist, file=sys.stderr)
    print('box', box, file=sys.stderr)
    print('seed', opt.seed, file=sys.stderr)

    x0list = extend_array(x0list, k0list.shape[0])
    box = extend_array(box, k0list.shape[1])
    assert x0.shape[1] == klist.shape[1]

    modellist = {
        'sac':model.SACModel,
        'dac':model.DACModel,
        'ecr':model.ECRModel,
        'sbm':model.GenSBModel
    }
    m_model = modellist[opt.model](*opt.args.split(','))    # TODO: try parsing
    assert m_model.kinetic_dim == len(x0)

    m_integrator = integrator.Integrator(opt.dt, opt.m)
    v0list = k0list / opt.m

    if opt.batch == 0:
        single.run_single_session(x0list[0], v0list[0], m_integrator, m_model, box, opt.task, 
            opt.obj, opt.seed, opt.dump.split(','))

    # Scattering task
    elif opt.obj == 'scatter':
    
        if opt.task == 'fssh':
            Etot, stat_matrices = batch.run_scatter_fssh(x0list, v0list, m_model, m_integrator,
                box, opt.nstep, opt.dstep, opt.seed, opt.batch, opt.np)
        elif opt.task == 'ehrenfest':
            Etot, stat_matrices = batch.run_scatter_ehrenfest(x0list, v0list, m_model, m_integrator, 
                box, opt.nstep, opt.dstep)

        dump.write_scatter_result(opt.output, klist, Etot, stat_matrices)
        
    # Population task
    elif opt.obj == 'population':

        if opt.task == 'fssh':
            t, pop = batch.run_population_fssh(x0list, v0list, m_model, m_integrator, 
                opt.nstep, opt.dstep, opt.seed, opt.np)
        elif opt.task == 'ehrenfest':
            t, pop = batch.run_population_ehrenfest(x0list, v0list, m_model, m_integrator, 
                opt.nstep, opt.dstep, opt.seed, opt.np)
        dump.write_population_result(opt.output, t, pop)

    else:
        print('Invalid option: %s' % opt.obj, fp=sys.stderr)
