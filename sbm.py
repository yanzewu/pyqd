#!/usr/bin/env python3
""" Generate spin boson model 
"""

import argparse
import numpy as np

class SBM:

    def __init__(self):
        self.dE = 0     # energy difference
        self.w = None   # frequency list of modes
        self.c = None   # coupling constant of modes

    def load_file(self, filename):
        """ Load a SBM definition file.
        Line number start from 0.
        Line 1: Energy difference (V is 1)
        Line 3: Frequency list
        Line 5: Coupling constant
        """
        
        lines = open(filename, 'r').readlines()
        self.dE = float(lines[1])
        self.w = np.array(list(map(float, lines[3].split())))
        self.c = np.array(list(map(float, lines[5].split())))

    def write_file(self, filename):

        fp = open(filename, 'w')
        fp.write('dE=\n%g\n' % self.dE)
        fp.write('w=\n' + ' '.join((str(w_) for w_ in self.w)) + '\n')
        fp.write('c=\n' + ' '.join((str(c_) for c_ in self.c)) + '\n')
        fp.close()

    def generate_initial_cond(self, T, bias=0.0):
        """ Generate initial position and momentum according to Boltzmann distribution.
        T: Temperature;
        bias: Initial location bias, -1 ~ 1
        Please set 'np.random.seed' to get repeated results.
        """
        #sigma_x = 1.0/np.sqrt(2*self.w*np.tanh(self.w/2/T))
        #sigma_k = np.sqrt(self.w/2/np.tanh(self.w/2/T))
        sigma_x = np.sqrt(T)/self.w
        sigma_k = np.ones(len(self.w)) * np.sqrt(T)

        bias_x = bias * self.c/self.w**2

        self.x0 = np.array([np.random.normal(bias_x[i], sigma_x[i]) for i in range(len(sigma_x))])
        self.k0 = np.array([np.random.normal(0.0, sigma_k[i]) for i in range(len(sigma_k))])

    def generate(self, eta, wc, N=100, spectrum='debye', wmax=6, plot=True):
        """ 
        spectrum = 'debye'/'ohm'
        dE need to be set additionally.
        """

        if spectrum == 'debye':
            # J = \eta * wc*w/(w^2+wc^2)
            # Choose PDF g = 2/\pi* wc/(w^2+wc^2); w > 0
            # CDF = 2/\pi * atan(w/wc); w > 0
            # INV CDF = wc * tan(\pi*x/2)
            
            _generator = lambda sz: wc*np.tan(np.random.uniform(size=sz) * np.pi/2)
            _prob = lambda w: 2*N/np.pi * wc/(w**2 + wc**2)
            _c_coeff = np.sqrt(eta / N)

        elif spectrum == 'ohm':
            # J = \eta * w * exp(-w/wc)
            # Choose PDF g = exp(-w/wc) / wc
            # CDF = 1 - exp(-w/wc)
            # INV CDF = - wc * ln(1 - x)

            _generator = lambda sz: -wc*np.log(1 - np.random.uniform(size=sz))
            _prob = lambda w: N*np.exp(-w/wc) / wc
            _c_coeff = np.sqrt(2*eta*wc/np.pi / N)

        w1 = _generator(N)
        while 1:
            w1 = w1[w1 < wmax*wc]
            if len(w1) == N:
                break
            else:
                w2 = _generator(N - len(w1))
                w1 = np.concatenate((w1, w2))

        self.w = w1
        self.c = _c_coeff * self.w

        if plot:
            try:
                import matplotlib.pyplot as plt

                plt.figure('c')
                plt.plot(self.w, self.c, 'kx-')

                plt.figure('w')
                hist, edges = np.histogram(self.w, bins=50)
                wlist = np.linspace(0, max(self.w), 100)
                plt.plot(edges[1:] - (edges[1]-edges[0])/2, hist, 'kx-')
                plt.plot(wlist, _prob(wlist) * (edges[1]-edges[0]), 'r-')
                
                plt.show()
            except:
                pass


class GeneralSBM:

    def __init__(self):
        pass

    def load_model(self, sbm:SBM):
        c_norm = np.linalg.norm(sbm.c)

        self.H0 = np.array([[sbm.dE/2, 1.0], [1.0, -sbm.dE/2]])
        self.H1 = np.array([[c_norm, 0.0], [0.0, -c_norm]])
        self.C1 = sbm.c/c_norm
        self.C2 = 0.5 * sbm.w**2

        self.x0 = sbm.x0
        self.k0 = sbm.k0

    def load_file(self, filename_prefix):
        """ Load model file, x, k file.
        """

        lines = open(filename_prefix + '-model.txt', 'r').readlines()

        el_dim = len(lines[1].split())

        self.H0 = np.zeros((el_dim, el_dim))
        self.H1 = np.zeros((el_dim, el_dim))

        for i in range(1, el_dim+1):
            linesplit = [w.split(',') for w in lines[i].split()]
            self.H0[i-1] = np.array([float(w[0]) for w in linesplit])
            self.H1[i-1] = np.array([float(w[1][:-1]) if len(w) > 1 else 0.0 for w in linesplit])

        self.C1 = np.array([float(c) for c in lines[el_dim+2].split()])
        self.C2 = np.array([float(c) for c in lines[el_dim+4].split()])

        self.x0 = np.array(list(map(float, open(filename_prefix + '-x.txt', 'r').readlines()[0].split(','))))
        self.k0 = np.array(list(map(float, open(filename_prefix + '-k.txt', 'r').readlines()[0].split(','))))

    def write_file(self, filename_prefix):

        fp = open(filename_prefix + '-model.txt', 'w')
        fp.write('H=\n')

        for i in range(self.H0.shape[0]):
            for j in range(self.H0.shape[1]):
                if self.H1[i,j] != 0.0:
                    fp.write('%.6g,%.6gQ' % (self.H0[i,j], self.H1[i,j]))
                else:
                    fp.write('%.6g' % self.H0[i,j])
                if j != self.H0.shape[1] - 1:
                    fp.write('\t')
            fp.write('\n')
        
        fp.write('C1=\n')
        fp.write('\t'.join((str(c) for c in self.C1)))
        fp.write('\nC2=\n')
        fp.write('\t'.join((str(c) for c in self.C2)))
        fp.write('\n')
        
        fp.close()

        open(filename_prefix + '-x.txt', 'w').write(','.join(list(map(str, self.x0))))
        open(filename_prefix + '-p.txt', 'w').write(','.join(list(map(str, self.k0))))

    def primary_quantize(self, nlevel):
        
        W = np.diag(self.C2)*2  # recover 1/2 factor, since it is included in C2

        normC = np.linalg.norm(self.C1)
        U, R = np.linalg.qr(np.hstack((self.C1[:,None]/normC, np.zeros((len(self.C1), len(self.C1)-1)))))
        U[:,0] /= R[0,0]

        Wp = U.T.dot(W.dot(U))
        D, S = np.linalg.eigh(Wp[1:,1:])
        
        # Quantize the first mode
        omega = np.sqrt(Wp[0,0])
        Hharm = np.diag((0.5 + np.arange(0, nlevel))*omega)
        q_op = np.zeros((nlevel, nlevel))

        assert nlevel > 1
        q_op[:-1,1:] = np.diag(np.sqrt(np.arange(nlevel-1))) / np.sqrt(2*omega)
        q_op += q_op.T

        self.H0 = np.kron(self.H0, np.eye(nlevel)) + np.kron(np.eye(len(self.H0)), Hharm) + np.kron(self.H1*normC, q_op)
        self.H1 = np.kron(np.eye(len(self.H0)), q_op)
        self.C1 = Wp[0,1:].dot(S)
        self.C2 = D*0.5

        self.x0 = S.T.dot(U.T.dot(self.x0)[1:])
        # self.k0 = ? TODO: Fix k0


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generate input of spin-boson model')
    parser.add_argument('-q', '--quantize', type=int, default=0, help='Time of quantization')
    parser.add_argument('-n', '--nlevel', type=int, default=5, help='Number of level')
    parser.add_argument('-T', '--temp', type=float, default=0.2, help='Temperature')
    parser.add_argument('-e', '--eta', type=float, default=1, help='Mass')
    parser.add_argument('-c', '--cutoff', type=float, default=1, help='Cutoff frequency')
    parser.add_argument('-N', '--nmode', type=int, default=100, help='Number of mode')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed')
    parser.add_argument('-j', '--spectrum', type=str, default='debye', help='debye/ohm')
    parser.add_argument('--task', type=str, default='genall', help='Command: gencond/genmodel/genall')
    #parser.add_argument('--autoname', default=False, type=bool, help='Renaming output automatically')
    parser.add_argument('filename', help='IO Filename')

    opt = parser.parse_args()

    np.random.seed(opt.seed)

    if opt.task in ('genall', 'ga', 'genmodel', 'gm'):    # Generate from new data
        sbm = SBM()
        sbm.generate(opt.eta, opt.cutoff, opt.nmode, opt.spectrum)
        sbm.generate_initial_cond(opt.temp)

    elif opt.task in ('gencond', 'gc'):     # Generate from existing model file
        sbm = SBM()
        sbm.load_file(opt.filename)
        sbm.generate_initial_cond(opt.temp)

    if opt.task in ('genmodel', 'gm'):  # Generate the model file only
        sbm.write_file(opt.filename)
        exit(0)


    gsbm = GeneralSBM()

    if opt.task in ('expand', 'ex'):    # Expanding current model
        gsbm.load_file(opt.filename)

        for i in range(opt.quantize):
            gsbm.primary_quantize(opt.nlevel)

    else:
        gsbm.load_model(sbm)

    output_name = opt.filename[:-4] if opt.filename.endswith('.txt') else opt.filename
    gsbm.write_file(output_name)

