
import numpy as np


class Model:

    def __init__(self, kinetic_dim, el_dim, calc_energy, multidim):
        self.kinetic_dim = kinetic_dim      # dimension of kinetical system
        self.el_dim = el_dim                # dimension of electronic system

        self.calc_energy = calc_energy      # calculating energy and drv coupling by self
        # if calc_energy is defined, using interface E(x) -> 1D array and dH(x) = <i|dH|j> -> 2D array
        # otherwise, using V(x) -> 2D array and dV(x) -> 2D array

        self.multidim = multidim            # Supports multidimensional kinetic problem
        # If multidim is False, a adaption will be made in Evaluator


class SACModel(Model):

    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        super().__init__(1, 2, False, False)

    def V(self, x):
        """ Halmitonian in two states
        """

        if x >= 0:
            V11 = self.A * (1 - np.exp(-self.B * x))
        else:
            V11 = - self.A * (1 - np.exp(self.B * x))

        V12 = self.C * np.exp(-self.D * x * x)

        return np.array([[V11, V12], [V12, -V11]])


    def dV(self, x):

        dV11 = self.A * self.B * np.exp(-self.B * abs(x))
        dV12 = - 2.0 * self.C * self.D * x * np.exp(-self.D * x * x)

        return np.array([[dV11, dV12], [dV12, -dV11]])
        

class DACModel(Model):

    def __init__(self, A, B, C, D, E0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E0 = E0

        super().__init__(1, 2, False, False)

    def V(self, x):
        """ Halmitonian in two states
        """

        V12 = self.C * np.exp(-self.D * x * x)

        return np.array([[0, V12], [V12, -self.A*np.exp(-self.B*x*x) + self.E0]])


    def dV(self, x):

        dV22 = 2.0 * self.A * self.B * x * np.exp(-self.B * x*x)
        dV12 = - 2.0 * self.C * self.D * x * np.exp(-self.D * x * x)

        return np.array([[0, dV12], [dV12, dV22]])   
        


class ECRModel(Model):

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

        super().__init__(1, 2, False, False)

    def V(self, x):
        """ Halmitonian in two states
        """
        if x < 0:
            V12 = self.B * np.exp(self.C * x)
        else:
            V12 = self.B * (2 - np.exp(-self.C * x))

        return np.array([[self.A, V12], [V12, -self.A]])


    def dV(self, x):

        dV12 = self.B * self.C * np.exp(-self.C * abs(x))

        return np.array([[0, dV12], [dV12, 0]])


class GenSBModel(Model):
    """ An abstract description of spin boson model, with

    H = H0 + H1*(CX) + C2*X^2
    """

    def __init__(self, filename):
        """ File format:
        Line 1->n: H0+H1, with format (a,bx)
        Line n+2: C1
        Line n+4: C2
        """

        lines = open(filename, 'r').readlines()

        el_dim = len(lines[1].split())

        self.H0 = np.zeros((el_dim, el_dim))
        self.H1 = np.zeros((el_dim, el_dim))

        for i in range(1, el_dim+1):
            linesplit = [w.split(',') for w in lines[i].split()]
            self.H0[i-1] = np.array([float(w[0]) for w in linesplit])
            self.H1[i-1] = np.array([float(w[1][:-1]) if len(w) > 1 else 0.0 for w in linesplit])

        self.C1 = np.array([float(c) for c in lines[el_dim+2].split()])
        self.C2 = np.array([float(c) for c in lines[el_dim+4].split()])

        super().__init__(len(self.C1), el_dim, False, True)

    def V(self, x):

        return self.H0 + self.H1 * self.C1.dot(x) + np.eye(self.el_dim) * (self.C2*x).dot(x)

    def dV(self, x):
        
        return self.H1[:,:,None]*self.C1[None,None,:] + (np.eye(self.el_dim))[:,:,None]*((2*self.C2*x)[None,None,:])

