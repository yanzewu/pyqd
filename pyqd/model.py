
from numpy import *


class SACModel:

    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def dim(self):
        return 2

    def V(self, x):
        """ Halmitonian in two states
        """

        if x >= 0:
            V11 = self.A * (1 - exp(-self.B * x))
        else:
            V11 = - self.A * (1 - exp(self.B * x))

        V12 = self.C * exp(-self.D * x * x)

        return array([[V11, V12], [V12, -V11]])


    def dV(self, x):

        dV11 = self.A * self.B * exp(-self.B * abs(x))
        dV12 = - 2.0 * self.C * self.D * x * exp(-self.D * x * x)

        return array([[dV11, dV12], [dV12, -dV11]])
        

class DACModel:
    def __init__(self, A, B, C, D, E0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E0 = E0

    def dim(self):
        return 2

    def V(self, x):
        """ Halmitonian in two states
        """

        V12 = self.C * exp(-self.D * x * x)

        return array([[0, V12], [V12, -self.A*exp(-self.B*x*x) + self.E0]])


    def dV(self, x):

        dV22 = 2.0 * self.A * self.B * x * exp(-self.B * x*x)
        dV12 = - 2.0 * self.C * self.D * x * exp(-self.D * x * x)

        return array([[0, dV12], [dV12, dV22]])   
        


class ECRModel:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def dim(self):
        return 2

    def V(self, x):
        """ Halmitonian in two states
        """
        if x < 0:
            V12 = self.B * exp(self.C * x)
        else:
            V12 = self.B * (2 - exp(-self.C * x))

        return array([[self.A, V12], [V12, -self.A]])


    def dV(self, x):

        dV12 = self.B * self.C * exp(-self.C * abs(x))

        return array([[0, dV12], [dV12, 0]])

