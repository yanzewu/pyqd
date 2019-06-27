
import numpy as np

from . import evaluator


class State:

    def __init__(self, x, v, rho_el, el_state=0):
        self.x = x
        self.v = v
        self.rho_el = rho_el
        self.el_state = el_state

        self.ad_energy = np.zeros_like(x)
        self.hopping_prob = np.zeros_like(x)

        self.force = np.zeros_like(self.x)
        self.H_el = np.zeros_like(rho_el)
        self.drv_coupling = np.zeros(
            (self.rho_el.shape[0], self.rho_el.shape[1], len(self.x))
            )


def create_pure_rho_el(dim, el_state=0):
    rho_el = np.zeros((dim, dim))
    rho_el[el_state, el_state] = 1.0
    return rho_el

