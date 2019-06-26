import numpy as np
from scipy.integrate import complex_ode

from . import state


class Integrator:

    def __init__(self, dt, m, using_ode=True):
        self.dt = dt
        self.m = m
        self.m_inv = 1.0/m
        self.using_ode = using_ode

    def initialize(self, state:state.State):
        self.a = 0.0
        self.d_new = state.drv_coupling
        self.E_new = state.ad_energy

    def update_first_half(self, state:state.State):
        state.v += 0.5 * self.a * self.dt
        state.x += state.v * self.dt
    
    def update_latter_half(self, state:state.State):
        
        self.a = self.m_inv * state.force
        state.v += 0.5 * self.a * self.dt

    def update_el_state(self, state:state.State):

        self.d_old = self.d_new
        self.d_new = state.drv_coupling
        self.E_old = self.E_new
        self.E_new = state.ad_energy

        dv_ave = 0.5*(self.dv_new.dot(state.v) + self.dv_old.dot(state.v))
        E_ave = 0.5*(self.E_new + self.E_old)

        H = np.diag(E_ave) - 1j * dv_ave

        if self.using_ode:

            def liouville(t, rho):
                rho_ = rho.reshape(H.shape)
                return (-1j*(H.dot(rho_) - rho_.dot(H))).flatten()

            r = complex_ode(liouville)
            r.set_initial_value(state.rho_el.flatten())
            state.rho_el = r.integrate(self.dt).reshape(H.shape)

        else:   # Verlet integration
            drho = -1j*(H.dot(state.rho_el) - state.rho_el.dot(H))
            rho_el_old_tmp = state.rho_el
            state.rho_el = 2*state.rho_el - self.rho_el_old + drho*dt**2
            self.rho_el_old = rho_el_old_tmp

        self.update_hopping_prob(state, dv_ave)

    def update_hopping_prob(self, state, dv_ave):
        b =  - 2*(state.rho_el.conjugate() * dv_ave).real
        state.hopping_prob = self.dt * b[:, state.el_state] / state.rho_el[state.el_state, state.el_state].real

    def try_hop(self, state:state.State):

        g = np.maximum(state.hopping_prob, np.zeros_like(state.hopping_prob))    # negative => set to 0
        g[state.el_state] = 1.0 - np.sum(g)

        # hopping or not?
        new_el_state = searchsorted(np.cumsum(g), np.random.uniform())

        if new_el_state != state.el_state:
            new_PE = state.ad_energy[new_el_state]
            
            PE, KE = self.get_energy(state)
            if new_PE - PE > KE:   # frustrated
                pass

            else:
                KE = KE - (new_PE - PE)
                direction = 0.5*(self.d_old[state.el_state, new_el_state] + self.d_new[state.el_state, new_el_state])
                direction /= np.linalg.norm(direction)
                state.v = sqrt(KE * 2.0 / self.m)*direction  #Scale velocity along drv coupling
                state.el_state = new_el_state
                return True

        return False

    def get_energy(self, state:state.State):
        
        return state.ad_energy[state.el_state], 0.5*np.dot(m, state.v*state.v)


def outside_box(state, box):
    
    return np.logical_and(state.x>box[:,1], state.v>0) or np.logical_and(state.x < box[:,0], state.v<0)

def outside_which_wall(state, box):

    if (state.x > box[:,1]).any():
        return np.argwhere(state.x > box[:,1])[0]*2+1
    elif (state.x < box[:,0]).any():
        return np.argwhere(state.x > box[:,0])[0]*2
    else:
        return -1
        
    
