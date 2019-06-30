import numpy as np
from scipy.integrate import complex_ode

from . import state


class Integrator:

    def __init__(self, dt, m, using_ode=True):
        self.dt = dt
        self.m = m
        self.m_inv = 1.0/m
        self.using_ode = using_ode

    def initialize(self, state:state.State, method='sh'):
        self.a = 0.0

        if method == 'sh':
            self.d_new = state.drv_coupling
            self.E_new = state.ad_energy
        elif method == 'mf':
            self.H_new = state.H_el

    def update_first_half(self, state:state.State):
        state.v += 0.5 * self.a * self.dt
        state.x += state.v * self.dt
    
    def update_latter_half(self, state:state.State):
        
        self.a = self.m_inv * state.force
        state.v += 0.5 * self.a * self.dt

    def update_el_state_mf(self, state:state.State):

        self.H_old = self.H_new
        self.H_new = state.H_el

        self.update_el_state_by_H(state, 0.5*(self.H_old + self.H_new))

    def update_el_state_sh(self, state:state.State):

        self.d_old = self.d_new
        self.d_new = state.drv_coupling
        self.E_old = self.E_new
        self.E_new = state.ad_energy

        dv_ave = 0.5*(self.d_new.dot(state.v) + self.d_old.dot(state.v))
        E_ave = 0.5*(self.E_new + self.E_old)

        self.update_el_state_by_H(state, np.diag(E_ave) - 1j * dv_ave)
        self.update_hopping_prob(state, dv_ave)

    def update_el_state_by_H(self, state, H):
        """ Update electronic state by Liouville equation
        """

        def _liouville(t, rho):
            rho_ = rho.reshape(H.shape)
            return (-1j*(H.dot(rho_) - rho_.dot(H))).flatten()

        if self.using_ode:
            r = complex_ode(_liouville)
            r.set_initial_value(state.rho_el.flatten())
            state.rho_el = r.integrate(self.dt).reshape(H.shape)

        else:   # Verlet integration
            drho = -1j*(H.dot(state.rho_el) - state.rho_el.dot(H))
            rho_el_old_tmp = state.rho_el
            state.rho_el = 2*state.rho_el - self.rho_el_old + drho*dt*dt
            self.rho_el_old = rho_el_old_tmp

    def update_hopping_prob(self, state, dv_ave):
        b =  - 2*(state.rho_el.conjugate() * dv_ave).real
        state.hopping_prob = self.dt * b[:, state.el_state] / state.rho_el[state.el_state, state.el_state].real

    def try_hop(self, state:state.State):

        g = np.maximum(state.hopping_prob, np.zeros_like(state.hopping_prob))    # negative => set to 0
        g[state.el_state] = 1.0 - np.sum(g)

        # hopping or not?
        new_el_state = np.searchsorted(np.cumsum(g), np.random.uniform())

        if new_el_state != state.el_state:
            new_PE = state.ad_energy[new_el_state]
            
            PE, KE = self.get_energy_ss(state)
            if new_PE - PE > KE:   # frustrated
                pass

            else:
                d = 0.5*(self.d_old[state.el_state, new_el_state] + self.d_new[state.el_state, new_el_state])

                # Solve equation 0.5*\sum_i{m_i*{v_i-d_i*scale}^2} = new_KE
                _eq_a = 0.5*(self.m*d).dot(d)
                _eq_b = (self.m*d).dot(state.v)
                _eq_c = new_PE - PE
                _eq_delta = np.sqrt(_eq_b*_eq_b - 4*_eq_a*_eq_c)

                scale = (-_eq_b + _eq_delta)/_eq_a * 0.5
                scale2 = (-_eq_b - _eq_delta)/_eq_a * 0.5

                if abs(scale2) < abs(scale):
                    scale = scale2

                state.v += scale * d
                state.el_state = new_el_state
                return True

        return False

    def get_energy_ss(self, state:state.State):
        
        return state.ad_energy[state.el_state], 0.5*(self.m*state.v).dot(state.v)

    def get_energy_mf(self, state:state.State):

        return np.trace(state.H_el.dot(state.rho_el)).real, 0.5*(self.m*state.v).dot(state.v)



def outside_box(state, box):
    
    return np.logical_and(state.x>box[:,1], state.v>0).any() or np.logical_and(state.x < box[:,0], state.v<0).any()

def outside_which_wall(state, box):

    if (state.x > box[:,1]).any():
        return np.argwhere(state.x > box[:,1])[0]*2+1
    elif (state.x < box[:,0]).any():
        return np.argwhere(state.x < box[:,0])[0]*2
    else:
        return -1
        
    
