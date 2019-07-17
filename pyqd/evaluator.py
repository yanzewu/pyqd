import numpy as np
import numpy.linalg as LA

from . import state
from . import model


class Evaluator:
    """ Provide interface for model that 
    does not diagonalize and evaluate derivative coupling
    """

    # The work that evaluator done should be achieved by model,
    # but here for compativity I included it.

    def __init__(self, model:model.Model):
        self.model = model

    def update_potential_ms_first_half(self, state:state.State):
        """ Update the state's
        force = tr(rho H)
        H_el = V
        """

        if self.model.calc_energy:
            if not self.model.multidim:
                state.H_el = np.diag(self.model.E(state.x[0]))
                dH = self.model.dH(state.x[0])[:,:,np.newaxis]
            else:
                state.H_el = np.diag(self.model.E(state.x))
                dH = self.model.dH(state.x)
       
            state.H_el += -1j*self._drv_coupling(dH, state.ad_energy).dot(state.v)
            # dH may not be correct here

        else:   # this is diabatic basis
            if not self.model.multidim:
                state.H_el = self.model.V(state.x[0])
                dH = self.model.dV(state.x[0])[:,:,np.newaxis]
            else:
                state.H_el = self.model.V(state.x)
                dH = self.model.dV(state.x)

        self.dH_tmp = dH

    def update_potential_ms_latter_half(self, state:state.State):
        # May not be correct for intrinsic complex Hamiltonian
        state.force = -np.trace(state.rho_el.real.dot(self.dH_tmp), 0, 0, 1)

    def update_potential_ss(self, state:state.State):
        """ Update the state's
        force = -<k|dH|k>
        ad_energy = eigvals(H)
        drv_coupling = <i|H|j>/(E[j] - E[i])
        """

        if self.model.calc_energy:

            if not self.model.multidim:
                state.ad_energy = self.model.E(state.x[0])
                dH = self.model.dH(state.x[0]).reshape((self.model.el_dim, self.model.el_dim, 1))
            else:
                state.ad_energy = self.model.E(state.x)
                dH = self.model.dH(state.x)
            
        else:   # do diagonalization myself

            if not self.model.multidim:
                state.ad_energy, ad_states_new = LA.eigh(self.model.V(state.x[0]))
                dV = self.model.dV(state.x[0]).reshape((self.model.el_dim, self.model.el_dim, 1))            
            else:
                state.ad_energy, ad_states_new = LA.eigh(self.model.V(state.x))
                dV = self.model.dV(state.x)

            try:
                ad_states = self._align_ad_states(ad_states_new)
            except AttributeError:
                ad_states = ad_states_new
            self.ad_states_old = ad_states

            dH = self._dH(dV, ad_states)
        
        self.dH_tmp = dH
        state.force = -dH[state.el_state, state.el_state]
        state.drv_coupling = self._drv_coupling(dH, state.ad_energy)
        
    def refresh_force_ss(self, state:state.State):
        state.force = -self.dH_tmp[state.el_state, state.el_state]

    def evaluate(self, x):
        
        E = np.zeros((len(x), self.model.el_dim))
        D = np.zeros((len(x), self.model.el_dim, self.model.el_dim, self.model.kinetic_dim))

        if self.model.calc_energy:
            for i in range(len(x)):
                
                if not self.model.multidim:
                    E[i,:] = self.model.E(x[i][0])
                    dH = self.model.dH(x[i][0])[:,:,np.newaxis]
                else:
                    E[i,:] = self.model.E(x[i])
                    dH = self.model.dH(x[i])

                D[i,:,:,:] = self._drv_coupling(dH, E[i])
        else:
            for i in range(len(x)):

                if not self.model.multidim:
                    E[i,:], ad_states = LA.eigh(self.model.V(x[i][0]))
                    dV = self.model.dV(x[i][0])[:,:,np.newaxis]
                else:
                    E[i,:], ad_states = LA.eigh(self.model.V(x[i]))
                    dV = self.model.dV(x[i])
                
                D[i,:,:,:] = self._drv_coupling(self._dH(dV, ad_states), E[i])     

        return E, D
        
    def to_adiabatic(self, rho, x):
        """ Convert diabatic density matrix to adiabatic basis
        """
        if self.model.multidim:
            _, ad_states = LA.eigh(self.model.V(x))
        else:
            _, ad_states = LA.eigh(self.model.V(x[0]))

        return ad_states.T.dot(rho.dot(ad_states))

    def to_diabatic(self, Y, x):
        """ Convert adiabatic state index / density matrix to diabatic basis.
        Y:  int -> adiabatic state index;
            array -> density matrix;
        """
        if self.model.multidim:
            _, ad_states = LA.eigh(self.model.V(x))
        else:
            _, ad_states = LA.eigh(self.model.V(x[0]))

        if isinstance(Y, int):
            c = ad_states[:, Y]
            return c[:,None].dot(c[None])
        else:
            return ad_states.dot(Y.dot(ad_states.T))

    def _align_ad_states(self, ad_states_new):

        ad_states = np.zeros_like(ad_states_new)
        for j in range(ad_states_new.shape[1]):
            if LA.norm(ad_states_new[:,j] - self.ad_states_old[:,j]) > 1.0:
                ad_states[:,j] = -ad_states_new[:,j]
            else:
                ad_states[:,j] = ad_states_new[:,j]
        return ad_states

    def _dH(self, dV, ad_states):

        # This could be simplified using matrix multiplication
        dH = np.zeros_like(dV)
        for n in range(dV.shape[2]):
            dH[:,:,n] = ad_states.conj().T.dot(dV[:,:,n].dot(ad_states))

        return dH


    def _drv_coupling(self, dH, ad_energy):

        d = np.zeros_like(dH)
        for i in range(dH.shape[0]):
            for j in range(i+1, dH.shape[1]):
                d[i,j,:] = dH[i,j,:] / (ad_energy[j] - ad_energy[i])

        for n in range(d.shape[2]):
            d[:,:,n] -= d[:,:,n].T
            
        return d

    def sample_adiabatic_states(self, state):
        """ Convert the state's diabatic density matrix to adiabatic basis,
        and change the el_state randomly according to population.
        """

        state.rho_el = self.to_adiabatic(state.rho_el, state.x)
        sum_pop = np.cumsum(np.diag(state.rho_el.real))
        state.el_state = np.searchsorted(sum_pop, np.random.uniform())
        

    def recover_diabatic_state(self, state):
        """ Convert the state's adiabatic density matrix to diabatic basis;
        Return the population vector
        """
        if self.model.multidim:
            _, ad_states = LA.eigh(self.model.V(state.x))
        else:
            _, ad_states = LA.eigh(self.model.V(state.x[0]))

        rhonew = state.rho_el.copy()
        np.fill_diagonal(rhonew, 0.0)
        rhonew[state.el_state, state.el_state] = 1.0

        state.rho_el = ad_states.dot(state.rho_el.dot(ad_states.T.conj()))

        return np.diag(ad_states.dot(rhonew.dot(ad_states.T.conj()))).real
        