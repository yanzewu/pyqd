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

    def update_potential_ms(self, state:state.State):
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

        for i in range(len(state.force)):
            state.force[i] = -np.trace(state.rho_el.real.dot(dH[:,:,i])) # WARNING: this could be incorrect for complex H


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
        
        state.force = -dH[state.el_state, state.el_state]
        state.drv_coupling = self._drv_coupling(dH, state.ad_energy)
        

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
        
    def to_adiabatic(self, state):
        """ Convert diabatic density matrix to adiabatic basis
        """
        if self.model.multidim:
            _, ad_states = LA.eigh(self.model.V(state.x))
        else:
            _, ad_states = LA.eigh(self.model.V(state.x[0]))
        return ad_states.T.dot(state.rho_el.dot(ad_states))

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
