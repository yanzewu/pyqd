import numpy as np
import numpy.linalg as LA

from . import state

class Evaluator:
    """ Provide interface for model that 
    does not diagonalize and evaluate derivative coupling
    """

    # The work that evaluator done should be achieved by model,
    # but here for compativity I included it.

    def __init__(self, model):
        self.model = model

    def update_potential(self, state:state.State):
        """ Update the state's
        force = -<k|dH|k>
        ad_energy, ad_states = eigh(H)
        drv_coupling = <i|H|j>/(E[j] - E[i])
        """
        
        state.ad_energy, ad_states_new = LA.eigh(self.model.V(state.x))
        try:
            ad_states = self._align_ad_states(ad_states_new)
        except AttributeError:
            ad_states = ad_states_new
            
        dV = self.model.dV(state.x)

        state.force = ad_states[:, self.state].T.dot(dV.dot(ad_states[:, self.state]))
        state.drv_coupling = self._drv_coupling(dV, ad_states, state.ad_energy)   # Should be achieved by model
        self.ad_states_old = ad_states

    def _align_ad_states(self, ad_states_new):

        ad_states = np.zeros_like(ad_states_new)
        for j in range(ad_states_new.shape[1]):
            if linalg.norm(ad_states_new[:,j] - self.ad_states_old[:,j]) > 1.0:
                ad_states[:,j] = -ad_states_new[:,j]
            else:
                ad_states[:,j] = ad_states_new[:,j]
        return ad_states

    def _drv_coupling(self, dV, ad_states, ad_eng):

        # This could be simplified using matrix multiplication
        d = np.zeros_like(dV)
        for n in range(dV.shape[2]):
            for i in range(dV.shape[0]):
                for j in range(i+1, dV.shape[1]):
                    d[i,j,n] = ad_states[:,i].T.dot(dV[:,:,n].dot(ad_states[:,j]))/(ad_eng[j]-ad_eng[i])

        d = d - d.T  # antisymmetric
        return d
