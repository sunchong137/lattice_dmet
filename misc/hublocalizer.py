from pyscf import lo
from pyscf import gto
import numpy as np

'''
    Thanks to ZHC for this module
'''

class HubbardPM(lo.pipek.PM):
    def __init__(self, *args, **kwargs):
        lo.pipek.PM.__init__(self, *args, **kwargs)
        self.init_guess = 'rand'

    def atomic_pops(self, mol, mo_coeff, method=None):
        return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)

    def get_init_guess(self, key='atomic'):
        '''Generate initial guess for localization.
        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        nmo = self.mo_coeff.shape[1]
        if isinstance(key, str) and key.lower() == 'atomic':
            u0 = atomic_init_guess(self.mol, self.mo_coeff)
        else:
            u0 = np.eye(nmo)
        if (isinstance(key, str) and key.lower().startswith('rand')
            or np.linalg.norm(self.get_grad(u0)) < 1e-5):
            # Add noise to kick initial guess out of saddle point
            dr = np.cos(np.arange((nmo-1)*nmo//2)) * np.random.rand()
            u0 = self.extract_rotation(dr)
        return u0
