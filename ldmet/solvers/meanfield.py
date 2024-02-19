# Copyright 2024 Lattice DMET developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

def uhf():
    return None

def make_h2el(g2e, rdm_core):
    '''
    Args:
        g2e: the lattice two-electron integral.
        rdm_core: 
    '''
    #subroutine that calculates the two-electron contribution to the fock matrix
    #g is in quantum chemistry notation
    
    nb = rdm_core.shape[0]/2
    h2el = np.zeros_like(rdm_core)
    ndim = g2e.ndim
    if ndim == 5:
        nb = rdm_core.shape[0]/2
        np.fill_diagonal(h2el[:nb,:nb], P[nb:,nb:].diagonal()*g[0])
        np.fill_diagonal(h2el[nb:,nb:], P[:nb,:nb].diagonal()*g[0])

    else:
        Jij = np.zeros_like(P)
        Xij = np.zeros_like(P)
    
        nb = P.shape[0]/2
        
        Jij[:nb,:nb] = np.tensordot(P[:nb,:nb],g,axes=([0,1],[3,2])) + np.tensordot(P[nb:,nb:],g,axes=([0,1],[3,2]))
        Jij[nb:,nb:] = Jij[:nb,:nb] 
        
        Xij[:nb,:nb] = np.tensordot(P[:nb,:nb],g,axes=([0,1],[1,2]))
        Xij[:nb,nb:nb+nb] = np.tensordot(P[:nb,nb:nb+nb],g,axes=([0,1],[1,2]))
        Xij[nb:nb+nb,:nb] = np.tensordot(P[nb:nb+nb,:nb],g,axes=([0,1],[1,2]))
        Xij[nb:nb+nb,nb:nb+nb] = np.tensordot(P[nb:nb+nb,nb:nb+nb],g,axes=([0,1],[1,2]))
    
        h2el = Jij - Xij

    return h2el