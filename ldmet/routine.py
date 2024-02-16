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

'''Functions used in DMET.'''
import numpy as np

def expand_u(norb, u_imp):
    '''
    Given the correlation potential of the impurity, generate the 
    u matrix for the whole lattice.
    args:
        norb: int, number of orbitals
        u_imp: 2D array or 3D array.
    Returns:
        An array of size (2, norb, norb).
    '''
    ndim = u_imp.ndim 
    nimp = u_imp.shape[-1]
    if norb % nimp > 0:
        raise ValueError("(Norb / Nimp) is not an integer!")
    n_rep = norb // nimp 
    if ndim == 3: # spin-up and spin-down different umat
        u_lat = np.kron(np.eye(n_rep), u_imp)
    else: 
        u_a = np.kron(np.eye(n_rep), u_imp)
        u_lat = np.array([u_a, u_a])
    return u_lat


def gen_rotmat_from_mix(norb, nimp, rdm1, tol=1e-15):
    '''
    SVD of the cross part of RDM1.
    '''
    nenv = norb - nimp 
    b, s, _ = np.linalg.svd(rdm1[:, nimp:, :nimp], full_matrices=False)
    if np.prod(s) < tol:
        print("WARNING: tot enough bath orbitals!")
    r = np.zeros((2, norb, 2 * nimp))
    r[0][:nimp, :nimp] = np.eye(nimp)
    r[1][:nimp, :nimp] = np.eye(nimp)
    r[0][nimp:, nimp:] = b[0]
    r[1][nimp:, nimp:] = b[1]
    return r

def gen_bath_from_env(norb, nimp, rdm1, tol=1e-15):
    pass 

def get_core():
    pass 

def gen_ham_embedding(h1e, h2e, rotmat, iform=True):
    '''
    Generate the embedding Hamiltonian.
    Args:
        h1e: 2D or 3D array.
        h2e: 4D or 5D array.
        
    '''
    pass
    