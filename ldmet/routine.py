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
from ldmet.solvers import meanfield

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

def project_ham(h1e, h2e, rotmat, nimp, iform=True):
    '''
    Project the lattice Hamiltonian to embedding space.
    The order of g2e spins take the Pyscf convention: [aa, ab, bb]
    Args:
        h1e: 2D or 3D array.
        h2e: 4D or 5D array.
        rotmat: 3D array, projection operator onto the embedding space.
        nimp: number of impurity orbitals
    Kwargs:
        iform: if True, the interacting formalism is used, where the 2-body interaction from
        the bath is included, otherwise, only include the 2-body interaction within the impurity.
    Returns:
        h1_emb: (2, N_emb, N_emb) array.
        v_emb: 
    '''
    if h1e.ndim == 3:
        h1_emb = np.einsum('sji, sjk, skl -> sil', rotmat.conj(), h1e, rotmat)
    else:
        h1_emb = np.einsum('sji, jk, skl -> sil', rotmat.conj(), h1e, rotmat)

    n_emb = rotmat.shape[-1]

    r_up = rotmat[0]
    r_dn = rotmat[1]
    if iform:
        if h2e.ndim == 4:
            v_aa = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_up.conj(), r_up, h2e, r_up.conj(), r_up)
            v_bb = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_dn.conj(), r_dn, h2e, r_dn.conj(), r_dn)
            v_ab = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_up.conj(), r_up, h2e, r_dn.conj(), r_dn)
        else:
            v_aa = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_up.conj(), r_up, h2e[0], r_up.conj(), r_up)
            v_bb = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_dn.conj(), r_dn, h2e[2], r_dn.conj(), r_dn)
            v_ab = np.einsum('pi, qj, pqrs, rk, sl -> ijkl', r_up.conj(), r_up, h2e[1], r_dn.conj(), r_dn)
        v_emb = np.array([v_aa, v_ab, v_bb])
    else:
        v_emb = np.zeros((3, n_emb, n_emb, n_emb, n_emb))
        if h2e.ndim == 4:
            v0 = h2e[:nimp, :nimp, :nimp, :nimp]
            v_emb[0][:nimp, :nimp] = v0 
            v_emb[1][:nimp, :nimp] = v0 
            v_emb[2][:nimp, :nimp] = v0 
        else:
            v_emb = h2e[:, :nimp, :nimp, :nimp, :nimp]

    return h1_emb, v_emb


def energy_embedding(h1_emb, h2_emb, rdm1_emb, rdm2_emb, nimp=None, iform=True):
    '''
    Evaluate the embedding energy. Interact form.
    '''
    if iform:
        E1 = np.einsum('sij, sji ->', h1_emb, rdm1_emb)
        E2 = 0.5 * np.einsum('sijkl, sjilk ->', h2_emb, rdm2_emb)
    else:
        if nimp is None:
            nimp = rdm1_emb.shape[-1] // 2
        ni = nimp
        E1 = np.einsum('sij, sji ->', h1_emb[:, :ni], rdm1_emb[:ni, :])
        E2 = 0.5 * np.einsum('sijkl, sjilk ->', h2_emb[:, :ni, :ni, :ni, :ni], rdm2_emb[:, :ni, :ni, :ni, :ni])
    return E1 + E2


def gen_emb_ham(h1e, g2e, uimp, fock2e=0, iform=True):
    '''
    Generate embedding hamiltonian from mean field solution.
    '''
    norb = h1e.shape[-1]
    nimp = uimp.shape[-1]
    umat = expand_u(norb, uimp)
    mu, rdm1 = meanfield.uhf(h1e, g2e, umat)

    #Calculate rotation matrix to embedding basis from HF 1RDM (same as calculating bath orbitals)
    rotmat = gen_rotmat_from_mix(norb, nimp, rdm1) 
    rdm_core = get_core()

    h_emb, v_emb = project_ham(h1e, g2e, rotmat, nimp, iform=iform)  
    return h_emb, v_emb, rotmat, rdm_core