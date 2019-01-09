import numpy as np

import sys
sys.path.append('/groups/changroup/members/csun2/dmet_ur/hfb_hubbard_2/')
import utils
from utils import *

from pyscf.ftsolver import ed_grandmu as fted
from pyscf.ftsolver import ftfci_grand as ftlan

from pyscf import lo
from pyscf import gto


####################################################################################
def ftimpsolver(dmet, h_emb,V_emb,Norb,Nelec):

    #print np.sum(V_emb)/h_emb.shape[-1]*2
    h_embn = perm1el(h_emb, dmet.Nimp, tomix=False)
    h1 = (h_embn[:Norb, :Norb], h_embn[Norb:, Norb:])
    #V2_aa = np.zeros((Norb,Norb,Norb,Norb))
    #V2_ab = ftutils.permV(V_emb)[:Norb, :Norb, :Norb, :Norb]
    #V2 = (V2_aa, V2_ab, V2_aa)
    
    # solve the impurity problem
    rdm1, rdm2, e = fted.rdm12s_fted(h1, V_emb, Norb, Nelec, dmet.T, symm=dmet.constrainType, mu=dmet.grandmu)

    #np.save("/home/sunchong/test/rdm1s.npy",np.asarray(rdm1))
    #np.save("/home/sunchong/test/rdm2s.npy",np.asarray(rdm2))
    #np.save("/home/sunchong/test/fcies.npy",e)
    #exit()
    #---------------------------
    RDM1 = np.zeros_like(h_emb)
    RDM1[:Norb, :Norb] = rdm1[0].real
    RDM1[Norb:, Norb:] = rdm1[1].real
    RDM1 = perm1el(RDM1, dmet.Nimp, tomix=True)

#   RDM2 = np.zeros_like(V_emb)
#   RDM2[:Norb,:Norb,:Norb,:Norb] = rdm2[0]
#   RDM2[Norb:,Norb:,Norb:,Norb:] = rdm2[2]
#   RDM2[Norb:,Norb:,:Norb,:Norb] = rdm2[1]
#   RDM2[:Norb,:Norb,Norb:,Norb:] = rdm2[1]

    return RDM1, rdm2, e
    
####################################################################################
def ftlanczos(dmet, h_emb, V_emb, Norb, Nelec):

    h_embn = perm1el(h_emb, dmet.Nimp, tomix=False)
    h1 = (h_embn[:Norb, :Norb], h_embn[Norb:, Norb:])    
    rdm1, rdm2, e = ftlan.rdm12s_ftfci(h1, V_emb, Norb, Nelec, dmet.T, mu=dmet.grandmu,m=100, symm=dmet.constrainType)
    RDM1 = np.zeros_like(h_emb)
    RDM1[:Norb, :Norb] = rdm1[0].real
    RDM1[Norb:, Norb:] = rdm1[1].real
    RDM1 = perm1el(RDM1, dmet.Nimp, tomix=True)

    return RDM1, rdm2, e

####################################################################################

class HubbardPM(lo.pipek.PM):
    def __init__(self, *args, **kwargs):
        lo.pipek.PM.__init__(self, *args, **kwargs)
        self.init_guess = 'rand'
    def atomic_pops(self, mol, mo_coeff, method=None):
        return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)

def localbath(mat, nloop=2):
    mol=gto.M()
    loc_orb = mat.copy()
    for cnt in range(nloop):
        locobj = HubbardPM(mol, loc_orb)
        loc_orb = locobj.kernel()
    return loc_orb

####################################################################################
def fiedler_order(mat):
    # absolute value
    matabs = np.abs(mat)
    # calculate laplacian
    lap = np.sum(matabs, axis=1)
    lap = np.diag(lap) - matabs
    # diagonalize laplacian
    e, w = np.linalg.eigh(lap)
    vec = w[:,1]
    fiedler_idx = np.argsort(vec)
    
    return fiedler_idx   

def fiedler_permute(mat, fiedler_idx, forward=True):
    l = mat.shape[-1]
    expected = fiedler_idx.copy()
    for i in range(l):
        expected[i] = fiedler_idx[l-1-i]

    mat_n = mat.copy()
    for i in range(l):
        for j in range(l):
            if forward:
                mat_n[i,j] = mat[expected[i], expected[j]]
            else:
                mat_n[expected[i], expected[j]] = mat[i,j]
            
    return mat_n

###############################################################################
def perm1el(matin, nimp, tomix=True):
    l = nimp
    lb = matin.shape[-1]/2 - nimp
    perm = np.zeros_like(matin)
    perm[:l,:l] = np.eye(l)
    perm[l:(l+lb), 2*l:(2*l+lb)] = np.eye(lb)
    perm[l+lb:2*l+lb, l:2*l] = np.eye(l)
    perm[2*l+lb:, 2*l+lb:] = np.eye(lb)
    if tomix:
        mato = np.dot(perm.T, np.dot(matin, perm))
    else:
        mato = np.dot(perm, np.dot(matin, perm.T))
    return mato

  
###############################################################################
def analyticGradientT(C0,E,dH,nocc,T,mu):
    L = dH.shape[-1]
    beta = 1./T
    Cocc = C0[:, [i for i in range(L) if E[i] < mu]]
    Cvir = C0[:, [i for i in range(L) if E[i] >= mu]]
    nocc = Cocc.shape[-1]
    
    Eocc = np.array([E[:nocc],]*(L-nocc))
    Evir = np.array([E[nocc:],]*nocc).T        

    Zmocc = -np.divide(np.dot(Cvir.conjugate().T,np.dot(dH,Cocc)),(Evir-Eocc))
    def fermi(E):
        return 1./(1.+np.exp((E-mu)*beta))

    Cmocc = np.dot(Cvir, Zmocc)    
    result = np.dot(Cocc, np.dot(np.diag(fermi(E[:nocc])),Cmocc.conjugate().T))\
            + np.dot(Cmocc, np.dot(np.diag(fermi(E[:nocc])),Cocc.conjugate().T))

    Zmvir = -np.divide(np.dot(Cocc.conjugate().T,np.dot(dH,Cvir)),(Eocc.T-Evir.T))
    Cmvir = np.dot(Cocc, Zmvir)
    result += np.dot(Cvir, np.dot(np.diag(fermi(E[nocc:])),Cmvir.conjugate().T))\
            + np.dot(Cmvir, np.dot(np.diag(fermi(E[nocc:])),Cvir.conjugate().T))

    # add the terms from derivative of fermi function
    #focc = beta/(2.*(1+np.cosh((E[:nocc]-mu)*beta)))
    #fvir = beta/(2.*(1+np.cosh((E[nocc:]-mu)*beta)))
    #result -= np.dot(Cocc, np.dot(np.diag(focc), Cocc.conj().T))\
    #        + np.dot(Cvir, np.dot(np.diag(fvir), Cvir.conj().T))
    #print result
    return result


###############################################################################
def fermi_smearing_occ(mu, mo_energy, beta):                                                                                                                  
    # get rho_mo
    occ = np.zeros_like(mo_energy)
    de = beta * (mo_energy - mu) 
    occ[de < 300] = 1.0 / (np.exp(de[de < 300]) + 1.0)
    return occ


#def analyticGradientT(mo_energy, mo_coeff, mu, beta, fix_mu = True):
##def get_rho_grad_full(mo_energy, mo_coeff, mu, beta, fix_mu = False):
#    """
#    full gradient corresponding to rho change term.
#    d rho_{ij} / d v_{kl} [where kl is triu part of the potential]
# 
#    Math:
#        d rho_ij / d v_kl = partial rho_ij / partial v_kl 
#            + partial rho_ij / partial mu * partial mu / partial v_kl
# 
#    """
# 
#    norb = mo_coeff.shape[0]
# 
#    rho_elec = fermi_smearing_occ(mu, mo_energy, beta)
#    rho_hole = 1.0 - rho_elec
# 
#    # rho_grad_fix_mu:
#    de_mat = mo_energy[:, None] - mo_energy
#    beta_de_mat = beta * de_mat
#    beta_de_mat[beta_de_mat > 300] = 300
#    exp_ep_minus_eq = np.exp(beta_de_mat)
# 
#    zero_idx = np.where(np.abs(de_mat) < 1.0e-13)
#    de_mat[zero_idx] = np.inf
#    de_mat_inv = 1.0 / de_mat
# 
#    K = np.einsum('p, q, pq, pq -> pq', rho_elec, rho_hole,\
#            exp_ep_minus_eq - 1.0, de_mat_inv)
# 
#    for p, q in zip(*zero_idx):
#        K[p, q] = rho_elec[p] * rho_hole[q] * beta
#
#    rho_grad = -np.einsum('mp, lp, pq, sq, nq -> lsmn', \
#            mo_coeff, mo_coeff.conj(), K, mo_coeff, mo_coeff.conj())
#    # symmetrize
#    rho_grad = rho_grad + rho_grad.transpose(1,0,2,3)
#    rho_grad[np.arange(norb), np.arange(norb)] *= 0.5
#    rho_grad = rho_grad[np.triu_indices(norb)]
# 
#    # contribution from mu change
#    if not fix_mu:
#        f = rho_elec * rho_hole    
#        
#        # partial rho_ij / partial mu
#        drho_dmu = mo_coeff.dot(np.diag(f)).dot(mo_coeff.conj().T)
#        drho_dmu *= beta
#        
#        # partial mu / partial v_{kl}
#        E_grad = np.einsum('ki, li -> kli', mo_coeff.conj(), mo_coeff)
#        mu_grad = np.dot(E_grad, f) / (f.sum())
#        mu_grad = mu_grad + mu_grad.T
#        mu_grad[np.arange(norb), np.arange(norb)] *= 0.5
#        mu_grad = mu_grad[np.triu_indices(norb)]
#        
#        # partial rho_{ij} / partial mu * partial mu / partial v_{kl}
#        rho_grad_mu_part = np.einsum('k, ij -> kij', mu_grad, drho_dmu)
# 
#        rho_grad += rho_grad_mu_part
#        
#    return rho_grad
#
