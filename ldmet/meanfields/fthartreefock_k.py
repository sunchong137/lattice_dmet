import numpy as np
import scipy.linalg as la
import time
from utils import *
import diis
from scipy.optimize import brentq,newton
from scipy.optimize import minimize_scalar as ms
import sys
sys.path.append('../')
import misc

def ni_calc_k(dmet, Nelec, Hcore, kgrid = None, needEnergy = True, display=True, muFind = False, muguess = None):

    N = np.array([dmet.Nbasis])
    I = np.array([dmet.Nimp])

    L = np.prod(N)*np.prod(I)

    if(kgrid is None):
        kgrid = misc.createKgrid(N,I)

    H = Hcore 

    evals,orbs,rho,_,ctime = misc.diagonalize(H,L,N,I,kgrid)
    rho = rho.real
    cndn = L-np.trace(rho[L:,L:])
    cnup = np.trace(rho[:L,:L])
    cne = cndn+cnup

    if(needEnergy):
        Enew = bcsEnergy(Hcore,vcorr,rho)
    else:
        Enew = 0.0

    return rho,orbs,Enew,evals

