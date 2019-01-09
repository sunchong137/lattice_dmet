#!/usr/bin/python

import numpy as np
import scipy.linalg as la
import time
import sys
sys.path.append('../')
from utils import *
import diis
from scipy.optimize import brentq,newton 
import sys

def ni_calc(Nelec,Hcore,vcorr,gs, S=None, needEnergy = False, display=True, muFind=False, muguess=0.0):

    if ( S is None ):
        S = np.identity(Hcore.shape[0])

    L = Hcore.shape[0]/2
	
    def muCreate(mu):
        M = np.zeros_like(Hcore,dtype=np.float64)
        np.fill_diagonal(M[:L,:L],-mu)
        np.fill_diagonal(M[L:,L:],mu)	
    	return M

    def mufit(mu):
        H = Hcore + vcorr + muCreate(mu)
        evals,orbs = diagonalize(H,S)
	
        rho = np.dot(orbs[:,:L],orbs[:,:L].conjugate().T);
        cndn = L-np.trace(rho[L:,L:])
        cnup = np.trace(rho[:L,:L])
        cne = cndn+cnup
	
#    	for i in range(orbs.shape[1]):
#            ne = np.dot(orbs[:dL,i],orbs[:dL,i].conjugate())
# 	    nh = np.dot(orbs[dL:,i],orbs[dL:,i].conjugate())
#           print "orbital ",i," np = ",ne," nh = ",nh," Energy = ",evals[i]
        if(display):
            print "mu: ",mu," N = ",cne," Nup = ",cnup," Ndn = ",cndn,' gap = ',evals[L],' energy = ',bcsEnergy(Hcore,vcorr,rho)
            sys.stdout.flush()
#	print
        dne = cne - Nelec*1.0
    	return dne

    if(muFind):
        success = False
        delta = 0.5
	offset = delta*gs[0]
        while(not success):
            try:
                #mu = newton(mufit,delta*gs[0],tol=1.0e-4)
                mu = brentq(mufit,delta+offset,offset-delta,xtol=1.0e-4)   
                success = True
            except ValueError:
                delta += 0.5
        M = muCreate(mu) 
        H = Hcore + vcorr + M		
    else:
        H = Hcore + vcorr 		
        M = np.zeros_like(H)
 
    evals,orbs = diagonalize(H,S)   

#   displayMatrix(orbs)

    rho = np.dot(orbs[:,:L],orbs[:,:L].conjugate().T);
    cndn = L-np.trace(rho[L:,L:])
    cnup = np.trace(rho[:L,:L])
    cne = cndn+cnup

    if(display and muFind):
        print "F mu: ",mu," cne: ",cne

    '''	
    displayMatrix(H)
    ro = extractImp(4,rho)
    displayMatrix(ro)
    print evals
    displayMatrix(orbs)
    '''
    if(needEnergy):
        Enew = bcsEnergy(Hcore,vcorr,rho)
    else:
        Enew = 0.0
	
    return M,rho,orbs,Enew,evals
#####################################################################

def bcsEnergy(Hcore,Vcorr,rho):
        
    L = Hcore.shape[0]/2
    E1 = 0.0
    E2 = 0.0
    for i in range(L):
        for j in range(L):
            E1 += Hcore[i,j]*rho[i,j] + Hcore[i+L,j+L]*rho[i+L,j+L]
            E2 += Vcorr[i,j]*rho[i,j] + Vcorr[i+L,j+L]*rho[i+L,j+L] + 2.0*Vcorr[i,j+L]*rho[i,j+L]
    
    return E1+0.5*E2   

#####################################################################
def rdmToArray(P, symmetry='HERM'):
	
    #The most basic is Hermitian Symmetry
    a = [P[i,i].real for i in range(P.shape[0])]
    a += [P[i,j].real for i in range(P.shape[0]) for j in range(i+1,P.shape[1])]
    if(np.iscomplexobj(P)):
	a += [P[i,j].imag for i in range(P.shape[0]) for j in range(i+1,P.shape[1])] 
    return np.array(a)
				
def arrayTordm(P, array, symmetry='HERM'):

    p = 0
    offset = P.shape[0]*(P.shape[0]-1)/2
    obj = np.zeros_like(P)	 
    if(np.iscomplexobj(P)):

	for i in range(P.shape[0]):
	    obj[i,i] = complex(array[p],0.0)
	    p += 1

        for i in range(P.shape[0]):
   	    for j in range(i+1,P.shape[1]):	
		obj[i,j] = complex(array[p],array[p+offset])
		obj[j,i] = complex(array[p],-array[p+offset])	
		p += 1
    else:

	for i in range(P.shape[0]):
	    obj[i,i] = array[p]
	    p += 1
	
        for i in range(P.shape[0]):
   	    for j in range(i+1,P.shape[1]):	
		obj[i,j] = array[p]
		obj[j,i] = array[p]
		p += 1

    return obj
	
#####################################################################
def rdm_1el(C,Ne):
    #subroutine that calculates and returns the one-electron density matrix
    Cocc = C[:,:Ne]
    P = np.dot( Cocc, np.conjugate(np.transpose(Cocc)))
    return P

#####################################################################

def rdm_2el(P):    
    #Recall physics = b^+ b^+ b b i.e. <ij|kl>
    #chemistry is b^+b b^+ b i.e. (il|jk)
    #We are returning in chemist notation 
    norb = P.shape[0]
    P2 = np.zeros([norb,norb,norb,norb],dtype=P.dtype)
    for i in range(0,norb):
        for j in range(0,norb):
            for k in range(0,norb):
                for l in range(0,norb):
		    P2[i,l,j,k] = P[i,l]*P[j,k] - P[i,k]*P[j,l]

    return P2 

#####################################################################

def make_h2el(g,P):
    #subroutine that calculates the two-electron contribution to the fock matrix
    #g is in quantum chemistry notation

    h2el = np.zeros_like(P,dtype=P.dtype)
    if(g.shape[0] == 2):
    	nb = P.shape[0]/2
        np.fill_diagonal(h2el[:nb,:nb], -P[nb:,nb:].diagonal()*g[0])
        np.fill_diagonal(h2el[nb:,nb:], -P[:nb,:nb].diagonal()*g[0])

	#Below is NOT part of standard hubbard model
	#np.fill_diagonal(h2el[:nb,nb:], P[nb:,:nb].diagonal() ,-g[0])
	#np.fill_diagonal(h2el[nb:,:nb], P[:nb,nb:].diagonal() ,-g[0])	
    else:
	assert(1==2)
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

#####################################################################

