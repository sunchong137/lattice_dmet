#!/usr/bin/python

import numpy as np
import numpy.fft as fft
import scipy.linalg as la
import time
import sys
sys.path.append('../')
from utils import *
import diis
from scipy.optimize import newton
from scipy.optimize import minimize
from hartreefock import hf_hubbard_calc as zeroThf
import scipy.linalg as la

#####################################################################
def order(x):

        if(x == 0.0):
                x = 1.0e-64

        if(x<1.):
                od = np.floor(np.log10(x)-0.5)
        else:
                od = np.floor(np.log10(x)+0.5)

        return 10.**int(od)
#####################################################################

def hf_hubbard_calc(dmet,Nelec,Hcore,Porig=None,S=None,itrmax=0, needEnergy=False, \
                    assertcheck=False, doDIIS=False, doShift=False, hfoscitr=0, \
                    gap=None):
    


    # HartreeFock procedure for grand-canonical ensemble
    # input: T, mu
    T = dmet.T
    mu = dmet.grandmu
    norb = Hcore.shape[-1]
    #Diagonalize hopping-hamiltonian
    evals,orbs = diagonalize(Hcore)

    #Fermi-Dirac function
    def fermi(mu_):
        return 1./(1.+np.exp((evals-mu_)*beta))

    if T < 1.e-3:
        return zeroThf(Nelec, Hcore, None,needEnergy=needEnergy)
    else:
        beta = 1./T
        eocc = fermi(mu)
    #Form the 1RDM

    P = np.dot(orbs, np.dot(np.diag(eocc),orbs.T.conj()))
   
    if(needEnergy):	
        Enew = np.trace(np.dot(P,Hcore))
    else:
        Enew = 0.0
    return P, orbs, Enew, evals

#####################################################################


def hf_hubbard_calc_Ne(dmet,Nelec,Hcore,Porig=None,S=None,itrmax=0, needEnergy=False, \
                    assertcheck=False, doDIIS=False, doShift=False, hfoscitr=0, \
                    gap=None):
    

    T = dmet.T
    #Hartree-Fcok procedure for spin alpha
    #If you want the total spin, you should sum up all spins
    norb = Hcore.shape[-1]
    #Diagonalize hopping-hamiltonian
    evals,orbs = diagonalize(Hcore)

    #Fermi-Dirac function
    def fermi(mu):
        return 1./(1.+np.exp((evals-mu)*beta))
    if T < 1.e-2:
        beta = np.inf
        eocc = np.ones(norb)
        eocc[Nelec:] *= 0
    else:
        beta = 1./T
        # fit mu
        mu0 = 0.
        mu = minimize(lambda x: (np.sum(fermi(x))-Nelec)**2, mu0, tol=1e-6).x
        eocc = fermi(mu)
    #Form the 1RDM

    P = np.dot(orbs, np.dot(np.diag(eocc),orbs.T.conj()))
    l = P.shape[-1]/2
    
    if(needEnergy):	
        Enew = np.trace(np.dot(P,Hcore))
    else:
        Enew = 0.0
    return P, orbs, Enew, evals


#####################################################################

def hf_calc_simple(Nelec,Hcore,g,Porig=None,S=None, itrmax = 10000):

    #subroutine to perform a hartree-fock calculation

    #full g or symmetric form?
    h2el_fn = []
    if(Hcore.shape[0]==g.shape[0]):
	    h2el_fn = make_h2el_full
    else:
        h2el_fn = make_h2el

    #Initialize over-lap to identity if None provided
    if ( S is None ):
        S = np.identity(Hcore.shape[0])

    #generate initial guess of the density matrix assuming fock operator is given by Hcore
    if ( Porig is None ):
        evals,orbs = diagonalize(Hcore,S)
        P = rdm_1el(orbs,Nelec)
    else:
        P = Porig
        
    Enew = 9999.9
    Ediff = 10.0
    itr = 0
    while Ediff > 1e-8 and itr < itrmax:
        #SCF Loop:

        #calculate 2e- contribution to fock operators
        h2el = h2el_fn(g,P)
       
        #form fock operators
        fock = Hcore+h2el

        #solve the fock equations
        evals,orbs = diagonalize(fock,S)

        #form the density matrices
        P = rdm_1el(orbs,Nelec)

        #calculate the new HF-energy and check convergence
        Eold = Enew
        Enew = Ehf(Hcore,fock,P)
        Ediff = np.fabs(Enew-Eold)

        #print "itr = ",itr,"Ediff = ",Ediff
        
        #accumulate iteration
        itr += 1

    if(itr!= 1 and itr==itrmax):
        print "Max iter reaached. itr = %d, maxiter = %d, Ediff = %10.6e" %(itr,itrmax,Ediff)
        assert(itr<itrmax)
    
    return P, orbs, Enew, evals

#####################################################################

def hf_calc(Nelec,Hcore,g,Porig=None,S=None, itrmax = 1000, needEnergy = False, assertcheck=False, doDIIS = False, doShift = True, hfoscitr = 5, gap = None):

    #subroutine to perform a hartree-fock calculation

    #full g or symmetric form?
    h2el_fn = []
    if(Hcore.shape[0]==g.shape[0]):
	h2el_fn = make_h2el_full
    else:
        h2el_fn = make_h2el

    #Initialize over-lap to identity if None provided
    if ( S is None ):
        S = np.identity(Hcore.shape[0])
    
    #generate initial guess of the density matrix assuming fock operator is given by Hcore
    if ( Porig is None ):
        evals,orbs = diagonalize(Hcore,S)
        P = rdm_1el(orbs,Nelec)
	Corig = orbs
    else:
        P = Porig
	Corig = None
       
    #This is typically used for minimizer 
    if(itrmax == 0):
	h2el = h2el_fn(g,P)
	fock = Hcore+h2el
        evals,orbs = diagonalize(fock,S)
        P = rdm_1el(orbs,Nelec)
	if(needEnergy):
        	Enew = Ehf(Hcore,fock,P)
        else:
		Enew = 0.0

	return P,orbs,Enew,evals
    
    #===========================================================================
    #Full SCF
    #===========================================================================
    #These tolerances must be used consistently
    hfetol = 5e-11
    hfptol = 5e-4

    def hfscf(P, Corig, shiftv = 0.001, hfitrmax = itrmax, debug = False):
       

  	hflstol = 1e-6 #Level shift tolerance

	#For full HF with Self Consistency energy must be computed	    
        Enew = 9999.9
        Ediff = 10.0
	Pdiff = 10.0
        itr = 0
        fock = []
	
	edifflist = []
	pdifflist = []
 
        if(doDIIS):
            diiswindow = 20
            dc = diis.FDiisContext(diiswindow)
            diisdtol = 1e-6 #Tolerance for difference in DIIS vectors
	    diisptol = 1e-4 #Tolerance for difference in P vector
 
        while Ediff > hfetol and itr < hfitrmax:
            #SCF Loop:

            #calculate 2e- contribution to fock operators
            h2el = h2el_fn(g,P)
           
            #form fock operators
            fock = Hcore+h2el
            if(Ediff>hflstol and Corig is not None and doShift):
            	deltavb = np.eye(P.shape[0]-Nelec)*shiftv
            	delta = np.zeros(P.shape)
            	delta[Nelec:,Nelec:] = deltavb
            	#Add shift to virtual orbital block		
            	fock = np.dot(Corig,np.dot( (np.dot(Corig.conjugate().T,np.dot(fock,Corig)) + delta) , Corig.conjugate().T))
            
            #solve the fock equations
            evals,orbs = diagonalize(fock,S)
            Corig = orbs

            #form the density matrices
            pv = rdmToArray(P)
            P = rdm_1el(orbs,Nelec)
            pvnew = rdmToArray(P)
            diffv = pvnew - pv
	    Pdiff = np.linalg.norm(diffv)

            #calculate the new HF-energy and check convergence
            Eold = Enew
	    try:	
	        Enew = Ehf(Hcore,fock,P)
	    except ValueError:
		return itr,10.0,10.0,P,orbs,Enew,evals,False	
            Ediff = np.fabs(Enew-Eold)

            if(doDIIS):
            	skipDiis = (itr<diiswindow or np.linalg.norm(diffv) > diisdtol or Pdiff > diisptol)
            	pvdiis,_,_ = dc.Apply(pvnew,diffv,Skip = skipDiis)
            	if(not skipDiis):
            		P = arrayTordm(P,pvdiis)

            #if(itr>0.20*itrmax and itrmax>200):
	    if(debug):
                print "itr: ",itr,"E: ",Enew,"|dE|: ",Ediff,"|P|: ",Pdiff,
                if(doDIIS):
                	print "diis: ",np.linalg.norm(pvdiis-pvnew),skipDiis
                else: 
                	print
	
            #accumulate iteration
            itr += 1
	    edifflist.append(Ediff)
	    pdifflist.append(Pdiff)
	
        print "SC itr: ",itr,"E: ",Enew," |dE|: ",Ediff," |P|: ",Pdiff," G: ",evals[Nelec] - evals[Nelec-1]

	if(itr >= hfitrmax):
		reduceTest = True
		for i in range(len(edifflist)-5,len(edifflist)-1):
			reduceTest = reduceTest and edifflist[i]>edifflist[i+1] and pdifflist[i]>pdifflist[i+1]
	else:
		reduceTest = True

	return itr, Ediff, Pdiff, P, orbs, Enew, evals, reduceTest

    #The first one is essential
    itr,Ediff,Pdiff,P,orbs,Enew,evals,rt = hfscf(P, Corig, shiftv=0.001)
    if(Ediff<hfetol or itr<itrmax):
	return P, orbs, Enew, evals
    
    yguess = [Ediff,Pdiff,P,orbs,Enew,evals]
    #now comes the hard part
    #First try a longer trajectory if that's what's needed indicated by rt which will
    #be true if values were decreasing
    pcount = 0	 
    while(rt and Pdiff>hfptol and pcount<3):
	#Assume just needs more iterations
	print "c: ",
    	itr,Ediff,Pdiff,P,orbs,Enew,evals,rt = hfscf(P, orbs, shiftv=0.001)
	pcount += 1
	
	score = yguess[0] + order(yguess[0])/order(yguess[1])*yguess[1]*0.20
	scorel = Ediff + order(Ediff)/order(Pdiff)*Pdiff*0.20
	if(scorel<score):
		yguess[0] = Ediff
		yguess[1] = Pdiff
		yguess[2] = P
		yguess[3] = orbs
		yguess[4] = Enew
		yguess[5] = evals
    
	if(Ediff<hfetol):
		return P, orbs, Enew, evals

    if(doShift):	
        #Next assume oscillations in which case we will try more aggressive level shifting
        #Usually can arise because of degneracies
        def costfn(sv, P, orbs, yguess):
            print "shift: ",sv,

            itr,Ediff,Pdiff,P,orbs,Enew,evals,rt = hfscf(P, orbs, shiftv = sv, hfitrmax=50, debug = False)    
            #Assume just needs more iterations under certain circumstances
            pcount = 0	
            while(rt and Pdiff>hfptol and pcount<3 and itr>=50):
            	print "shift c: ",sv,
            	itr,Ediff,Pdiff,P,orbs,Enew,evals,rt = hfscf(P, orbs, shiftv = sv, hfitrmax=50, debug=False)
            	pcount += 1

            score = yguess[0] + order(yguess[0])/order(yguess[1])*yguess[1]*0.20
            scorel = Ediff + order(Ediff)/order(Pdiff)*Pdiff*0.20
            #if(Ediff<yguess[0] and (Pdiff<yguess[1] or Pdiff<5.0e-4)):
            if(scorel<score):
            	yguess[0] = Ediff
            	yguess[1] = Pdiff
            	yguess[2] = P
            	yguess[3] = orbs
            	yguess[4] = Enew
            	yguess[5] = evals
            
            #print "score: ",scorel	
            return scorel	 	

        try:
            sv = newton(costfn,0.2, args=(P,orbs,yguess), tol=hfetol, maxiter=hfoscitr)
            #res = minimize(costfn, 0.002, args=(P,orbs), method='Nelder-Mead', tol=1.0e-9, options = {'maxiter':20})
            #sv = res.x
        except RuntimeError:
            print "Nothing worked. Best guess |dE|: ",yguess[0]," |P|: ",yguess[1]
            return yguess[2],yguess[3],yguess[4],yguess[5]

        itr,Ediff,Pdiff,P,orbs,Enew,evals,rt = hfscf(P, Corig, shiftv = sv)

    ''' 
    if(itr!= 0 and itr==itrmax and assertcheck):
        print "Max iter reaached. itr = %d, maxiter = %d, Ediff = %10.6e" %(itr,itrmax,Ediff)
        assert(itr<itrmax)
    '''

    return P, orbs, Enew, evals


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
    
    nb = P.shape[0]/2
    h2el = np.zeros_like(P,dtype=P.dtype)

    if(g.shape[0] != nb):
        nb = P.shape[0]/2
        np.fill_diagonal(h2el[:nb,:nb], P[nb:,nb:].diagonal()*g[0])
        np.fill_diagonal(h2el[nb:,nb:], P[:nb,:nb].diagonal()*g[0])

        #Below is NOT part of standard hubbard model
        #np.fill_diagonal(h2el[:nb,nb:], P[nb:,:nb].diagonal() ,-g[0])
        #np.fill_diagonal(h2el[nb:,:nb], P[:nb,nb:].diagonal() ,-g[0])	
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

#####################################################################

def make_h2el_full(g,P):
    h2el = np.zeros_like(P,dtype=P.dtype)

    Jij = np.tensordot(P,g,axes=([0,1],[3,2]))
    Xij = np.tensordot(P,g,axes=([0,1],[1,2]))
    h2el = Jij - Xij

    return h2el

#####################################################################

def Ehf(Hcore,fock,P):

    #subroutine that calculates the RHF-energy
    Ehf = 0.5*np.trace( np.dot(P,Hcore)+ np.dot(P,fock) )

    #Remove imaginary part associated with what should be numerical error
    if( Ehf.imag > 1e-15 ):
        print 'ERROR: HF energy has non-neglibigle imaginary part: ',Ehf.imag
        raise ValueError
    else:
        Ehf = Ehf.real

    return Ehf

#####################################################################

