import os
import sys
import utils
import random
import string
import shutil
import subprocess
import numpy as np
import time
from scipy.optimize import newton,brentq 
sys.path.append('../')
import ftmodules
from ftsolver_fci import microIteration as ftfci
from solver_mps import microIteration as mps_gs
import itensor_adaptor as adaptor


def microIterationMPS(dmet,mu0,R,h_emb,V_emb,targetN,gtol,maxiter=20, tau=0.1,tol=1E-10):
    
    if dmet.T < 1e-3:
        return mps_gs(dmet,mu0,R,h_emb,V_emb,targetN, gtol, maxiter)
    maxm = dmet.MaxMPSM
    tau = dmet.TimeStep
    mu = dmet.grandmu
    print "Solving impurity problem with chemical potential = \n", dmet.grandmu
    print("tau = %f     maxm = %d      tol=%s\n"%(tau,maxm,str(tol)))
    sys.stdout.flush()
    t1 = time.time()
    Pn, P2n, Enewn = adaptor.call_ftmps(dmet,h_emb,V_emb,mu,tau,maxm,tol)
    t2 = time.time()
    print "Time used to solve impurity is ", t2-t1
    dmet.ImpurityEnergy = Enewn
    dmet.IRDM1 = Pn
    dmet.IRDM2 = P2n

    return mu0

#########################################################################
def microIterationMPS_fitmu(dmet,mu0,R,h_emb,V_emb,targetN,gtol,maxiter=20, tau=0.1,tol=1E-10):

    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    if dmet.T < 1e-3:
        return mps_gs(dmet,mu0,R,h_emb,V_emb,targetN, gtol, maxiter)
    maxm = dmet.MaxMPSM
    tau = dmet.TimeStep
    sys.stdout.flush()
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis],dtype = np.float64)

    if dmet.iformalism:
        call_ftmps = adaptor.call_ftmps_ibath
    else:
        call_ftmps = adaptor.call_ftmps

    def fn(mu, muc, dn):

        #Construct mu matrix
        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = -mu
        MumB = np.dot(R.conj().T,np.dot(Mum,R))

        ###############################################################################     
        #Do high level calculation with MPS
        t1 = time.time()
        Pn, P2n, Enewn = call_ftmps(dmet,h_emb,V_emb,mu,tau,maxm,tol)
        t2 = time.time()
        print "Time used to solve impurity is ", t2-t1
        ###############################################################################     

        dg = Pn.diagonal()
        cd = sum(dg[:2*dmet.Nimp])

        print "mu = %16.9e\tVariational Energy = %10.6e\tDensity on impurity = %10.6e" %(mu,Enewn,cd)
        sys.stdout.flush()

        dmet.ImpurityEnergy = Enewn


        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
        dmet.MumB = MumB

        muc.append(mu)
        dn.append(cd-targetN)

        return (cd-targetN)

    muc = []
    dn = []
    try:
        mu = newton(fn,mu0, args=(muc,dn), tol=gtol, maxiter = maxiter)
    except RuntimeError:

        abv = [abs(x) for x in dn]
        smv = min(abv)
        smi = abv.index(smv)
        mu = muc[smi]

        print "Didn't converge using ",maxiter," iterations. So using mu = ",mu
        #ERROR
        if(abv[smi]>10*gtol):
            raise NameError('Difference in density is too large. Consider increasing maxiter')

        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = -mu
        MumB = np.dot(R.conj().T,np.dot(Mum,R))

        ###############################################################################     
        #Do high level calculation with FCI
        t1 = time.time()
        Pn, P2n, Enewn = call_ftmps(dmet,h_emb,V_emb,mu,tau,maxm,tol)
        t2 = time.time()
        print "Time used to solve impurity is ", t2-t1
        ###############################################################################     

        dg = Pn.diagonal()
        cd = sum(dg[:2*dmet.Nimp])

        print "mu = %16.9e\tVariational Energy = %10.6e\tDensity on impurity = %10.6e" %(mu,Enewn,cd)
        sys.stdout.flush()

        dmet.ImpurityEnergy = Enewn
        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
        dmet.MumB = MumB

    return mu


