import os
import sys
import utils
import numpy as np
from scipy.optimize import newton,brentq 
from scipy.optimize import minimize
#import pyscf.fci.fci_spinless as fci
sys.path.append('../')
import ftmodules

    
def microIteration(dmet,mu0,R,h_emb,V_emb,targetN, gtol, maxiter=20,fitmu=False):

    sys.stdout.flush() 
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis],dtype = np.float64)

    ''' 
    for i in range(V_emb.shape[0]):
        for j in range(V_emb.shape[1]):
            for k in range(V_emb.shape[2]):
                for l in range(V_emb.shape[3]):
                    print i,j,k,l,V_emb[i,j,k,l]
            
    sys.exit()
    '''

    def fn(mu, muc, dn):
    
        #Construct mu matrix
        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = -mu
        MumB = np.dot(R.conj().T,np.dot(Mum,R))
    
        ###############################################################################     
        #Do high level calculation with FCI
        #Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount) 
        #Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)   
        Pn, P2n, Enewn = ftmodules.ftimpsolver(dmet,h_emb+MumB,V_emb,h_emb.shape[0]/2,dmet.actElCount)

        #Do high level calculation with HF (FOR CHECKING)
                #Pn, orbsn, Enewn, evalsn = hf.hf_calc(dmet.actElCount,h_emb+MumB,V_emb)
                #P2n = hf.rdm_2el(Pn)
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

    if fitmu:
        print "Figuring out correct chemical potential for target density (on impurity): ",targetN
        # test if fitmu is needed
        ndiff = fn(mu0, muc, dn)
        if abs(ndiff) < 1e-6:
            print "Fitting chemical potential mu is NOT needed!"
            return mu0
        
        
        try:    
            #mu = minimize(fn, mu0, method='Newton-CG',args=(muc, dn), tol=gtol, options={'disp':True})
            mu = newton(fn, mu0,args=(muc, dn), tol=gtol, maxiter=maxiter)
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
            #Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount) 
            #Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)   
            Pn, P2n, Enewn = ftmodules.ftimpsolver(dmet,h_emb+MumB,V_emb,h_emb.shape[0]/2,dmet.actElCount)
            ###############################################################################     
            
            dg = Pn.diagonal()
            cd = sum(dg[:2*dmet.Nimp])
            
            print "mu = %16.9e\tVariational Energy = %10.6e\tDensity on impurity = %10.6e" %(mu,Enewn,cd)
            sys.stdout.flush() 
 
            dmet.ImpurityEnergy = Enewn
            dmet.IRDM1 = Pn
            dmet.IRDM2 = P2n
            dmet.MumB = MumB

    else:
        ndiff = fn(mu0,muc,dn)
        return mu0

    return mu            
    
