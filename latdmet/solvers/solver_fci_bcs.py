import os
import sys
import utils
import numpy as np
from scipy.optimize import newton,brentq 
import pyscf.fci.fci_spinless as fci

def microIterationBCS(dmet,mu0,R,h_emb,V_emb,targetN, gtol, maxiter = 50):
            
    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    sys.stdout.flush() 
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis],dtype = np.float64)

    le = (h_emb.shape[0]-2*dmet.Nimp)/2
    
    def fn(phi, muc, dn):

        #Construct mu matrix to adjust particles on impurity
        phi = np.real(phi)

        #Note that h_emb already has globalMu in it
        off = np.zeros_like(dmet.globalMu)
        #goff = np.zeros_like(dmet.globalMu)
        #np.fill_diagonal(off[:dmet.Nbasis,:dmet.Nbasis],phi) 
        #np.fill_diagonal(off[dmet.Nbasis:,dmet.Nbasis:],-phi)
        #off = np.copy(goff)
        for i in range(dmet.Nimp):
            off[i,i] = -phi
            off[i+dmet.Nbasis,i+dmet.Nbasis] = phi

        '''
        for i in range(le):
            Mum[i+dmet.Nimp,i+dmet.Nimp] += -phi
            Mum[i+dmet.Nbasis+dmet.Nimp,i+dmet.Nimp+dmet.Nbasis] += phi #For Hole Sector
        '''
        MumB = np.dot(R.conj().T,np.dot(off,R))

        #utils.displayMatrix(MumB)

        ###############################################################################     
        #Do high level calculation with FCI

        ###########
        
        #h1body = h_emb+MumB
        #np.save('h1body.dat',h1body)
        #np.save('h2body.dat',V_emb)
        #np.save('h1badj.dat',h_emb-0.5*dmet.jk_c)            
        ###########

        Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount) 
        Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)   

        '''
        if(dmet.jk_c != []): 
            Efrag = dmet.get_impurityEnergy(h_emb-0.5*dmet.jk_c, V_emb, Pn, P2n).real
        else:
            Efrag = dmet.get_impurityEnergy(h_emb, V_emb, Pn, P2n).real
        print "Efrag = ",Efrag
        '''

        #Do high level calculation with HF (FOR CHECKING)
        #Pn, orbsn, Enewn, evalsn = hf.hf_calc(dmet.actElCount,h_emb+MumB,V_emb)
        #P2n = hf.rdm_2el(Pn)
        ###############################################################################     
        
        dg = Pn.diagonal()
        #print phi,dg
        
        impnup = sum(dg[:dmet.Nimp])
        impndn = dmet.Nimp - sum(dg[dmet.Nimp:dmet.Nimp*2])
        impdensity = impnup+impndn
                
        nb = (R.shape[1] - 2*dmet.Nimp)/2
        #bathup = sum(dg[2*dmet.Nimp:2*dmet.Nimp+ne])
        #bathdn = sum(dg[2*dmet.Nimp+ne:])
        bathdensity = sum(dg[2*dmet.Nimp:])

        #dmet.ImpurityEnergy = Enewn + dmet.g2e_site[0]*dmet.Nbasis/2 
        dmet.ImpurityEnergy = Enewn  
        print "mu = %16.9e\tVariational Energy = %10.6e\tDensity on impurity = (%10.6e + %10.6e) %10.6e\tDensity on bath: %10.6e\ttotal = %10.6e" %(phi,dmet.ImpurityEnergy,impnup,impndn,impdensity,bathdensity,impdensity+bathdensity)
        sys.stdout.flush() 

        #Update global observables
        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
        dmet.MumB = MumB

        #sys.exit()
        #Book Keeping
        muc.append(phi)
        dn.append(impdensity-targetN)
        return (impnup+impndn-targetN)

    muc = []
    dn = []
    success = False        

    delta = 0.5        
    while(not success):
        try:    
            #mu = least_squares(fn, -0.1, ftol = gtol, args=(muc,dn))
            #mu = brentq(fn,delta,-delta,xtol=gtol,args=(muc,dn))   
            success = True
            mu = newton(fn,mu0, args=(muc,dn), tol=gtol, maxiter = maxiter)
       
            print "Final mu: ",mu
            dd = fn(mu,muc,dn)
            print "Converged with tol: ",dd
            
            #goff = np.zeros_like(dmet.globalMu)
            #np.fill_diagonal(goff[:dmet.Nbasis,:dmet.Nbasis],mu) 
            #np.fill_diagonal(goff[dmet.Nbasis:,dmet.Nbasis:],-mu)
            #dmet.globalMu += goff
        
        except ValueError:
            delta += 0.5

        except RuntimeError:
            print "Not implemented"
            assert(false)
            success = True    
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
                Mum[i+dmet.Nbasis,i+dmet.Nbasis] = mu
            MumB = np.dot(R.conj().T,np.dot(Mum,R))
        
            ###############################################################################     
            #Do high level calculation with FCI
            Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount) 
            Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)   
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
    
