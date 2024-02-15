import os
import sys
import utils
import numpy as np
from scipy.optimize import newton,brentq 
from scipy import stats
import pyscf.fci.fci_spinless as fci

def microIterationBCSbk(dmet,mus,R,h_emb,V_emb, targetN, hfgap, gtol,maxitr=20):

    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    sys.stdout.flush() 

    impd = []
    done = False
    microitr = 0
    
    mu = mus[-1] 
    lastmu = mu
    crossmu = False
    lastbelow = False

    lmu = 0.0
    lN = 0
    umu = 0.0
    uN = 0

    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis])
    starti = 1
    oldvalueset = False
    orbsn = None

    while not done and microitr<maxitr:
        
        #print 
        print "Microiteration: ",microitr,
        print " mu: %16.9e" %(mu),
        sys.stdout.flush() 
        
        Mum = np.zeros([dmet.Nbasis*2,dmet.Nbasis*2])
        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = mu
        MumB = np.dot(R.conj().T,np.dot(Mum,R))
        
        #t1 = time.time()
        Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount,orbsn) 
        Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)        
        #t2 = time.time()
        #time_for_corr = t2 - t1

        #Calculate the DMET site energy and density of bath
        dg = Pn.diagonal()
        
        impnup = sum(dg[:dmet.Nimp])
        impndn = dmet.Nimp - sum(dg[dmet.Nimp:dmet.Nimp*2])
        impdensity = impnup+impndn
        impd.append(impdensity)                
           
        bathdensity = sum(dg[2*dmet.Nimp:])
        #STORE FOR GLOBAL USE LATER            
        dmet.MumB = MumB
        dmet.ImpurityEnergy = Enewn
        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
         
        #utils.displayMatrix(Pn)
        print "\tEnergy = %10.6e\tDensity on impurity (%10.6e %10.6e)= %10.6e" %(Enewn,impnup,impndn,impd[microitr]),
        print " total: %10.6e " %(sum(dg))           
        sys.stdout.flush() 

        if(abs(impd[microitr]-targetN)<gtol):
            done = True   
        elif(microitr==0):
            if(mu<0):
                sgn = -1.0
            else:
                sgn = 1.0
          
            lastmu = mu
    
            zeromu = False
            if(abs(mu)<1.0e-10):
                zeromu = True 
    
            if(impd[microitr]>targetN):                
                mu = (mu)*(1.0 - sgn*0.2)
            else:
                lastbelow = True
                mu = (mu)*(1.0 + sgn*0.2)           

            if(zeromu):
                mu = 1.0e-4

        elif (not crossmu):

            if(mu<0):
                sgn = -1.0
            else:
                sgn = 1.0
          

            if(impd[microitr]>targetN and lastbelow or impd[microitr]<targetN and not lastbelow):
                crossmu = True

                if(impd[microitr-1]<targetN):
                    lN = impd[microitr-1]
                    uN = impd[microitr]
                    lmu = lastmu
                    umu = mu                
                else:
                    lN = impd[microitr]
                    uN = impd[microitr-1]
                    lmu = mu
                    umu = lastmu                

                lastmu = mu
                muslope = (uN-lN)/(umu-lmu)
                mu = (targetN - lN)/muslope + lmu              

            elif(impd[microitr]>targetN):
                lastmu = mu
                mu = (mu)*(1.0 - sgn*0.2)
            elif(impd[microitr]<targetN):
                lastmu = mu
                mu = (mu)*(1.0 + sgn*0.2)
        else:             
            
            if(impd[microitr]<targetN):
                lN = impd[microitr]           
                lmu = mu
            else:
                uN = impd[microitr]
                umu = mu
            
            lastmu = mu
            muslope = (uN-lN)/(umu-lmu)
            mu = (targetN - lN)/muslope + lmu              

        microitr += 1

    if not done:
        print "ERROR: Embedded problem did not converge. Try other strategies"
        sys.exit()  

    return mu   

