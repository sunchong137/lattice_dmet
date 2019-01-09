import os
import sys
import utils
import numpy as np
from scipy.optimize import newton,brentq 
#import pyscf.fci.fci_spinless as fci
from scipy import stats
sys.path.append('../')
import ftmodules
    
def microIterationSimple(dmet,mus,R,h_emb,V_emb, targetN, hfgap, gtol, maxitr=50):

    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    sys.stdout.flush() 

    impd = []
    done = False
    microitr = 0
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis])

    mus[0] = 0.0
    starti = 1

    oldvalueset = False

    while not done and microitr<maxitr:
        
        #print 
        print "itr: ",microitr,
        if(dmet.muFileOption and os.path.isfile(dmet.muFileName)):
            stdin,stdout = os.popen2("tail -n 1 "+dmet.muFileName)
            stdin.close()
            lines = stdout.readlines(); stdout.close()
            value = float(lines[0])

            #Do not use if file hasn't changed
            if(oldvalueset and abs(value-oldvalue)>1.0e-9):
                mus[microitr] = float(lines[0])
                oldvalue = float(lines[0])
            elif((not oldvalueset) and microitr>0):
                mus[microitr] = float(lines[0])
                oldvalue = float(lines[0])
                oldvalueset = True           

        mu = mus[microitr]
        print "mu: %16.9e" %(mu),
        sys.stdout.flush() 
        
        Mum = np.zeros([dmet.Nbasis*2,dmet.Nbasis*2])
        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = -mu
            
        #utils.displayMatrix(Mum)
        MumB = np.dot(R.conj().T,np.dot(Mum,R))
        
        #t1 = time.time()
        ##########################################################################
        #Enewn, orbsn = fci.kernel(h_emb+MumB,V_emb,h_emb.shape[0],dmet.actElCount) 
        #Pn, P2n = fci.make_rdm12(orbsn,h_emb.shape[0],dmet.actElCount)        
        Pn, P2n, Enewn = ftmodules.ftimpsolver(dmet,h_emb+MumB,V_emb,h_emb.shape[0]/2,dmet.actElCount)
        ##########################################################################
        #t2 = time.time()
        #time_for_corr = t2 - t1

        #Calculate the DMET site energy and density of bath
        dg = Pn.diagonal()
        impd.append(sum(dg[:2*dmet.Nimp]))                
           
        #STORE FOR GLOBAL USE LATER            
        dmet.MumB = MumB
        dmet.ImpurityEnergy = Enewn
        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
 
        #utils.displayMatrix(Pn)

        print "\tEnergy per site = %10.6e\tDensity on impurity = %10.6e" %(Enewn/dmet.Nbasis,impd[microitr]),
        print " total: %10.6e" %(sum(dg))           
        sys.stdout.flush() 

        if(abs(impd[microitr]-targetN)<gtol):
            done = True   
        elif (microitr == 0):
            if (impd[microitr]>targetN and abs(hfgap)>1.0e-9):
                mus[1] = -hfgap
            elif(abs(hfgap)>1.0e-9):
                mus[1] = hfgap
        elif (microitr>=starti and microitr<=4):                                 
            slope, intercept, r_value, p_value, std_err = stats.linregress(impd, mus)
            newmu = slope*targetN + intercept
            mus.append(newmu)
            #print "R-squared of fit",r_value**2
        elif(microitr>4):              
            #Get 5 nearest points
            r = abs(np.asarray(impd)-targetN)
            idx = np.argsort(r) 
            simpd = [impd[idx[x]] for x in range(4)]
            smus = [mus[idx[x]] for x in range(4)] 
            slope, intercept, r_value, p_value, std_err = stats.linregress(simpd, smus)
            newmu = slope*targetN + intercept
            mus.append(newmu)
            #print "R-squared of fit",r_value**2
    
        ''' 
        else:
            #Find the two closest points and interpolate
            idx = np.argsort(impd)
            sr = np.sort(impd)      #sorted densities
            msr = [mus[i] for i in idx] #corresponding mus
            loc = np.searchsorted(sr,targetN) #Find where it should be inserted
            
            ll = len(sr)-1
            if(loc>=ll):
                lmu = msr[ll-1]
                rmu = msr[ll]
                lden = sr[ll-1]
                rden = sr[ll]
            elif(loc<=0):
                lmu = msr[0]
                rmu = msr[1]
                lden = sr[0]
                rden = sr[1]
            else:
                lmu = msr[loc-1]
                rmu = msr[loc]
                lden = sr[loc-1]
                rden = sr[loc]

            print lden,rden 
            slope = (rmu-lmu)/(rden-lden)
                    newmu = lmu + slope*(targetN - lden)
            mus.append(newmu)   
        '''         
        microitr += 1

    return mus[len(mus)-1]    

