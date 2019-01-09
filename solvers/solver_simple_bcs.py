import os
import sys
import utils
import numpy as np
from scipy.optimize import newton,brentq 
from scipy import stats
import pyscf.fci.fci_spinless as fci

def microIterationSimpleBCS(dmet,mus,R,h_emb,V_emb, targetN, hfgap, gtol, maxitr=50):

    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    sys.stdout.flush() 

    impd = []
    done = False
    microitr = 0
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis])

    #mus[0] = 0.0
    starti = 1

    oldvalueset = False
    orbsn = None

    while not done and microitr<maxitr:
        
        #print 
        print "Microiteration: ",microitr,
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
        print " mu: %16.9e" %(mu),
        sys.stdout.flush() 
        
        Mum = np.zeros([dmet.Nbasis*2,dmet.Nbasis*2])
        #np.fill_diagonal(Mum[:dmet.Nbasis,:dmet.Nbasis],mu)
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
        elif (microitr <= 4):

            mus.append(0.0)
            '''
            if (impd[microitr]>targetN and abs(hfgap)>1.0e-9):
                mus[1] = -hfgap
            elif(abs(hfgap)>1.0e-9):
                mus[1] = hfgap
            '''
            if(mus[microitr]<0):
                sgn = -1.0
            else:
                sgn = 1.0

            if(impd[microitr]>targetN):
                mus[microitr+1] = (mus[microitr]+1.0e-4)*(1.0 - sgn*0.2)
            else:
                mus[microitr+1] = (mus[microitr]+1.0e-4)*(1.0 + sgn*0.2)

                    
            '''            
            elif (microitr>=starti and microitr<=3):                                 
                slope, intercept, r_value, p_value, std_err = stats.linregress(impd, mus)
                newmu = slope*targetN + intercept
            
                mus.append(newmu)
                #print "R-squared of fit",r_value**2    
            '''
        elif(microitr>4):             
            '''
            r = abs(np.asarray(impd)-targetN)
            idx = np.argsort(r) 
            newmu = mus[idx[0]] + (mus[idx[0]]-mus[idx[1]])*rnd.random()*1.0e-1
            newmu1 = mus[idx[0]] + (mus[idx[0]]-mus[idx[1]])*rnd.random()*1.0e-2
            microitr = -1 
            
            mus = []
            mus.append(newmu)
            mus.append(newmu1)
            
            impd = []
            '''

            
            #Get 5 nearest points
            found = False
            wsz = 4
            while not found:
                r = abs(np.asarray(impd)-targetN)
                idx = np.argsort(r) 
                simpd = [impd[idx[x]] for x in range(wsz)]
                smus = [mus[idx[x]] for x in range(wsz)] 
                slope, intercept, r_value, p_value, std_err = stats.linregress(simpd, smus)
                newmu = slope*targetN + intercept

                duplicate=False
                for x in mus:
                    if (abs(newmu-x)<1.0e-9):
                        duplicate=True
                        wsz += 1
                        break

                if not duplicate:
                    mus.append(newmu)
                    found = True
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

    if not done:
        print "ERROR: Embedded problem did not converge. Try other strategies"
        sys.exit()  

    return mus[len(mus)-1]    

