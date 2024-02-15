import os
import sys
import utils
import numpy as np
import time
import string
import shutil
import random
import subprocess
sys.path.append('../')
import ftmodules
from scipy.optimize import newton,brentq 

    
def microIteration(dmet,mu0,R,h_emb,V_emb,targetN, gtol, maxiter=20, getrdm2=False):
    '''
    Dependency: itensor
    '''
            
    print "Figuring out correct chemical potential for target density (on impurity): ",targetN
    sys.stdout.flush() 
    Mum = np.zeros([2*dmet.Nbasis,2*dmet.Nbasis],dtype = np.float64)


    def fn(mu, muc, dn):
    
        #Construct mu matrix
        for i in range(dmet.Nimp):
            Mum[i,i] = -mu
            Mum[i+dmet.Nbasis,i+dmet.Nbasis] = -mu
        MumB = np.dot(R.conj().T,np.dot(Mum,R))
    
        #******************************************************************************     
        #Do high level calculation with FCI
        t1 = time.time()
        Pn, P2n, Enewn= call_mps(dmet,h_emb+MumB,V_emb,dmet.actElCount,getrdm2=getrdm2) 
        t2 = time.time()
        #print "Time used to solve impurity is ", t2-t1
        #******************************************************************************     
        
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
    
        #******************************************************************************     
        #Do high level calculation with FCI
        t1 = time.time()
        Pn, P2n, Enewn= call_mps(dmet,h_emb+MumB,V_emb,dmet.actElCount,getrdm2=getrdm2) 
        t2 = time.time()
        #print "Time used to solve impurity is ", t2-t1
        #******************************************************************************     
        
        dg = Pn.diagonal()
        cd = sum(dg[:2*dmet.Nimp])
        
        print "mu = %16.9e\tVariational Energy = %10.6e\tDensity on impurity = %10.6e" %(mu,Enewn,cd)
        sys.stdout.flush() 

        dmet.ImpurityEnergy = Enewn
        dmet.IRDM1 = Pn
        dmet.IRDM2 = P2n
        dmet.MumB = MumB
    
    if not getrdm2:
    
        #******************************************************************************     
        #Do high level calculation with FCI
        t1 = time.time()
        Pn, P2n, Enewn= call_mps(dmet,h_emb+dmet.MumB,V_emb,dmet.actElCount,getrdm2=True) 
        t2 = time.time()
        dmet.IRDM2 = P2n
        print "Time used to solve impurity is ", t2-t1
        #******************************************************************************     

    return mu            
    
###############################################################################     
def call_mps(dmet,h_emb,V_emb,nelec,tol=1E-10,nmpsiter=2,getrdm2=False,tmpdir=None,mpsdir=None,fiedler=True):

    norb  = h_emb.shape[-1]/2
    nimp  = dmet.Nimp
    u     = dmet.g2e_site[0]
    ibath = dmet.iformalism 
    maxm  = dmet.MaxMPSM
    #if ibath:
    #    ibath = "yes"
    #else:
    #    ibath = "no"
    
    fiedler = False #FIXME implement permutation with V_emb, also check whether 1-body term
                    #FIXME is the criteria for choosing Fiedler order.
    # generate a random string for tmp dir 
    randstr = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(6)])
    
    if tmpdir == None:
        #tmpdir = '/scratch/global/sunchong/dmet/finiteTMPS'+ randstr +'/'
        tmpdir = '/home/sunchong/work/jobs/scratch/dmet/'+ randstr +'/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if mpsdir == None:
        mpsdir = '/home/sunchong/work/itensor/sample/'
    
    hamfile = tmpdir + 'hamfile.txt' #1 body Hamiltonian
    intfile = tmpdir + 'intfile.txt' #2 body Hamiltonian
    impidfile = tmpdir + 'impidfile.txt'
    infile  = tmpdir + 'input_mps'

    # generate DMRG schedule
    maxiter_schedule = 10
    i = 0
    mlst = []
    m = 100
    noise_lst = []
    minnoise = 1e-12
    noise = 1e-7
    while (i<maxiter_schedule):
        if (maxm-m < 100 or m>=maxm):
            m = maxm
            noise = minnoise
            mlst.append(m)
            mlst.append(m)
            noise_lst.append(noise)
            noise_lst.append(0)
            break
        mlst.append(m)
        noise_lst.append(noise)
        noise *= 1e-1
        if (noise==minnoise*10):
            noise = 0.0
        m *= 2    
    nsweeps = len(mlst)
    
        
    #### generate input file
    fin = open(infile, 'w')
    fin.write("input\n{\n")
    fin.write("N = %d\n"%norb)
    fin.write("Npart = %d\n"%nelec)
    fin.write("Nimp = %d\n"%nimp)
    fin.write("U = %f\n"%u)
    fin.write("ibath = %s\n"%ibath)
    fin.write("getrdm2 = %s\n"%getrdm2)

    fin.write("hamfile = %s\n"%hamfile)
    fin.write("intfile = %s\n"%intfile)
    fin.write("impsite = %s\n"%impidfile)
    fin.write("outdir = %s\n"%tmpdir)
    # schedule part
    fin.write("nsweeps = %d\n"%nsweeps)
    fin.write("sweeps\n")
    fin.write("{\n")
    fin.write("maxm  minm  cutoff  niter  noise\n")
    for i in range(nsweeps):
        fin.write("%d   %d   %s   %d   %s\n"%(mlst[i],20,str(tol),nmpsiter,str(noise_lst[i])))
    fin.write("}\n")
    fin.write("quiet = yes\n}")

    fin.close()

    #### write 1 body hamiltonian
    h_embn = ftmodules.perm1el(h_emb, dmet.Nimp, tomix=False)
    
    # Fiedler reordering
    #np.set_printoptions(linewidth=1000)
    h1ea = h_embn[:norb, :norb]
    h1eb = h_embn[norb:,norb:]
    if fiedler:
        fiedler_idx_a = ftmodules.fiedler_order(h1ea) # assume that a and b share the same idx
        fiedler_idx_b = fiedler_idx_a
    else:
        fiedler_idx_a = np.arange(norb)[::-1] 
        fiedler_idx_b = np.arange(norb)[::-1] 
    
    h1ea = ftmodules.fiedler_permute(h1ea, fiedler_idx_a)
    h1eb = ftmodules.fiedler_permute(h1eb, fiedler_idx_b)

    h1e_n = np.asarray([h1ea,h1eb]).reshape(2*norb,norb)
    np.savetxt(hamfile,h1e_n)
    
    #### write 2-body Hamiltonian ####
    '''
    The two body Hamiltonian is under normal order of embedding basis,
    no need to call the permutation function.
    '''
    #TODO reorder orbitals with Fiedler localization
    Vab = V_emb[1].reshape(norb**3,norb) # only the alpha-beta interaction term is non-zero for Hubbard model
    np.savetxt(intfile, Vab)
    
    
    #### get imp id
    imp_id = np.zeros(dmet.Nimp, dtype=np.int8)
    if fiedler:
        for i in range(dmet.Nimp):
            for f_id in range(norb):
                if fiedler_idx_a[f_id] == i:
                    imp_id[i] = norb-1-f_id
    else:
        imp_id = np.arange(dmet.Nimp)
    np.savetxt(impidfile, imp_id+1, fmt='%d')    

    #################### call itensor ######################
    subprocess.call([mpsdir + "impsolver_dmet", infile])
    ########################################################
    # read rdms
    e = np.loadtxt(tmpdir + "energy.txt")
    rdm1 = np.loadtxt(tmpdir + "rdm1s.txt")

    # convert formats of rdm1 and rdm2
    RDM1 = np.zeros_like(h_emb)
    RDM1[:norb,:norb] = ftmodules.fiedler_permute(rdm1[:norb, :], fiedler_idx_a, False)
    RDM1[norb:,norb:] = ftmodules.fiedler_permute(rdm1[norb:, :], fiedler_idx_b, False)
    #RDM1[:norb,:norb] = rdm1[:norb, :]
    #RDM1[norb:,norb:] = rdm1[norb:, :]
    RDM1 = ftmodules.perm1el(RDM1, nimp, tomix=True)
    
    RDM2 = np.zeros((3,norb,norb,norb,norb))
    if getrdm2:
        rdm2 = np.loadtxt(tmpdir + "rdm2.txt")
        if ibath:
            RDM2[1] = rdm2.reshape(norb,norb,norb,norb)
        else:
            for i in range(nimp):
                RDM2[1][i,i,i,i] = rdm2[imp_id[i]]

    # remove tmpdir
    shutil.rmtree(tmpdir)

    return RDM1, RDM2, e
