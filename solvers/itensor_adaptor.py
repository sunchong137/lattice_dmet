import os
import shutil
import sys
import random
import string
import subprocess

import numpy as np
sys.path.append('../')
import ftmodules


#########################################################################
def call_ftmps(dmet,h_emb,V_emb,mu,tau=0.01,maxm=1000,tol=1E-7,tmpdir=None,mpsdir=None,fiedler=True):
    
    norb = h_emb.shape[-1]/2
    nimp = dmet.Nimp
    u    = dmet.g2e_site[0]
    #mu   = dmet.grandmu
    beta = 1./dmet.T

    # generate a random string for tmp dir 
    randstr = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(6)])
    
    if tmpdir == None:
        tmpdir = '/scratch/global/sunchong/dmet/finiteTMPS'+ randstr +'/'
        #tmpdir = '/home/sunchong/work/jobs/scratch/dmet/'+ randstr +'/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if mpsdir == None:
        mpsdir = '/home/sunchong/work/finiteTMPS/'
    
    hamfile = tmpdir + 'hamfile.txt'
    impidfile = tmpdir + 'impidfile.txt'
    infile  = tmpdir + 'input_mps'

    # generate input file
    fin = open(infile, 'w')
    fin.write("input\n{\n")
    fin.write("hamfile = %s\n"%hamfile)
    fin.write("impsite = %s\n"%impidfile)
    fin.write("outdir = %s\n"%tmpdir)
    fin.write("N = %d\n"%norb)
    fin.write("Nimp = %d\n"%nimp)
    fin.write("U = %f\n"%u)
    fin.write("mu = %f\n"%mu)
    fin.write("beta = %f\n"%beta)
    fin.write("tau = %f\n"%tau)
    fin.write("maxm = %d\n"%maxm)
    fin.write("cutoff = %s\n"%(str(tol)))
    fin.write("realstep = yes\n")
    fin.write("fitmpo = yes\n")
    fin.write("rungekutta = yes\n")
    fin.write("verbose = no\n}")
    
    fin.close()

    # write 1 body hamiltonian
    h_embn = ftmodules.perm1el(h_emb, dmet.Nimp, tomix=False)
    
    # Fiedler reordering
    #np.set_printoptions(linewidth=1000)
    h1ea = h_embn[:norb, :norb]
    h1eb = h_embn[norb:,norb:]
    if fiedler:
        fiedler_idx_a = ftmodules.fiedler_order(h1ea) # assume that a and b share the same idx
        fiedler_idx_b = fiedler_idx_a
    #fiedler_idx_a = ftmodules.fiedler_order(h1ea[dmet.Nimp:, dmet.Nimp:])+dmet.Nimp
    #fiedler_idx_b = ftmodules.fiedler_order(h1eb[dmet.Nimp:, dmet.Nimp:])+dmet.Nimp
    #fiedler_idx_a = np.concatenate((fiedler_idx_a, np.arange(dmet.Nimp)[::-1]))
    #fiedler_idx_b = np.concatenate((fiedler_idx_b, np.arange(dmet.Nimp)[::-1]))
    else:
        fiedler_idx_a = np.arange(norb)[::-1] 
        fiedler_idx_b = np.arange(norb)[::-1] 
    
    h1ea = ftmodules.fiedler_permute(h1ea, fiedler_idx_a)
    h1eb = ftmodules.fiedler_permute(h1eb, fiedler_idx_b)

    # get imp id
    imp_id = np.zeros(dmet.Nimp, dtype=np.int8)
    for i in range(dmet.Nimp):
        for f_id in range(norb):
            if fiedler_idx_a[f_id] == i:
                imp_id[i] = norb-1-f_id
    np.savetxt(impidfile, imp_id+1, fmt='%d')    

    #h1e_n = np.asarray([h_embn[:norb, :norb],h_embn[norb:,norb:]]).reshape(2*norb,norb)
    h1e_n = np.asarray([h1ea,h1eb]).reshape(2*norb,norb)
    np.savetxt(hamfile,h1e_n)

    subprocess.call([mpsdir + "impsolver_ancilla", infile])

    # read rdms
    e = np.loadtxt(tmpdir + "energy.txt")
    rdm1 = np.loadtxt(tmpdir + "rdm1s.txt")
    rdm2 = np.loadtxt(tmpdir + "rdm2.txt")

    # convert formats of rdm1 and rdm2
    RDM1 = np.zeros_like(h_emb)
    RDM1[:norb,:norb] = ftmodules.fiedler_permute(rdm1[:norb, :], fiedler_idx_a, False)
    RDM1[norb:,norb:] = ftmodules.fiedler_permute(rdm1[norb:, :], fiedler_idx_b, False)
    #RDM1[:norb,:norb] = rdm1[:norb, :]
    #RDM1[norb:,norb:] = rdm1[norb:, :]
    RDM1 = ftmodules.perm1el(RDM1, nimp, tomix=True)
    
    RDM2 = np.zeros((3,norb,norb,norb,norb))
    for i in range(nimp):
        RDM2[1][i,i,i,i] = rdm2[imp_id[i]]

    # remove tmpdir
    shutil.rmtree(tmpdir)

    return RDM1, RDM2, e

#########################################################################
def call_ftmps_ibath(dmet,h_emb,V_emb,mu,tau=0.01,maxm=1000,tol=1E-7,tmpdir=None,mpsdir=None,fiedler=True):
    '''
    Interacting bath requires non-trivial V_emb.
    An effective Hamiltonian is introduced to calculate the DMET energy.
    '''
    
    norb = h_emb.shape[-1]/2
    nimp = dmet.Nimp
    u    = dmet.g2e_site[0]
    #mu   = dmet.grandmu
    beta = 1./dmet.T

    # generate a random string for tmp dir 
    randstr = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(6)])
    
    if tmpdir == None:
        tmpdir = '/scratch/global/sunchong/dmet/finiteTMPS'+ randstr +'/'
        #tmpdir = '/home/sunchong/work/jobs/scratch/dmet/'+ randstr +'/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if mpsdir == None:
        mpsdir = '/home/sunchong/work/finiteTMPS/'
    
    hamfile   = tmpdir + 'hamfile.txt'
    ehamfile  = tmpdir + 'ehamfile.txt'
    intfile   = tmpdir + 'intfile.txt'
    eintfile  = tmpdir + 'eintfile.txt'
    impidfile = tmpdir + 'impidfile.txt'
    infile    = tmpdir + 'input_mps'

    # generate input file
    fin = open(infile, 'w')
    fin.write("input\n{\n")
    fin.write("hamfile = %s\n"%hamfile)
    fin.write("ehamfile = %s\n"%ehamfile)
    #fin.write("impsite = %s\n"%impidfile)
    fin.write("vfile = %s\n"%intfile)
    fin.write("evfile = %s\n"%eintfile)
    fin.write("outdir = %s\n"%tmpdir)
    fin.write("N = %d\n"%norb)
    fin.write("Nimp = %d\n"%nimp)
    #fin.write("U = %f\n"%u)
    fin.write("mu = %f\n"%mu)
    fin.write("beta = %f\n"%beta)
    fin.write("tau = %f\n"%tau)
    fin.write("maxm = %d\n"%maxm)
    fin.write("cutoff = %s\n"%(str(tol)))
    fin.write("realstep = yes\n")
    fin.write("fitmpo = yes\n")
    fin.write("rungekutta = yes\n")
    fin.write("verbose = no\n}")
    
    fin.close()

    # write 1 body hamiltonian
    h_embn = ftmodules.perm1el(h_emb, dmet.Nimp, tomix=False)
    
    # Fiedler reordering
    #np.set_printoptions(linewidth=1000)
    h1ea = h_embn[:norb, :norb]
    h1eb = h_embn[norb:,norb:]

    #--------------------write files-----------------------
    h1e_n = np.asarray([h1ea,h1eb]).reshape(2*norb,norb)
    np.savetxt(hamfile,h1e_n)

    Vab = V_emb[1].copy()
    Vab = Vab.reshape(norb**3,norb)
    np.savetxt(intfile, Vab)

    h1e_eff = get_h1e_eff(h1e_n,norb,nimp)
    np.savetxt(ehamfile, h1e_eff)
    
    Vab_eff = get_v2e_eff(V_emb[1],nimp)
    Vab_eff = Vab_eff.reshape(norb**3,norb)
    np.savetxt(eintfile, Vab_eff)
    #------------------------------------------------------


    #----------------------call FTMPS---------------------------
    subprocess.call([mpsdir + "impsolver_ancilla_ibath", infile])
    #-----------------------------------------------------------
    # read rdms
    e = np.loadtxt(tmpdir + "energy.txt")
    rdm1 = np.loadtxt(tmpdir + "rdm1s.txt")

    # convert formats of rdm1 and rdm2
    RDM1 = np.zeros_like(h_emb)
    RDM1[:norb,:norb] = rdm1[:norb, :]
    RDM1[norb:,norb:] = rdm1[norb:, :]
    RDM1 = ftmodules.perm1el(RDM1, nimp, tomix=True)
    
    RDM2 = np.zeros((norb,norb,norb,norb))
    
    # remove tmpdir
    shutil.rmtree(tmpdir)

    return RDM1, RDM2, e


#########################################################################
def get_h1e_eff(mat,norb,nimp):
    h1e_eff = mat.copy()
    h1e_eff = h1e_eff.reshape(2,norb,norb)
    for s in range(2):
        h1e_eff[s][nimp:,:nimp] *= 0.5
        h1e_eff[s][:nimp,nimp:] *= 0.5
        h1e_eff[s][nimp:,nimp:] *= 0.0

    return h1e_eff.reshape(norb*2,norb)
#########################################################################
def get_v2e_eff(mat,nimp):
    v2e = mat.copy()
    v2e[:nimp,:nimp,:nimp,nimp:] *= 0.75    
    v2e[:nimp,:nimp,nimp:,:nimp] *= 0.75
    v2e[:nimp,nimp:,:nimp,:nimp] *= 0.75
    v2e[nimp:,:nimp,:nimp,:nimp] *= 0.75

    v2e[:nimp,:nimp,nimp:,nimp:] *= 0.5
    v2e[:nimp,nimp:,:nimp,nimp:] *= 0.5
    v2e[:nimp,nimp:,nimp:,:nimp] *= 0.5
    v2e[nimp:,:nimp,:nimp,nimp:] *= 0.5
    v2e[nimp:,:nimp,nimp:,:nimp] *= 0.5
    v2e[nimp:,nimp:,:nimp,:nimp] *= 0.5

    v2e[nimp:,nimp:,nimp:,:nimp] *= 0.25
    v2e[nimp:,nimp:,:nimp,nimp:] *= 0.25
    v2e[nimp:,:nimp,nimp:,nimp:] *= 0.25
    v2e[:nimp,nimp:,nimp:,nimp:] *= 0.25

    v2e[nimp:,nimp:,nimp:,nimp:] *= 0.0

    return v2e

