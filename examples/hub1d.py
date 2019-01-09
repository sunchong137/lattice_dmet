
import sys
import os
sys.path.append('/home/sunchong/work/dmet_ur/ft_hubbard_v2')
import ftdmetTI as dmet
import numpy as np
import pyscf.fci.fci_slow as fci
import loggers.simpleLogger as log
import random as rnd

L = 202
nimp = 2
cmtype = 'UHF'

U = float(sys.argv[1])
T = float(sys.argv[3])
mu = float(sys.argv[2])
bathorder=1

#======================================================================
bdir = '%s/order%d'%(cmtype,bathorder)
chkdir = '%s/chkfiles'%cmtype
if not os.path.exists(bdir):
    os.makedirs(bdir)
if not os.path.exists(chkdir):
    os.makedirs(chkdir)

v = str(L) + str(bathorder) + '_U' + str(U) + '_mu' + str(mu) + "_T" + format(T,'1.2f')

sys.stdout = log.Logger(bdir + '/log' + v + '.sc.txt')
sys.stderr = log.Logger(bdir + '/elog' + v + '.sc.txt')

#======================================================================
print
print "#########################################################"
print "Starting 1D Hubbard Model"
print "#########################################################"
print 

factor = 1.0
nelec = int(L*factor+0.5)
mtype = np.float64

print "Interaction Strength (U/t):",U
print "No. of electrons:",nelec
print "No. of sites:",L
print 

#======================================================================

h1 = np.zeros([L,L])

for i in range(L):
    h1[i,(i+1)%L] = -1.0
    h1[i,(i-1)%L] = -1.0

g2e = np.zeros([2])
g2e[0] = U

AFMorder = 0.0
u0 = U/2.
if(nimp == 1):
    umg = np.diag([0.2,-0.2])
else:
    hn = nimp/2
        
    #All symmetries enforced
    if(cmtype == 'RHF'):
        umg = np.diag([0.0,0.0]*hn + [0.0,0.0]*hn)
    else:
        if T < 1.e-3:
            u0 = 0.
        a = [u0+AFMorder, u0-AFMorder]*hn
        b = [u0-AFMorder, u0+AFMorder]*hn
        #b = [x for x in a] 
        umg = np.diag(a + b)
    
    ##Put in superconducting pairing
    #vcorr = np.zeros([nimp,nimp])
    #for x in range(nimp):
    #    vcorr[x,nimp-x-1] = 0.0e-4*rnd.random()
    #umg[:nimp,nimp:] = vcorr
    #umg[nimp:,:nimp] = vcorr.conjugate().T

#umg = np.zeros([nimp*2, nimp*2])

#======================================================================
obj = dmet.ftdmet(L,nelec,nimp,h1,g2e, mtype=mtype, u_matrix=umg,\
                  ctype=cmtype,T=T,grandMu=mu,BathOrder=bathorder)

obj.diisStart = 4 
obj.diisDim = 4

obj.dlog = log.Logger(chkdir + '/rdmlog' + v + '.txt') 

obj.chkPointInterval = 1
obj.chkPointFile = chkdir + '/chkptsoc' + v + '.sc.hdf5'
obj.resFile = chkdir + '/chkptsoc' + v + '.sc.hdf5'

obj.doRestart = False
obj.fitBath = False
obj.iformalism = False

obj.muSolverSimple = False
obj.dmetitrmax = 100
obj.tableFile = chkdir + '/table' + v + '.txt'
 
#--------------------------------------------------------
obj.solve_groundstate()
