import sys
sys.path.append('../')
import dmetTI as dmet
import numpy as np
import pyscf.fci.fci_slow as fci
import utils 
import numpy.linalg as la
import loggers.simpleLogger as log
import random


U = float(sys.argv[1]) 
#lam = float(sys.argv[2])
bdir = sys.argv[2] 
factor = float(sys.argv[3]) #0.8 
cmtype = 'BCS'

#------------------------------------------------------------------------------------

v = str(U).replace('.','_')
sys.stdout = log.Logger(bdir + '/log' + v + '.txt')
sys.stderr = log.Logger(bdir + '/elog' + v + '.txt')

#------------------------------------------------------------------------------------
print
print "#########################################################"
print "Starting 2D Hubbard Model"
print "#########################################################"
print 

#2d Hubbard Model
Nx = 15 
Ny = 15
Ix = 2
Iy = 2

#--------------------------------------------------------
Lx = Nx*Ix
Ly = Ny*Iy

norb = Lx*Ly
nimp = Ix*Iy

mtype = np.float64
#mtype = np.complex128
h1 = np.zeros([norb,norb],dtype=mtype)
#h1soc = np.zeros([norb,norb],dtype=mtype)
#g2e = np.zeros([norb,norb,norb,norb])

for j in range(Ny):
    for i in range(Nx):
        
        for y in range(Iy):
            for x in range(Ix):

                site = (Nx*j + i)*nimp + Ix*y + x
        
                if(y+1>=Iy):
                    nny = (j+1)%Ny
                    cy = 0
                else:
                    nny = j
                    cy = y+1

                if(x+1>=Ix):
                    nnx = (i+1)%Nx
                    cx = 0
                else:
                    nnx = i
                    cx = x+1

                if(x-1<0):
                    bnx = (i-1)%Nx
                    bx = Ix-1
                else:
                    bnx = i
                    bx = x-1

                if(y-1<0):
                    bny = (j-1)%Ny
                    by = Iy-1
                else:
                    bny = j
                    by = y-1

                siteu = (Nx*nny + i)*nimp + Ix*cy + x
                siter = (Nx*j + nnx)*nimp + Ix*y + cx
                sited = (Nx*bny + i)*nimp + Ix*by + x
                sitel = (Nx*j + bnx)*nimp + Ix*y + bx               

#               print i,j,x,y,site,siteu,siter,sited,sitel              
                         
                h1[site,siteu] = -1.0
                h1[site,siter] = -1.0
                h1[site,sited] = -1.0
                h1[site,sitel] = -1.0
                
                '''
                h1soc[site,siteu] = lam*1j
                h1soc[site,siter] = lam*1j
                h1soc[site,sited] = lam*1j
                h1soc[site,sitel] = lam*1j
                '''
#               g2e[site,site,site,site] = 4.0


#Always check this for h1
assert(np.sum(np.matrix(h1) - np.matrix(h1).H)<=1.0e-16)
#--------------------------------------------------------
#gse,c = fci.kernel(h1,g2e,norb,norb)
#print "FCI energy per site: ",gse/norb
#--------------------------------------------------------

um = np.zeros([2*Ix*Iy,2*Ix*Iy])
if(True):
    hn = nimp/2
    #um = np.diag([0.1,-0.1]*hn + [-0.1,0.1]*hn)
    #um = np.diag([0.001,-0.001,-0.001,0.001] + [0.001,-0.001,-0.001,0.001])
    #um = np.diag([0.1,-0.1,-0.1,0.1] + [0.1,-0.1,-0.1,0.1])
    #um = np.diag([0.1,-0.1,-0.1,0.1] + [-0.1,0.1,0.1,-0.1])
    #um = np.diag([0.06525,0.02475,0.02475,0.06525] + [-0.02475,-0.06525,-0.06525,-0.02475])*U*10.0 
    if(U == 0.0):
        uf = 1.0e-02
    else:
        uf = U

    if(cmtype == 'BCS'):    
    	um = np.diag([0.75,0.25,0.25,0.75] + [-0.25,-0.75,-0.75,-0.25])*uf*factor
    elif(cmtype == 'UHF' or cmtype == 'SOC'):
    	um = np.diag([0.75,0.25,0.25,0.75] + [0.25,0.75,0.75,0.25])*uf*factor
    else:
    	um = np.diag([0.75,0.25,0.25,0.75] + [0.75,0.25,0.25,0.75])*uf*factor

    odg = np.zeros([Ix*Iy,Ix*Iy])
    for i in range(Ix*Iy):
        for j in range(i+1,Ix*Iy):
            odg[i,j] = random.random()*0.0e-1
    um[:Ix*Iy,:Ix*Iy] += odg + odg.conjugate().T
    um[Ix*Iy:,Ix*Iy:] += odg + odg.conjugate().T

    if(cmtype == 'BCS'):
        vcorr = np.zeros([Ix*Iy,Ix*Iy]) 
        for i in range(Ix*Iy):
            for j in range(Ix*Iy):
                vcorr[i,j] = random.random()*1.0e-3
        
        um[:Ix*Iy,Ix*Iy:] = vcorr
        um[Ix*Iy:,:Ix*Iy] = vcorr.conjugate().T

#evals,eorbs = la.eigh(h1)
#print evals
#sys.exit()

nelec = int(norb*factor)  
g2e = np.zeros([2])
g2e[0] = U 
g2e[1] = 0.0

#--------------------------------------------------------

print "Interaction Strength (U/t):",U
print "No. of electrons:",nelec
print "No. of sites:",norb
print


#obj = dmet.dmet(norb,nelec,nimp,h1,h1soc,g2e, mtype = mtype)
obj = dmet.dmet(norb,nelec,nimp,h1,g2e, mtype = mtype, u_matrix = um,ctype=cmtype)

obj.diisStart = 4 
obj.diisDim = 4

obj.dlog = log.Logger(bdir + '/rdmlog' + v + '.txt') 

obj.chkPointInterval=1
obj.chkPointFile= bdir + '/hub2d_chkpt_' + v + '.hdf5'
obj.resFile = bdir + '/hub2d_chkpt_' + v + '.hdf5'

obj.doRestart = True 
obj.fitBath = False 
obj.iformalism = True 

obj.muSolverSimple = True 
obj.dmetitrmax = 20
obj.corrIter = 50
obj.startMuDefault = True
obj.startMu = 2.0 

obj.tableFile = bdir + '/table' + v + '.txt'

#--------------------------------------------------------
obj.solve_groundstate()
