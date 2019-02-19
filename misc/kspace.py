import numpy as np
import sys
import random
import scipy.linalg as la
sys.path.append('../')
import utils
import time

def cartesian(arrays, out=None):
    """
    Thanks to Pauli from stackoverflow for this solution:
    Generate a cartesian product of input arrays.
    
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def switchNotation(H):
    # Do |s,i> to |i,s>
    Hb = np.zeros_like(H)
    L = Hb.shape[0]/2
    Lf = Hb.shape[0]

    Hb[:Lf:2,:Lf:2] = H[:L,:L]
    Hb[:Lf:2,1:Lf:2] = H[:L,L:]
    Hb[1:Lf:2,:Lf:2] = H[L:,:L]
    Hb[1:Lf:2,1:Lf:2] = H[L:,L:]

    return Hb

def extractStripe(h1,N,I):
    Nt = np.prod(N)
    It = np.prod(I)
    hstripe = np.zeros([Nt,It,It],dtype=h1.dtype)
    for i in range(Nt):
        hstripe[i] = h1[:It,i*It:(i+1)*It]
    return hstripe

def createKgrid(N,m):
    
    Nt = np.prod(N)
    mt = np.prod(m)

    dpi = 2.0*np.pi
    #kval = cartesian((-np.pi + dpi/N[0]*np.array(range(N[0])),-np.pi + dpi/N[1]*np.array(range(N[1])))) 
    kval = cartesian((dpi/N[1]*np.array(range(N[1])), dpi/N[0]*np.array(range(N[0])))) 
    rval = cartesian((range(N[1]),range(N[0])))

    kgrid = np.zeros([Nt,Nt,mt,mt],dtype=np.complex128)
    for (i,k) in enumerate(kval):
        for (a,r) in enumerate(rval):
            kgrid[i][a] = np.ones([mt,mt])*np.exp(1j*np.dot(k,r))

    return kgrid

def transform2d(hstripe,N,m, kgrid = None):
    '''
        hstripe = [N][mxm]
        N = no. of unit cells
        m = no. of points per unit cell
        
        kgrid = [N][N][mxm] needed to perform the FT.
        [0] = k values
        [1] = R values
        [2-3] = sites x sites
    '''

    Nt = np.prod(N)
    mt = np.prod(m)
 
    if(kgrid is None):
        dpi = 2.0*np.pi
        #kval = cartesian((-np.pi + dpi/N[0]*np.array(range(N[0])),-np.pi + dpi/N[1]*np.array(range(N[1])))) 
        kval = cartesian((dpi/N[1]*np.array(range(N[1])), dpi/N[0]*np.array(range(N[0])))) 
        rval = cartesian((range(N[1]),range(N[0])))

        kgrid = np.zeros([Nt,Nt,mt,mt],dtype=np.complex128)
        for (i,k) in enumerate(kval):
            for (a,r) in enumerate(rval):
                kgrid[i][a] = np.ones([mt,mt])*np.exp(1j*np.dot(k,r))

    Hk = [(hstripe*kgrid[i]).sum(0) for i in range(Nt)]
    
    return Hk, kgrid

def diagonalize(hfull,norb,N,I,kgrid = None):
    
    #tstart1 = time.time()
    Hk, kgrid = createHk(hfull,N,I)
    #tend1 = time.time()
    #print 'T: ',tend1-tstart1

    tstart2 = time.time()
    Es,Cs = diagonalizeHk(Hk,N,I,kgrid) 
    tend2 = time.time()
    rdm = np.dot(Cs[:,:norb],Cs[:,:norb].conj().T)

    '''
    print 'First terms:'
    aa = Cs[:,:norb].conj().T
    summ = 0.0
    for i,x in enumerate(Cs[0,:norb]):
        summ = summ + x*aa[i,0] 
        print x,aa[i,0],summ
    print 'xxxxx'
    '''

    return Es,Cs,rdm,kgrid,tend2-tstart2

def createHk(h1,N,I,kgrid = None):
    '''
        N = no. of clusters
        I = no. of impurity sites
    '''
    Nt = np.prod(N)
    It = np.prod(I)
    norb = Nt*It

    h1u = h1[:norb,:norb]
    h1d = h1[norb:,norb:]
    h1ud = h1[:norb,norb:]

    h1ustripe = extractStripe(h1u,N,I)
    h1dstripe = extractStripe(h1d,N,I)
    h1udstripe = extractStripe(h1ud,N,I)

    '''
    for i,hk in enumerate(h1ustripe):
        utils.displayMatrix(h1ustripe[i])
        print
        utils.displayMatrix(h1dstripe[i])
        print
        utils.displayMatrix(h1udstripe[i])
        print
    '''

    hku, kgrid = transform2d(h1ustripe,N,I,kgrid)
    hkd, kgrid = transform2d(h1dstripe,N,I,kgrid)
    hkud, kgrid = transform2d(h1udstripe,N,I,kgrid)

    '''
    for i,hk in enumerate(hku):
        utils.displayMatrix(hku[i])
        print
    '''

    block = np.zeros([Nt,2*It,2*It],dtype=hku[0].dtype)
    for i,hk in enumerate(hku):
        u = hk
        d = hkd[i]
        ud = hkud[i]

        block[i][:It,:It] = u
        block[i][:It,It:] = ud
        block[i][It:,:It] = ud.T  
        block[i][It:,It:] = d

    return block, kgrid

def diagonalizeHk(Hk,N,I,kgrid):

    Nt = np.prod(N)
    It = np.prod(I)
    norb = Nt*It
    dnimp = 2*It

    '''
    kgrid was k-values x r-values i.e. e^{ikr}|k,u><r,v|
    kgridT becomes e^{-ikr} |r,v><k,u|
    '''
    kgridT = kgrid.conj().swapaxes(2,3)
    kgridT = kgridT.conj().swapaxes(0,1)
    #print 'kgT: ',kgridT.shape 

    Es = np.zeros([2*norb])
    Cr = np.zeros([2*norb,2*norb],dtype=np.complex128) #Convert to r-space
    
    dds = [2] + N + [It,2*norb]
    Ck = np.zeros(dds,dtype=np.complex128)
    for i in range(Nt):
        e,v = la.eigh(Hk[i])  
        Es[i*dnimp:(i+1)*dnimp] = e
        #C[i*It:(i+1)*It,i*dnimp:(i+1)*dnimp] = v[:It,:].real
        #C[(norb+i*It):(norb+(i+1)*It),i*dnimp:(i+1)*dnimp] = v[It:,:].real
 
        '''   
        print
        print i
        utils.displayMatrix(Hk[i])

        print
        utils.displayMatrix(v)
        print
        ''' 
        
        ''' 
        print 
        for k in range(Nt):
            print i," ",k
            utils.displayMatrix(kgridT[k,i]*v[:It,:It])
            print
            utils.displayMatrix(kgridT[k,i]*v[:It,It:])
            print
            utils.displayMatrix(kgridT[k,i]*v[It:,:It])
            print
            utils.displayMatrix(kgridT[k,i]*v[It:,It:])
        '''
        
        #spin-up projection
        Cr[:norb,i*dnimp:i*dnimp+It] += (kgridT[:,i]*v[:It,:It]).reshape([norb,It])
        Cr[norb:,i*dnimp:i*dnimp+It] += (kgridT[:,i]*v[It:,:It]).reshape([norb,It])
        
        #spin-dn projection
        Cr[:norb,i*dnimp+It:(i+1)*dnimp] += (kgridT[:,i]*v[:It,It:]).reshape([norb,It])
        Cr[norb:,i*dnimp+It:(i+1)*dnimp] += (kgridT[:,i]*v[It:,It:]).reshape([norb,It])

    idx = np.argsort(Es)
    Es = Es[idx]
    C = Cr[:,idx]/(Nt**0.5)

    '''
    print
    utils.displayMatrix(C)
    print
    '''

    return Es,C


if __name__ == '__main__':

    #2d Hubbard Model
    Nx = 10 
    Ny = 10
    N = [Nx,Ny]
 
    Ix = 2
    Iy = 2
    I = [Ix,Iy]
   
    U = 1.0
    factor = 1.0 
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
        
                    h1[site,site] = 0.0 #-0.588452576493                    
                
                    if(site!=siteu):     
                        h1[site,siteu] = -1.0
                
                    if(site!=siter):
                        h1[site,siter] = -1.0
                    
                    if(site!=sited):
                        h1[site,sited] = -1.0
                    
                    if(site!=sitel):
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
    
        basev = 0.75
        for y in range(Iy):
            for x in range(Ix):
                site = Ix*y+x
                um[site,site] = basev*uf*factor
                um[site+nimp,site+nimp] = (basev-1.0)*uf*factor
                basev = 1.0 - basev
    
    if(True):
        random.seed(10)
        vcorr = np.zeros([Ix*Iy,Ix*Iy]) 
        for i in range(Ix*Iy):
            for j in range(Ix*Iy):
                vcorr[i,j] = random.random()*1.0e-4
        
        um[:Ix*Iy,Ix*Iy:] = vcorr
        um[Ix*Iy:,:Ix*Iy] = vcorr.conjugate().T

    def replicate_u_matrix(u_imp,Nbasis,Nimp):

        #Number of copies of impurity cluster in total lattice
        Ncopies = Nbasis / Nimp

        #Copy impurity u-matrix across entire lattice
        u_mat_replicate = np.zeros( [2*Nbasis,2*Nbasis],dtype=u_imp.dtype)
        for cpy in range(0,Ncopies):
            u_mat_replicate[cpy*Nimp:Nimp+cpy*Nimp,cpy*Nimp:Nimp+cpy*Nimp] = u_imp[:Nimp,:Nimp]
            u_mat_replicate[cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis,cpy*Nimp:Nimp+cpy*Nimp] = u_imp[Nimp:Nimp+Nimp,:Nimp]
            u_mat_replicate[cpy*Nimp:Nimp+cpy*Nimp,cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis] = u_imp[:Nimp,Nimp:Nimp+Nimp]
            u_mat_replicate[cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis,cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis] = u_imp[Nimp:Nimp+Nimp,Nimp:Nimp+Nimp]
            
        return u_mat_replicate
    

    def loadfile(fn):
        with open(fn) as f:
            ll = f.read()
        x = ll.split('\n')
        xl = len(x)-1
        yl = 0
        for l in x[0].split(' '):
            try:
                float(l)
                yl+=1
            except ValueError:
                pass

        u0 = np.zeros([xl,yl])
        for i,l in enumerate(x):
            j = 0
            for v in x[i].split(' '):
                try:
                    vv = float(v)
                    u0[i,j] = vv
                    j += 1
                except ValueError:
                    pass

        return u0 
    #um = loadfile('u0.dat')
    
    hfull = np.zeros([2*norb,2*norb])
    hfull[:norb,:norb] = h1
    hfull[norb:,norb:] = -h1
    vfull = replicate_u_matrix(um,norb,nimp)
    hfull += vfull
    tstart = time.time()
    Ef,Cf = la.eigh(hfull)
    tend = time.time()
    print 'T: ',tend-tstart
    rdmf = np.dot(Cf[:,:norb],Cf[:,:norb].T)

    '''
    print
    utils.displayMatrix(rdmf)
    print
    '''

    Hk, kgrid = createHk(hfull,N,I)
    tstart = time.time()
    Es,Cs = diagonalizeHk(Hk,N,I,kgrid) 
    tend = time.time()
    print 'k T:',tend-tstart
    rdm = np.dot(Cs[:,:norb],Cs[:,:norb].conj().T)
    '''
    print
    utils.displayMatrix(rdm)
    print
    '''
    rawdiff = la.norm(rdm-rdmf)
    print 'rdm diff: ',rawdiff,rawdiff/np.prod(rdm.shape)
    print 'degen check: ',Es[norb-1],Es[norb]
