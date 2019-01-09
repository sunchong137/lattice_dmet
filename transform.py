import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import utils

def transformTI(H,m,K = None):

    '''
    Expect H in |s,i> notation where s is spin
    This routine will convert it into |k,s> notation
    m is the number of sites in the unit cell
    Expects i to be arranged so that all sites in the unit cell are contiguous
    If K is given then use it otherwise construct it
    '''

    'First do |s,i> to |i,s> because it is easier do this first'

    Hb = np.zeros_like(H)
    L = Hb.shape[0]/2
    Lf = Hb.shape[0]

    Hb[:Lf:2,:Lf:2] = H[:L,:L]
    Hb[:Lf:2,1:Lf:2] = H[:L,L:]
    Hb[1:Lf:2,:Lf:2] = H[L:,:L]
    Hb[1:Lf:2,1:Lf:2] = H[L:,L:]
    
    ms = m*2 #2 spins x number of sites in unit cell
    N = Lf/ms

    if (K is None):    
    	K = np.zeros_like(Hb,dtype=np.complex128)
    	kblock = np.zeros([ms,N*ms],dtype=np.complex128)
    	div = np.zeros([ms,N*ms],dtype=np.complex128)
    	for i in range(N):
        	kblock[:,i*ms:i*ms+ms] = np.eye(ms)*2.*np.pi*i/N 
        	div[:,i*ms:i*ms+ms] = np.eye(ms)*(1.0*ms/Lf)**0.5  
#    utils.displayMatrix(kblock)
#    print "@@@@"
    	for x in range(N):       
        	K[x*ms:x*ms+ms,:] = np.multiply(np.exp(1.j*kblock*x),div) 
#    utils.displayMatrix(K)
#    print "####"
#    utils.displayMatrix(np.dot(K.conjugate().T,K))
#    print "xxxx"
 
    Hk = np.dot(K.conjugate().T,np.dot(Hb,K))
    kvals = np.array([2.*np.pi*n/N for n in range(N)])
    return Hk,K,kvals

def spectrum(Hk,m,K):
    '''
    Expects Hk in block diagonal form and returns spectrum and vectors
    |k,s>
    '''
    L = Hk.shape[0]
    ms = 2*m
    V = K.copy()
    spec = np.zeros([ms,L/ms])
    for i in range(0,L,ms):
        hb = Hk[i:i+ms,i:i+ms]
        e,val = la.eigh(hb)
        
        vf = np.asarray(list(val.flatten())*(L/ms)).reshape(L,ms)
        V[:,i:i+ms] = np.multiply(V[:,i:i+ms],vf)       

        spec[:,i/ms] = e

    return spec,V

def plotSpectrum(k,spec,nelec,nbsp, svf = 'save.png'):
   
    symboll = ['*','s','p','o','1','2','3','4','h','H','+']
    colorl = ['red','green','blue','cyan','navy','teal','maroon','gray','olive','purple']
 
    spf = spec.flatten()
    fs = np.argsort(spf)
    pos = np.array(fs[:nelec])

    print 'LUMO-HOMO: ',spf[fs[nelec]] - spf[fs[nelec-1]]

    for i in range(2*nbsp):
        plt.plot(k,spec[i,:],'-' + symboll[i],color = colorl[i])
   
    ll = len(k) 
    fig = plt.plot(k[pos%ll],spf[pos],'sk',markersize=10, fillstyle='none')
    plt.savefig(svf)
#   plt.show()

def makePlot(H, ucells, nelec, svf='save.png'):
    #H = Hamiltonian
    #Size of unit cell
    Hk,Km,k = transformTI(H,ucells)    
    spec,V = spectrum(Hk,ucells,Km)
    plotSpectrum(k,spec,nelec,ucells,svf) 

    print "Spectrum from block diagonalization: ",spec

if __name__ == "__main__":


    #Setup some dummy SOC Hamiltonian in |s,i> notation
    L = 6 
    lam = 0.2 
    nelec = int(L*1.+.5)
    nbsp = 1 #size of unit cell

    H = np.zeros([2*L,2*L],dtype=np.complex128)
    T = np.zeros([L,L],dtype=np.complex128)
    Hsoc = np.zeros([L,L],dtype=np.complex128)
    for i in range(L):
        T[i,(i+1)%L] = -1.0
        T[i,(i-1)%L] = -1.0

        Hsoc[i,(i+1)%L] = 1.j*lam
        Hsoc[i,(i-1)%L] = -1.j*lam
    Hsoc[0,L-1] = -1.j*lam
    Hsoc[L-1,0] = 1.j*lam
            
    H[:L,:L] = T
    H[L:,L:] = T
    H[:L,L:] = Hsoc
    H[L:,:L] = Hsoc.conjugate().T
    
    E,V = la.eigh(H)
    print "From direct diagonalization: ",E
 
    Hk,Km,k = transformTI(H,nbsp)    
    spec,V = spectrum(Hk,nbsp,Km)

    plotSpectrum(k,spec,nelec,nbsp) 
    print "Spectrum: ",spec        
    print "Difference in Spectra: ",la.norm(np.sort(spec.flatten())-E)
          
