import numpy as np
import h5py

def loadfile(fname):

    f = h5py.File(fname,'r')

    def readMatrix(basename):
        s = f.get(basename+'.s')[:]
        l = np.prod(s)
        try:
            mmr = f.get(basename + '.r')[:]
            mmi = f.get(basename + '.i')[:]
            mm = np.array(map(lambda (a,b): a+1j*b,zip(mmr,mmi)))
        except TypeError:
            mm = np.array(f.get(basename + '.r')[:]) 
        mm = np.reshape(mm,s)
        return mm

#    Efrag = f.get('observables')[1]
    IRDM1 = readMatrix('rdm1')
    f.close()   
    return IRDM1

def oparam(irdm1):

    rdm_ab = irdm1[:4,4:8]
    rdm_a = irdm1[:4,:4]
    rdm_b = np.eye(4) - irdm1[4:8,4:8]

    m0 = 0.5*(rdm_a[0,0]-rdm_b[0,0])
    m3 = 0.5*(rdm_a[3,3]-rdm_b[3,3])
    m1 = 0.5*(rdm_a[1,1]-rdm_b[1,1])
    m2 = 0.5*(rdm_a[2,2]-rdm_b[2,2])
    afm = 0.25*(m0+m3-m1-m2)

    s = 0.5**0.5
    d01 = s*(rdm_ab[0,1]+rdm_ab[1,0])
    d23 = s*(rdm_ab[2,3]+rdm_ab[3,2])
    d02 = s*(rdm_ab[0,2]+rdm_ab[2,0])
    d13 = s*(rdm_ab[1,3]+rdm_ab[3,1])
    dwv = 0.25*(d01+d23-d02-d13)

    return afm,dwv

if __name__ == '__main__':

    import sys
    fname = sys.argv[1]

    irdm1 = loadfile(fname)
    afm, dwv = oparam(irdm1)

    print 'AFM: %8.6e\t DWV: %8.6e\n' %(afm,dwv)
