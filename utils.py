
import numpy as np
import scipy.linalg as la
import scipy.special
import pyscf.fci
import math
#####################################################################

def diagonalize(H,S=None):
    #subroutine to solve the general eigenvalue problem HC=SCE
    #returns the matrix of eigenvectors C, and a 1-d array of eigenvalues
    #NOTE that H must be Hermitian

    if S is None:
    	E,C = la.eigh(H)
    else:
    	E,C = la.eigh(H,S)

    return E,C

#####################################################################

def rot1el( h_orig, rotmat ):
    #subroutine to rotate one electron integrals

    tmp = np.dot( h_orig, rotmat )
    if( np.iscomplexobj(rotmat) ):
        h_rot = np.dot( rotmat.conjugate().transpose(), tmp )
    else:
        h_rot = np.dot( rotmat.transpose(), tmp )

    return h_rot

#####################################################################
def rot2el_chem_full(V_orig, rotmat):
    #subroutine to rotate two electron integrals, V_orig must be in chemist notation
    # [ij|kl] = |i>|k> <j|<l|
       
    #V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns
    if( np.iscomplexobj(rotmat) ):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()
    
    V_new = np.einsum( 'trus,sy -> truy', V_orig, rotmat )
    #V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum( 'vu,truy -> trvy', rotmat_conj, V_new )
    #V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum( 'trvy,rx -> txvy', V_new, rotmat )
    #V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum( 'wt,txvy -> wxvy', rotmat_conj, V_new )
    #V_new now Ns x Ns x Ns x Ns

    return V_new

def rot2el_phys_full(V_orig, rotmat):
    #subroutine to rotate two electron integrals, Po must be in physics notation
    # <ij|kl> P(i,j,k,l)
       
    if( np.iscomplexobj(rotmat) ):
       rotmat_conj = rotmat.conjugate().transpose()
    else:
       rotmat_conj = rotmat.transpose()

    V_new = np.einsum( 'turs,sy -> tury', V_orig, rotmat )
    #V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum( 'tury,rx -> tuxy', V_new, rotmat )
    #V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum( 'vu,tuxy -> tvxy', rotmat_conj, V_new )
    #V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum( 'wt,tvxy -> wvxy', rotmat_conj, V_new )
    #V_new now Ns x Ns x Ns x Ns

    return V_new

'''
def rot2el_chem( V_orig, rotmat ):
    #subroutine to rotate two electron integrals, V_orig must be in chemist notation
    # [ij|kl] = |i>|k> <j|<l|
       
    #V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns
    if( np.iscomplexobj(rotmat) ):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    nbasis = rotmat.shape[0]/2
    
    V_new = np.einsum( 'ijkl,lp -> ijkp', V_orig, rotmat[:nbasis,:] ) + np.einsum('ijkl,lp->ijkp', V_orig,rotmat[nbasis:2*nbasis,:])   
    #V_new now Nb x Nb x Nb x Ns
    
    V_new = np.einsum( 'pk,ijkq -> ijpq', rotmat_conj[:,:nbasis], V_new ) + np.einsum( 'pk,ijkq -> ijpq', rotmat_conj[:,nbasis:2*nbasis], V_new )
    #V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum( 'ijqr,jp -> ipqr', V_new, rotmat[:nbasis,:] ) + np.einsum( 'ijqr,jp -> ipqr', V_new, rotmat[nbasis:2*nbasis,:] ) 
    #V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum( 'pi,iqrs -> pqrs', rotmat_conj[:,:nbasis], V_new ) + np.einsum( 'pi,iqrs -> pqrs', rotmat_conj[:,nbasis:2*nbasis], V_new )
    #V_new now Ns x Ns x Ns x Ns

    return V_new
'''
#####################################################################

def commutator( Mat1, Mat2 ):
    #subroutine to calculate the commutator of two matrices

    return np.dot(Mat1,Mat2) - np.dot(Mat2,Mat1)

#####################################################################

def chemps2_to_pyscf_CIcoeffs( CIcoeffs_chemps2, Norbs, Nalpha, Nbeta ):
    #subroutine to unpack the 1d vector of CI coefficients obtained from a FCI calculation using CheMPS2
    #to the correctly formatted 2d-array of CI coefficients for use with pyscf

    Nalpha_string = scipy.special.binom( Norbs, Nalpha )
    Nbeta_string = scipy.special.binom( Norbs, Nbeta )

    CIcoeffs_pyscf = np.reshape( CIcoeffs_chemps2, (Nalpha_string, Nbeta_string), order='F' )

    return CIcoeffs_pyscf

#####################################################################

def matrix2array( mat, diag=False ):

    #Subroutine to flatten a symmetric matrix into a 1d array
    #Returns a 1d array corresponding to the upper triangle of the symmetric matrix
    #if diag=True, all diagonal elements of the matrix should be the same
    #and first index of 1d array will be the diagonal term, and the rest the upper triagonal of the matrix

    if( diag ):
        array = mat[ np.triu_indices( len(mat),1 ) ] 
        array = np.insert(array, 0, mat[0,0])
    else:
        array = mat[ np.triu_indices( len(mat) ) ] 

    return array

#####################################################################

def array2matrix( array, diag=False ):

    #Subroutine to unpack a 1d array into a symmetric matrix
    #Returns a symmetric matrix
    #if diag=True, all diagonal elements of the returned matrix will be the same corresponding to the first element of the 1d array

    if( diag ):
        dim = (1.0+np.sqrt(1-8*( 1-len(array) )))/2.0
    else:
        dim = (-1.0+np.sqrt(1+8*len(array)))/2.0

    mat = np.zeros( [dim,dim] )

    if( diag ):
        mat[ np.triu_indices(dim,1) ] = array[1:]
        np.fill_diagonal( mat, array[0] )
    else:
        mat[ np.triu_indices(dim) ] = array

    mat = mat + mat.transpose() - np.diag(np.diag(mat))

    return mat 

#####################################################################

def matrix2array_nosym( mat, diag=False ):

    #Subroutine to flatten a general matrix into a 1d array
    #Returns a 1d array corresponding to the upper triangle of the symmetric matrix
    #if diag=True, all diagonal elements of the matrix should be the same
    #and first index of 1d array will be the diagonal term, and the rest the upper triagonal of the matrix

    if( diag ):
        array = mat[ np.triu_indices( len(mat),1 ) ] 
        array = np.insert(array, 0, mat[0,0])
    else:
        array = mat[ np.triu_indices( len(mat) ) ] 

    return array

#####################################################################

def printarray( array, filename='array.dat', long_fmt=False ):
    #subroutine to print out an ndarry of 2,3 or 4 dimensions to be read by humans

    dim = len(array.shape)

    filehandle = file(filename,'w')
    filehandle = file(filename,'a')

    comp_log = np.iscomplexobj( array )

    if( comp_log ):

        if( long_fmt ):
            fmt_str = '%20.8e%+.8ej'
        else:
            fmt_str = '%10.4f%+.4fj'

    else:

        if( long_fmt ):
            fmt_str = '%20.8e'
        else:
            fmt_str = '%8.4f'


    if ( dim == 1 ):

        Ncol = 1
        np.savetxt(filehandle, array, fmt_str*Ncol )

    elif ( dim == 2 ):

        Ncol = array.shape[1]
        np.savetxt(filehandle, array, fmt_str*Ncol )

    elif ( dim == 3 ):

        for dataslice in array:
            Ncol = dataslice.shape[1]
            np.savetxt(filehandle, dataslice, fmt_str*Ncol )
            filehandle.write('\n')

    elif ( dim == 4 ):

        for i in range( array.shape[0] ):
            for dataslice in array[i,:,:,:]:
                Ncol = dataslice.shape[1]
                np.savetxt(filehandle, dataslice, fmt_str*Ncol )
                filehandle.write('\n')
            filehandle.write('\n')

    else:
        print 'ERROR: Input array for printing is not of dimension 2, 3, or 4'
        exit()

#####################################################################

def readarray( filename='array.dat' ):
    #subroutine to read in arrays generated by the printarray subroutine defined above
    #currently only works with 1d or 2d arrays

    #num_lines = sum(1 for line in open(filename))

    #with file(filename) as f:
    #    line = f.readline()
    #num_col = len( line.split())

    array = np.loadtxt( filename, dtype = np.complex128 )

    chk_cmplx = np.any( np.iscomplex( array ) )

    if( not chk_cmplx ):
        array = np.copy( np.real( array ) )

    return array

#####################################################################

def extractImp(nimp,m):
    
    #Extract impurtiy parts from a matrix organized with different spin sectors
    #Recall that the organization is [[0 1] [1 0]]
    nbasis = m.shape[0]/2    
    out = np.zeros([2*nimp,2*nimp],dtype=m.dtype)
    
    out[:nimp,:nimp] = m[:nimp,:nimp]
    out[nimp:nimp+nimp,:nimp] = m[nbasis:nbasis+nimp,:nimp] 
    out[:nimp,nimp:nimp+nimp] = m[:nimp,nbasis:nbasis+nimp]
    out[nimp:nimp+nimp,nimp:nimp+nimp] = m[nbasis:nbasis+nimp,nbasis:nbasis+nimp]
    
    return out

def displayMatrix(m,fieldWidth=6,precision=6,fmt="e"):
    
    s = np.shape(m)
     
    ff = "% " + str(fieldWidth) + "." + str(precision) + fmt
    tf = ff + " +" + ff + "\t"
    sf = ff + "\t"
    for i in range(s[0]):
        for j in range(s[1]):
            if(np.iscomplex(m[i][j])):
                a = m[i][j]
                print  tf %(a.real,a.imag),
            else:
                print sf %(m[i][j]),
        print 

def analyticGradient(H, dH, nocc):
    
    #Provide the bare Hamiltonian H
    #and the number of occupied orbitals
    
     L = H.shape[1]
     
     E,C = la.eigh(H)
     
     Cocc = C[:,:nocc]
     Cvir = C[:,nocc:]
     Eocc = np.array([E[:nocc],]*(L-nocc))
     Evir = np.array([E[nocc:],]*nocc).T
     
     Zm = -np.divide(np.dot(Cvir.conjugate().T,np.dot(dH,Cocc)),(Evir-Eocc))
     
     Cmocc = np.dot(Cvir,Zm)
     result = np.dot(Cocc,Cmocc.conjugate().T) + np.dot(Cmocc,Cocc.conjugate().T)
     
     return result
     
def analyticGradientO(C,E, dH, nocc):
    
    #Provide the orbitals and energies of the bare hamiltonian 
    #and the number of occupied orbitals
    #print 'Energies'
    #print E
    #print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'

     L = dH.shape[1]
     
     Cocc = C[:,:nocc]
     Cvir = C[:,nocc:]
     Eocc = np.array([E[:nocc],]*(L-nocc))
     Evir = np.array([E[nocc:],]*nocc).T
     
     Zm = -np.divide(np.dot(Cvir.conjugate().T,np.dot(dH,Cocc)),(Evir-Eocc))
 
     Cmocc = np.dot(Cvir,Zm)
     result = np.dot(Cocc,Cmocc.conjugate().T) + np.dot(Cmocc,Cocc.conjugate().T)
     
     return result

def analyticGradientObcs(C,E, dH, nocc):
    
    #Provide the orbitals and energies of the bare hamiltonian 
    #and the number of occupied orbitals
    #print 'Energies'
    #print E
    #print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'

     L = dH.shape[1]
     
     #Check for degeneracies
     etol=1.0e-10
     idx = abs(E[1:]-E[:-1])<etol
     idx = np.append(idx,False)

     edeg = E[idx]
     pos = np.array(range(len(E)))
     deglocs = None
     for e in edeg:
	 pps = pos[abs(e-E)<etol]	
	 if(nocc-1 in pps and nocc in pps):	     
	     deglocs = np.array(pps)
	     break                
	
     if not deglocs is None:
         Cs = C[:,deglocs]
         Vemb = np.dot(Cs.conjugate().T,np.dot(dH,Cs))
         ee, vv = la.eigh(Vemb)
         Cnew = np.dot(Cs,vv)
         C[:,deglocs] = Cnew
     	
     #now do everything as per usual     
     Cocc = C[:,:nocc]
     Cvir = C[:,nocc:]

     Eocc = np.array([E[:nocc],]*(L-nocc))
     Evir = np.array([E[nocc:],]*nocc).T
     dE = Eocc-Evir

     if not deglocs is None:
	 print "D:",deglocs
	 occdegs = deglocs[deglocs<nocc] 
	 virdegs = deglocs[deglocs>=nocc] - nocc
         print "O:",occdegs
	 print "V:",virdegs 
	 dE[virdegs,occdegs] = 1.0
         
	 idE = np.divide(np.ones_like(dE),dE)
         idE[virdegs,occdegs] = 0.0 
         Zm = np.multiply(np.dot(Cvir.conjugate().T,np.dot(dH,Cocc)),idE)	 
     else:
         Zm = np.divide(np.dot(Cvir.conjugate().T,np.dot(dH,Cocc)),dE)
 
     Cmocc = np.dot(Cvir,Zm)
     result = np.dot(Cocc,Cmocc.conjugate().T) + np.dot(Cmocc,Cocc.conjugate().T)
     
     return result
 
def analyticGradientO_bcs(C,E,dH,degenCheck=False,etol=1.0e-10):
    
     L = dH.shape[1]/2
     L2 = dH.shape[1]
     
     if(degenCheck):
         #Provide the orbitals and energies of the bare hamiltonian 
         #Find degenerate states     
         idx = abs(E[1:]-E[:-1])<etol
         idx = np.append(idx,False)
         edeg = E[idx]
         pos = np.array(range(len(E)))
         deglocs = []
         for e in edeg:
            deglocs.append(pos[abs(e - E)<etol])                

         for d in deglocs:
            Cs = C[:,d]
            Vemb = np.dot(Cs.conjugate().T,np.dot(dH,Cs))
            ee, vv = la.eigh(Vemb)
            Cnew = np.dot(Cs,vv)
            C[:,d] = Cnew
 
     Ex = np.array([E[:L],]*L2)
     Ey = np.array([E,]*L).T
     dE = Ex-Ey

     if(degenCheck):
         locs = abs(dE)<etol
         dE[locs] = 1.0    
         idE = np.divide(np.ones_like(dE),dE)
         idE[locs] = 0.0 
         Zm = np.multiply(np.dot(C.conjugate().T,np.dot(dH,C[:,:L])),idE)
     else:
         Zm = np.divide(np.dot(C.conjugate().T,np.dot(dH,C[:,:L])),dE)
         
     np.fill_diagonal(Zm[:L,:L],0.0)

     Cocc = C[:,:L]
     C1occ = np.dot(C,Zm)
     result = np.dot(Cocc,C1occ.conjugate().T) + np.dot(C1occ,Cocc.conjugate().T)
    
     return result
    

    




