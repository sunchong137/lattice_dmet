import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton,brentq 
from scipy.optimize import least_squares 
from scipy.optimize import leastsq
import utils

def minimizeBFGS_BCS(dmet, R, gtol = 1.0e-8, miter=400):

    #Define parameters to optimize from previous DMET guess (vs guess from previous cluster in the loop ) of u-matrix associated with impurity sites in cluster
    #Note since u-matrix must be symmetric (if real) only optimize half the parameters
    #Note if constraining diagonal of u-matrix, first index of params will be the diagonal term, and the rest the upper triagonal of the u-matrix
    u_mat_imp = utils.extractImp(dmet.Nimp,dmet.u_mat)
    params = dmet.matrix2array( u_mat_imp )
    
    #TEST
    if(dmet.debugPrintRDMDiff):
        dmet.dlog.write('\n')
        dmet.dlog.write('Beginning minimization\n')
        dmet.dlog.write('==================================================\n')
        dmet.dlog.flush()

    h1body = dmet.h1 + dmet.fock2e + dmet.globalMu 

    def costf(x):
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
        
        gmu,hf1RDM_new, hforbs, hfE, hfevals = dmet.hfsolver(dmet.Nelec_tot,h1body,u_mat_ext,dmet.g2e_site)
        #ROTATE RDM TO EMBEDDING SPACE
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))

        #FIT
        impfit = (hf1RDM_b-dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex]
        diff = np.linalg.norm(impfit)
        
        if(dmet.debugPrintRDMDiff):
            #jacn = np.linalg.norm(jacf(x))
            #print "Guess diff: ",diff," grad: ",jacn
            #print "Guess diff: ",diff
            dmet.dlog.write('Guess diff: %10.6e\n' %(diff))
            dmet.dlog.flush()

        return diff**2            

    def jacf(x):
                                              
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )

        gmu,hf1RDM_new, orbs, hfE, energies = dmet.hfsolver(dmet.Nelec_tot,h1body,u_mat_ext,dmet.g2e_site)
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))
        
        jac = np.zeros([len(x),R.shape[1],R.shape[1]], dtype = dmet.mtype)
        
        #Only concerned with corners
        for k in range(len(x)):
            dV = np.zeros_like(x)
            dV[k] = 1
            a2m = dmet.array2matrix(dV)
            dVfull = dmet.replicate_u_matrix(a2m)
            #jac[k] = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO_bcs(orbs,energies,dVfull,True),R))
            #jac[k] = np.dot(R.conjugate().T,np.dot(utils.analyticGradientObcs(orbs,energies,dVfull,dmet.Nbasis),R))
            jac[k] = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nbasis),R))
            
        #Gradient function: careful with complex stuff
        gradfn = np.zeros(len(x),dtype=np.float64)
        diffrdm = (hf1RDM_b - dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex] 
        diffrdmR = diffrdm.real
        diffrdmI = diffrdm.imag
        
        for k in range(len(x)):
            #gradfn[k] = np.sum(2.*np.multiply(jac[k][:dmet.fitIndex,:dmet.fitIndex],diffrdmR))
            J = jac[k][:dmet.fitIndex,:dmet.fitIndex]
            gradfn[k] = 2.*np.sum(np.multiply(J.real,diffrdmR) + np.multiply(J.imag,diffrdmI))
        
        #print "Gradient: ",np.linalg.norm(gradfn)    
        return gradfn
    
    #Minimize difference between HF and correlated DMET 1RDMs
    min_result = minimize( costf, params, method = 'BFGS', jac = jacf , tol = gtol, options={'maxiter': miter, 'disp': True})
    #min_result = minimize( costf, params, method = 'BFGS', tol = gtol, options={'maxiter': miter})
    x = min_result.x
    
    print "Final Diff: ",min_result.fun**0.5,"Converged: ",min_result.status," Jacobian: ",np.linalg.norm(min_result.jac)      
    if(not min_result.success):
        print "WARNING: Minimization unsuccessful. Message: ",min_result.message
        dmet.checkPoint('analyze.hdf5')

    #Update new u-matrix from optimized parameters
    u_mat_imp = dmet.array2matrix(x)
    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )
    #utils.displayMatrix(dmet.u_mat_new)
    #####################################################################


