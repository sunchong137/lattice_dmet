import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton,brentq 
from scipy.optimize import least_squares 
from scipy.optimize import leastsq
import utils
sys.path.append('../')
import ftmodules

def minimizeBFGS(dmet, R, gtol = 1.0e-8, miter=1000):

    #Define parameters to optimize from previous DMET guess (vs guess from previous cluster in the loop ) of u-matrix associated with impurity sites in cluster
    #Note since u-matrix must be symmetric (if real) only optimize half the parameters
    #Note if constraining diagonal of u-matrix, first index of params will be the diagonal term, and the rest the upper triagonal of the u-matrix
    u_mat_imp = utils.extractImp(dmet.Nimp,dmet.u_mat)
    params = dmet.matrix2array( u_mat_imp )
    
    #TEST
    if(dmet.debugPrintRDMDiff):
        dmet.dlog.write('\n')
        dmet.dlog.write('Beginning minimization-BFGS\n')
        dmet.dlog.write('==================================================\n')
        dmet.dlog.flush()
 
    def costf(x):
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
        
        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        #hf1RDM_new, hforbs, hfE, hfevals = dmet.hfsolver(dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)
        hf1RDM_new, hforbs, hfE, hfevals = dmet.hfsolver(dmet,dmet.Nelec_tot,sh,dmet.g2e_site)
        #ROTATE RDM TO EMBEDDING SPACE
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))

        #FIT
        impfit = (hf1RDM_b-dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex]
        diff = np.linalg.norm(impfit)
        
        if(dmet.debugPrintRDMDiff):
            dmet.dlog.write('Guess diff: %10.6e\n' %(diff))
            dmet.dlog.flush()

        return diff**2            

    def jacf(x):
                                              
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )

        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        #hf1RDM_new, orbs, hfE, energies = dmet.hfsolver(dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)
        hf1RDM_new, orbs, hfE, energies = dmet.hfsolver(dmet,dmet.Nelec_tot,sh,dmet.g2e_site)
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))
        
        jac = np.zeros([len(x),R.shape[1],R.shape[1]], dtype = dmet.mtype)
        
        #Only concerned with corners
        for k in range(len(x)):
            dV = np.zeros_like(x)
            dV[k] = 1
            dVfull = dmet.replicate_u_matrix(dmet.array2matrix(dV))
            if dmet.T == 0.0:
                jac[k] = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nelec_tot),R))
            else:
                jac[k] = np.dot(R.conjugate().T,np.dot(ftmodules.analyticGradientT(orbs,energies,dVfull,dmet.Nelec_tot,dmet.T,dmet.grandmu),R))
            
        #Gradient function: careful with complex stuff
        gradfn = np.zeros(len(x),dtype=np.float64)
        diffrdm = (hf1RDM_b - dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex] 
        diffrdmR = diffrdm.real
        diffrdmI = diffrdm.imag
        
        for k in range(len(x)):
            #gradfn[k] = np.sum(2.*np.multiply(jac[k][:dmet.fitIndex,:dmet.fitIndex],diffrdmR))
            J = jac[k][:dmet.fitIndex,:dmet.fitIndex]
            gradfn[k] = 2.*np.sum(np.multiply(J.real,diffrdmR) + np.multiply(J.imag,diffrdmI))
            
        return gradfn
    
    #Minimize difference between HF and correlated DMET 1RDMs
    min_result = minimize( costf, params, method = 'BFGS', jac = jacf , tol = gtol, options={'maxiter': miter})
    #min_result = minimize( costf, params, method = 'BFGS' , tol = gtol, options={'maxiter': miter})
    #min_result = minimize( costf, params, method = 'CG' , tol = gtol, options={'maxiter': miter})
    #min_result = minimize( costf, params, method = 'BFGS', tol = gtol, options={'maxiter': miter})
    x = min_result.x
    
    print "Final Diff: ",min_result.fun,"Converged: ",min_result.status," Jacobian: ",np.linalg.norm(min_result.jac)
    
    if(not min_result.success):
        print "WARNING: Minimization unsuccessful. Message: ",min_result.message

    #Update new u-matrix from optimized parameters
    u_mat_imp = dmet.array2matrix(x)
    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )
    #utils.displayMatrix(dmet.u_mat_new)
    #####################################################################

