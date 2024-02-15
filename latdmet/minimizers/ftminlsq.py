import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton,brentq 
from scipy.optimize import least_squares 
from scipy.optimize import leastsq
import utils
sys.path.append('../')
import ftmodules

def minimizelsq( dmet, R, gtol = 1.0e-6, miter = 10):

    #Define parameters to optimize from previous DMET guess (vs guess from previous cluster in the loop ) of u-matrix associated with impurity sites in cluster
    #Note since u-matrix must be symmetric (if real) only optimize half the parameters
    #Note if constraining diagonal of u-matrix, first index of params will be the diagonal term, and the rest the upper triagonal of the u-matrix
    
    #TEST
    if(dmet.debugPrintRDMDiff):
        dmet.dlog.write('\n')
        dmet.dlog.write('Beginning minimization\n')
        dmet.dlog.write('==================================================\n')
        dmet.dlog.flush()
    
    ll = np.triu_indices(dmet.fitIndex)

    def costf(x):

        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
            
        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        hf1RDM_new, hforbs, hfE, hfevals = dmet.hfsolver(dmet, dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)
        #ROTATE RDM TO EMBEDDING SPACE
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))

        #FIT
        #print "norm", np.linalg.norm(hf1RDM_b[:2,:2]-hf1RDM_b[2:4, 4:6])
        impfit = (hf1RDM_b-dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex]

        residuals = impfit[ll]
        if dmet.fitIndex == 2*dmet.Nimp:
            residuals = dmet.matrix2array(impfit)
        else:
            residuals = impfit[ll]

        if(dmet.debugPrintRDMDiff):
            dmet.dlog.write('Guess diff: %10.6e\n' %(np.linalg.norm(impfit)))
            dmet.dlog.flush()
    
        return residuals 
 
    def jacf(x):
    
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )

        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        hf1RDM_new, orbs, hfE, energies = dmet.hfsolver(dmet, dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)

        #Jacobian must be function by variables 
        #jac = np.zeros([len(x),dmet.paramss], dtype = np.float64)
        jac = np.zeros([dmet.paramss,len(x)], dtype = np.float64)

        #Only concerned with corners
        for k in range(len(x)):
            dV = np.zeros_like(x)
            dV[k] = 1
            dVfull = dmet.replicate_u_matrix(dmet.array2matrix(dV))
           
            #drho = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nelec_tot),R))[:dmet.fitIndex,:dmet.fitIndex]
            #Get d\rho/dV[k]
            if dmet.T == 0.0:
                drho = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nelec_tot),R))[:dmet.fitIndex,:dmet.fitIndex]
            else:
                drho = np.dot(R.conjugate().T,np.dot(ftmodules.analyticGradientT(orbs,energies,dVfull,dmet.Nelec_tot,dmet.T,dmet.grandmu),R))[:dmet.fitIndex,:dmet.fitIndex]

            if dmet.fitIndex == 2*dmet.Nimp:
                erho = dmet.matrix2array(drho)
            else:
                erho = drho[ll]
            jac[:,k] = erho
        return jac 
    
    #Minimize difference between HF and correlated DMET 1RDMs
    u_mat_imp = utils.extractImp(dmet.Nimp,dmet.u_mat)
    params = dmet.matrix2array( u_mat_imp )
    pbounds = ([-5.0 for x in params],[5.0 for x in params])
    
    #min_result = least_squares(costf, params, ftol = gtol, jac = '2-point', max_nfev = miter*len(params), bounds = pbounds)
    min_result = least_squares(costf, params, ftol = gtol, jac = jacf, max_nfev = miter*len(params), bounds = pbounds)
    #min_result = least_squares(costf, params, ftol = gtol, max_nfev = miter*len(params), bounds = pbounds)
    x = min_result.x
    ier = min_result.status
    cost = min_result.cost

    #x,cov,info,msg,ier = leastsq(costf, params, ftol = gtol, Dfun = jacf, full_output = True, factor=0.1)
    #x,cov,info,msg,ier = leastsq(costf, params, ftol = gtol, full_output = True, factor=0.1)
    #cost = np.linalg.norm(info['fvec'])

    print "Final Diff: ",cost,"Converged: ",ier," Gradient: ",np.linalg.norm(min_result.grad)
    print "Converge Msg: ",min_result.message    

    #Update new u-matrix from optimized parameters
    u_mat_imp = dmet.array2matrix(x)

    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )
    #utils.displayMatrix(dmet.u_mat_new)



#####################################################################
 
def minimizelsq_fixdiag( dmet, R, gtol = 1.0e-6, miter = 10):

    #Define parameters to optimize from previous DMET guess (vs guess from previous cluster in the loop ) of u-matrix associated with impurity sites in cluster
    #Note since u-matrix must be symmetric (if real) only optimize half the parameters
    #Note if constraining diagonal of u-matrix, first index of params will be the diagonal term, and the rest the upper triagonal of the u-matrix
    
    #TEST
    if(dmet.debugPrintRDMDiff):
        dmet.dlog.write('\n')
        dmet.dlog.write('Beginning minimization\n')
        dmet.dlog.write('==================================================\n')
        dmet.dlog.flush()
    
    #h2el = hf.make_h2el(dmet.g2e_site,hf1rdm_site)    
    #fock = dmet.h1 + h2el  
    ll = np.triu_indices(dmet.fitIndex)

    def costf(x):

        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix_fixdiag(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
            
        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        hf1RDM_new, hforbs, hfE, hfevals = dmet.hfsolver(dmet, dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)
        #ROTATE RDM TO EMBEDDING SPACE
        hf1RDM_b = np.dot(R.conjugate().T,np.dot(hf1RDM_new,R))

        #FIT
        #print "norm", np.linalg.norm(hf1RDM_b[:2,:2]-hf1RDM_b[2:4, 4:6])
        impfit = (hf1RDM_b-dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex]

        residuals = impfit[ll]
        if dmet.fitIndex == 2*dmet.Nimp:
            residuals = dmet.matrix2array(impfit)
        else:
            residuals = impfit[ll]

        if(dmet.debugPrintRDMDiff):
            dmet.dlog.write('Guess diff: %10.6e\n' %(np.linalg.norm(impfit)))
            dmet.dlog.flush()
    
        return residuals 
 
    def jacf(x):
    
        R = dmet.RotationMatrix
        u_mat_imp = dmet.array2matrix_fixdiag(x)
        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )

        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        hf1RDM_new, orbs, hfE, energies = dmet.hfsolver(dmet, dmet.Nelec_tot,sh,dmet.g2e_site, itrmax = dmet.searchIter, doDIIS = True, hfoscitr = dmet.searchOscIter, gap = dmet.hfgap)

        #Jacobian must be function by variables 
        #jac = np.zeros([len(x),dmet.paramss], dtype = np.float64)
        jac = np.zeros([dmet.paramss,len(x)], dtype = np.float64)

        #Only concerned with corners
        for k in range(len(x)):
            dV = np.zeros_like(x)
            dV[k] = 1
            dVfull = dmet.replicate_u_matrix(dmet.array2matrix_fixdiag(dV))
           
            #drho = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nelec_tot),R))[:dmet.fitIndex,:dmet.fitIndex]
            #Get d\rho/dV[k]
            if dmet.T == 0.0:
                drho = np.dot(R.conjugate().T,np.dot(utils.analyticGradientO(orbs,energies,dVfull,dmet.Nelec_tot),R))[:dmet.fitIndex,:dmet.fitIndex]
            else:
                drho = np.dot(R.conjugate().T,np.dot(ftmodules.analyticGradientT(orbs,energies,dVfull,dmet.Nelec_tot,dmet.T,dmet.grandmu),R))[:dmet.fitIndex,:dmet.fitIndex]

            if dmet.fitIndex == 2*dmet.Nimp:
                erho = dmet.matrix2array(drho)
            else:
                erho = drho[ll]
            jac[:,k] = erho
        return jac 
    
    #Minimize difference between HF and correlated DMET 1RDMs
    u_mat_imp = utils.extractImp(dmet.Nimp,dmet.u_mat)
    params = dmet.matrix2array_fixdiag( u_mat_imp )
    pbounds = ([-5.0 for x in params],[5.0 for x in params])
    
    min_result = least_squares(costf, params, ftol = gtol, jac = jacf, max_nfev = miter*len(params), bounds = pbounds)
    #min_result = least_squares(costf, params, ftol = gtol, max_nfev = miter*len(params), bounds = pbounds)
    x = min_result.x
    ier = min_result.status
    cost = min_result.cost

    #x,cov,info,msg,ier = leastsq(costf, params, ftol = gtol, Dfun = jacf, full_output = True, factor=0.1)
    #x,cov,info,msg,ier = leastsq(costf, params, ftol = gtol, full_output = True, factor=0.1)
    #cost = np.linalg.norm(info['fvec'])

    print "Final Diff: ",cost,"Converged: ",ier," Gradient: ",np.linalg.norm(min_result.grad)
    print "Converge Msg: ",min_result.message    

    #Update new u-matrix from optimized parameters
    u_mat_imp = dmet.array2matrix_fixdiag(x)

    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )

