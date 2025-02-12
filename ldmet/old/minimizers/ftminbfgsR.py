import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton,brentq 
from scipy.optimize import least_squares 
from scipy.optimize import leastsq
import utils
sys.path.append('../')
import ftmodules

def minimizeBFGSR(dmet, R, gtol = 1.0e-12, miter=1000):

    u_mat_imp = utils.extractImp(dmet.Nimp,dmet.u_mat)
    params = dmet.matrix2array( u_mat_imp )
    Nimp = dmet.Nimp
    Nbasis = dmet.Nbasis
    
    #TEST
    if(dmet.debugPrintRDMDiff):
        dmet.dlog.write('\n')
        dmet.dlog.write('Beginning minimization-BFGS\n')
        dmet.dlog.write('==================================================\n')
        dmet.dlog.flush()

 
    R = dmet.RotationMatrix
    sh0 = dmet.h1 + dmet.fock2e
    sh0_emb = np.dot(R.conj().T, np.dot(sh0, R))
    Nb = dmet.Nbasis/dmet.Nimp
    def costf(x):
        u_mat_imp = dmet.array2matrix(x)

        #u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
        #sh = dmet.h1 + dmet.fock2e + u_mat_ext
        #sh_emb = np.dot(R.conj().T, np.dot(sh, R))

        su = np.zeros_like(sh0_emb)
        for k in range(Nb):
            Rpu = R[k*Nimp:(k+1)*Nimp]
            Rpd = R[(Nbasis+k*Nimp):(Nbasis+(k+1)*Nimp)]
            su += np.dot(Rpu.conj().T, np.dot(u_mat_imp[:Nimp,:Nimp], Rpu))
            su += np.dot(Rpd.conj().T, np.dot(u_mat_imp[Nimp:,Nimp:], Rpd))
        sh_emb = sh0_emb + su

        hf1RDM_b,_,_,_ = dmet.hfsolver(dmet, dmet.actElCount,sh_emb)

        impfit = (hf1RDM_b-dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex]
        diff = np.linalg.norm(impfit)
        
        if(dmet.debugPrintRDMDiff):
            dmet.dlog.write('Guess diff: %10.6e\n' %(diff))
            dmet.dlog.flush()

        return diff**2            

    def jacf(x):
                                              
        u_mat_imp = dmet.array2matrix(x)

        u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
        sh = dmet.h1 + dmet.fock2e + u_mat_ext
        sh_emb = np.dot(R.conj().T, np.dot(sh, R))

        #su = np.zeros_like(sh0_emb)
        #for k in range(Nb):
        #    Rpu = R[k*Nimp:(k+1)*Nimp]
        #    Rpd = R[(Nbasis+k*Nimp):(Nbasis+(k+1)*Nimp)]
        #    su += np.dot(Rpu.conj().T, np.dot(u_mat_imp[:Nimp,:Nimp], Rpu))
        #    su += np.dot(Rpd.conj().T, np.dot(u_mat_imp[Nimp:,Nimp:], Rpd))
        #sh_emb = sh0_emb + su

        hf1RDM_b,orbs,_,energies = dmet.hfsolver(dmet, dmet.actElCount,sh_emb)
        nb = hf1RDM_b.shape[-1]
        
        jac = np.zeros([len(x),nb,nb], dtype = dmet.mtype)
        
        #TODO implement the gradient
        for k in range(len(x)):
            dV = np.zeros_like(x)
            dV[k] = 1
            dVfull = dmet.replicate_u_matrix(dmet.array2matrix(dV))
            dVemb = np.dot(R.conj().T, np.dot(dVfull, R))
            if dmet.T == 0.0:
                jac[k] = utils.analyticGradientO(orbs,energies,dVemb,dmet.actElCount)
            else:
                jac[k] = ftmodules.analyticGradientT(orbs,energies,dVemb,dmet.actElCount,dmet.T,dmet.grandmu)
            
        #Gradient function: careful with complex stuff
        gradfn = np.zeros(len(x),dtype=np.float64)
        diffrdm = (hf1RDM_b - dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex] 
        diffrdmR = diffrdm.real
        diffrdmI = diffrdm.imag
        
        for k in range(len(x)):
            #gradfn[k] = np.sum(2.*np.multiply(jac[k][:dmet.fitIndex,:dmet.fitIndex],diffrdmR))
            J = jac[k][:dmet.fitIndex,:dmet.fitIndex]
            gradfn[k] = 2.*np.sum(np.multiply(J.real,diffrdmR) + np.multiply(J.imag,diffrdmI))
            
        #np.savetxt('/home/sunchong/anagrad.txt',gradfn)
        #ng_ = numgrad(x)
        #np.savetxt('/home/sunchong/numgrad.txt',ng_)
        #ng_n = jacf_n(x)
        #np.savetxt('/home/sunchong/ftgrad.txt',ng_n)
        #exit()
        return gradfn

    #def jacf_n(x):
    #    u_mat_imp = dmet.array2matrix(x)
    #    u_mat_ext = dmet.replicate_u_matrix( u_mat_imp )
    #    limp = u_mat_imp.shape[-1]
    #    print limp
    #        
    #    sh = dmet.h1 + dmet.fock2e + u_mat_ext
    #    sh_emb = np.dot(R.conj().T, np.dot(sh, R))
    #    hf1RDM_b,orbs,_,energies = dmet.hfsolver(dmet, dmet.actElCount,sh_emb)
    #    nb = hf1RDM_b.shape[-1]
    #    grad_emb = ftmodules.analyticGradientT_N(orbs,energies,dmet.T,dmet.grandmu)
    #    ugrad = np.einsum('ik,lj->ijkl', (R.conj().T)[:,:limp], R[:limp,:])
    #    ugrad = ugrad[np.triu_indices(nb)]
    #    jac = np.einsum('pij, pkl -> ijkl', ugrad,grad_emb)

    #    jac = jac[np.triu_indices(limp)][:,:dmet.fitIndex,:dmet.fitIndex]
    #    diffrdm = (hf1RDM_b - dmet.IRDM1)[:dmet.fitIndex,:dmet.fitIndex] 
    #    grad = np.einsum('pij,ij -> p', jac, diffrdm)
    #    
    #    return grad
        

    def numgrad(x):
        gradfn = np.zeros(len(x))
        for k in range(len(x)):
            xmin = x.copy()
            xmax = x.copy()
            xmin[k] -= 0.01
            xmax[k] += 0.01
            fmin = costf(xmin)
            fmax = costf(xmax)
            gradfn[k] = (fmax-fmin)/0.02
        return gradfn

    #Minimize difference between HF and correlated DMET 1RDMs
    #min_result = minimize( costf, params, method = 'BFGS', jac = jacf , tol = gtol, options={'maxiter': miter})
    min_result = minimize( costf, params, method = 'BFGS', tol = gtol, options={'maxiter': miter})

    x = min_result.x
    print "Final Diff: ",min_result.fun,"Converged: ",min_result.status," Jacobian: ",np.linalg.norm(min_result.jac)
    
    if(not min_result.success):
        print "WARNING: Minimization unsuccessful. Message: ",min_result.message

    #Update new u-matrix from optimized parameters
    u_mat_imp = dmet.array2matrix(x)
    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )
    #utils.displayMatrix(dmet.u_mat_new)
    #####################################################################

