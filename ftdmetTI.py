#from dmet_ur.hfb_hubbard_2 import dmetTI
from dmetTI import dmet
import numpy as np
import sys
import os
import time

import utils
from scipy import linalg as la
from scipy import stats
import h5py
import diis
import random as rnd

import loggers.simpleLogger as log
from meanfields import *    
import minimizers as mmizer 
import solvers as solver    

# FT modules
import ftmodules

class ftdmet(dmet):
    def __init__(self, Nbasis, Nelec_tot, Nimp, h1e_site, g2e_site, h1esoc_site=None, SolverType='FCI', u_matrix=None, mtype = np.float64, ctype = 'UHF', globalMu=None, T=0.0, grandMu=0.0, BathOrder=1, maxm=2000, tau=0.1, fix_udiag=False, hbath=False, fitmu=False, utol=5e-6,etol=1e-5,ntol=1e-5):

        dmet.__init__(self, Nbasis, Nelec_tot, Nimp, h1e_site, g2e_site, h1esoc_site, SolverType, u_matrix, mtype, ctype, globalMu,utol,etol,ntol)

        # add the temperature argument
        self.T = T
        self.iformalism = False #will be redefined after declaring the dmet obj
        self.grandmu = grandMu #grand canonical ensemble mu
        self.bath_order = BathOrder # the order of bath
        self.MaxMPSM = maxm
        self.TimeStep = tau
        self.doDIIS = True
        #self.minimize_u_matrix = mmizer.minimizelsq
        self.minimize_u_matrix = mmizer.minimizeBFGSR
        #self.minimize_u_matrix = mmizer.minimizeBFGS
        self.hbath = hbath
        self.fitmu = fitmu
        if fix_udiag:
            #self.minimize_u_matrix = mmizer.minimizeBFGS_fixdiag 
            self.minimize_u_matrix = mmizer.minimizelsq_fixdiag

########################################################################################
    def updateParams(self):
        dmet.updateParams(self)
        if(self.solverType=="MPS"):
            self.misolver = solver.microIterationMPS
        elif(self.solverType=="FCI"):
            self.misolver = solver.microIteration
        if not self.fitmu:
            self.startMu = self.grandmu
    def displayParameters(self):
        dmet.displayParameters(self)
        print "Temperature of the system: %4.4f t"%self.T
        print "Chemical potential of the system: ", self.grandmu

########################################################################################
    def get_rotmat_svd(self, RDM1, use_ham=False):

        # get_rotmat function is defined differently from the one in dmet class
        # SVD of AB block
        Nbasis = self.Nbasis
        Nimp = self.Nimp

        if use_ham:
            print "Using lattice hamiltonian to construct the bath orbitals!"
            h_lat = self.h1 + self.u_mat + self.fock2e
            rdma = h_lat[:Nbasis,:Nbasis]
            rdmb = h_lat[Nbasis:,Nbasis:]
        else:
            rdma = RDM1[:Nbasis, :Nbasis]
            rdmb = RDM1[Nbasis:, Nbasis:]
     
        
        ABa = rdma[:Nimp, Nimp:Nbasis]
        ABb = rdmb[:Nimp, Nimp:Nbasis]
 
        _,sngvla,BaT = np.linalg.svd(ABa, full_matrices=False)
        _,sngvlb,BbT = np.linalg.svd(ABb, full_matrices=False)
 
        if self.constrainType == 'RHF':
            BbT = BaT.copy()
 
        lba = BaT.shape[0]
        lbb = BbT.shape[0]

        rotmat = np.zeros([ 2*Nbasis, 2*Nimp + lba + lbb])
        rotmat[:Nimp,:Nimp] = np.eye(Nimp)
        rotmat[Nbasis:Nimp+Nbasis,Nimp:2*Nimp] = np.eye(Nimp)
        rotmat[Nimp:Nbasis,2*Nimp:2*Nimp + lba] = BaT.T.conj()
        rotmat[Nbasis + Nimp:Nbasis+Nbasis,2*Nimp+lba:] = BbT.T.conj()

        self.actElCount = (Nimp*2 + lba + lbb)/2
        #The embedding system is always half-filling!
        Ncore = (self.Nelec_tot - self.actElCount)/2
        print 'Virtual orbital #: ', Nbasis-Ncore-(Nimp*2 + lba + lbb)/2
        print 'Bath orbital #:     ', Nimp
        print 'Core orbital #:     ', Ncore
        print 'Actual Electron #: ', self.actElCount

        if(self.fitBath):
            self.fitIndex = rotmat.shape[-1] #includes imp+bath orbitals
            self.paramss = self.fitIndex*(self.fitIndex+1)/2
        else:
            self.fitIndex = self.Nimp*2
            #self.paramss = self.fitIndex*(self.fitIndex+1)/2
            self.paramss = self.Nimp*(self.Nimp+1)

        rdmCsite = np.zeros((2*Nbasis, 2*Nbasis)) # rdmCsite is not used in NIbath formula
        if self.iformalism:
            Rup = np.zeros((Nbasis, lba+Nimp))
            Rdn = np.zeros((Nbasis, lbb+Nimp))
            Rup[:Nimp, :Nimp] = np.eye(Nimp)
            Rdn[:Nimp, :Nimp] = np.eye(Nimp)
            Rup[Nimp:, Nimp:] = BaT.T.conj()
            Rdn[Nimp:, Nimp:] = BbT.T.conj()
            rdma_emb = np.dot(Rup.conj().T, np.dot(rdma, Rup))
            rdmb_emb = np.dot(Rdn.conj().T, np.dot(rdmb, Rdn))
            rdmCsite[:Nbasis, :Nbasis] = rdma - np.dot(Rup, np.dot(rdma_emb, Rup.conj().T))
            rdmCsite[Nbasis:, Nbasis:] = rdmb - np.dot(Rdn, np.dot(rdmb_emb, Rdn.conj().T))


        return rotmat, rdmCsite

##################################################################################
    def get_rotmat(self, RDM1):
        
        # generate rotmat from orders of RDM1, i.e., I, RDM1, RDM1**2,..., RDM1**qrpower
        # impurity basis is from I, bath is from RDM1, second order bath from RDM1**2

        #if self.T < 1e-3:
        #    return self.get_rotmat_svd(RDM1)

        qrpower = self.bath_order
        Nbasis = self.Nbasis
        Nimp = self.Nimp
        use_ham = self.hbath

        if use_ham:
            print "USING one-body Hamiltonian to calculate the bath orbitals!"
            h_lat = self.h1 + self.u_mat + self.fock2e
            rdma = h_lat[:Nbasis,:Nbasis]
            rdmb = h_lat[Nbasis:,Nbasis:]
        else:
            rdma = RDM1[:Nbasis, :Nbasis]
            rdmb = RDM1[Nbasis:, Nbasis:]
        
        Qsa = []
        Qsb = []
        for p in range(qrpower+1):
            dmpowa = np.linalg.matrix_power(rdma,p) 
            dmpowb = np.linalg.matrix_power(rdmb,p) 
            Qa, _ = np.linalg.qr(dmpowa[:, :Nimp])
            Qb, _ = np.linalg.qr(dmpowb[:, :Nimp])
            Qsa.append(Qa)
            Qsb.append(Qb)

        mergedQsa = np.hstack(Qsa)
        mergedQsb = np.hstack(Qsb)
        Qtota, _ = np.linalg.qr(mergedQsa) # re-orthogonalize
        Qtotb, _ = np.linalg.qr(mergedQsb) 
        bath_a = Qtota[Nimp:, Nimp:]
        bath_b = Qtotb[Nimp:, Nimp:]

        lba = Qtota.shape[-1]
        lbb = Qtotb.shape[-1]

        Nemb = lba + lbb # including the impurity orbs

        print "Bath orbitals are localized with Boys method!"
        rotmat = np.zeros((Nbasis*2, Nemb))
        rotmat[:Nimp,:Nimp] = np.eye(Nimp)
        rotmat[Nbasis:Nimp+Nbasis,Nimp:2*Nimp] = np.eye(Nimp)
        rotmat[Nimp:Nbasis,2*Nimp:(Nimp + lba)] = bath_a #Qtota[Nimp:, Nimp:]
        rotmat[(Nbasis+Nimp):Nbasis+Nbasis,(Nimp+lba):] = bath_b #Qtotb[Nimp:, Nimp:]


        #np.set_printoptions(3, linewidth=1000)

        # No need to calculate embedding electron number
        self.actElCount = Nemb/2
        Ncore = (self.Nelec_tot - self.actElCount)/2

        print 'The order of the bath: ', qrpower
        print 'Bath orbital #: ', (Nemb - 2*Nimp)/2
        print 

        if(self.fitBath):
            self.fitIndex = rotmat.shape[-1] #includes imp+bath orbitals
            self.paramss = self.fitIndex*(self.fitIndex+1)/2
        else:
            self.fitIndex = self.Nimp*2
            self.paramss = self.Nimp*(self.Nimp+1)

        rdmCsite = np.zeros((2*Nbasis, 2*Nbasis)) # rdmCsite is not used in NIbath formula

        if self.iformalism:
            Rup = np.zeros((Nbasis, lba))
            Rdn = np.zeros((Nbasis, lbb))
            Rup[:Nimp, :Nimp] = np.eye(Nimp)
            Rdn[:Nimp, :Nimp] = np.eye(Nimp)
            Rup[Nimp:, Nimp:] = bath_a
            Rdn[Nimp:, Nimp:] = bath_b
            rdma_emb = np.dot(Rup.conj().T, np.dot(rdma, Rup))
            rdmb_emb = np.dot(Rdn.conj().T, np.dot(rdmb, Rdn))
            rdmCsite[:Nbasis, :Nbasis] = rdma - np.dot(Rup, np.dot(rdma_emb, Rup.conj().T))
            rdmCsite[Nbasis:, Nbasis:] = rdmb - np.dot(Rdn, np.dot(rdmb_emb, Rdn.conj().T))

        return rotmat, rdmCsite

########################################################################################
    def get_embedding_ham( self, R ):

        #augment 1e- terms w/u-matrix only on bath because not doing casci
        #note, the following works b/c u_mat is block diagonal for each cluster        
        h_curr = np.copy(self.h1+self.globalMu)
 
        #rotate the 1/2 e- terms into embedding basis using rotation matrix
        #note, if doing dca it is the dca embedding basis
        h_emb = utils.rot1el( h_curr, R )
      
        #Shift for NI-formalism  

        ld = 2*self.Nbasis
        if(self.iformalism): 
            #Interacting Bath
            if(self.g2e_site.shape[0] == self.Nbasis):
                V2 = np.zeros([ld,ld,ld,ld],dtype=self.mtype)
                V2[:self.Nbasis,:self.Nbasis,:self.Nbasis,:self.Nbasis] = self.g2e_site
                V2[self.Nbasis:ld,self.Nbasis:ld,self.Nbasis:ld,self.Nbasis:ld] = self.g2e_site
                V2[:self.Nbasis,:self.Nbasis,self.Nbasis:ld,self.Nbasis:ld] = self.g2e_site
                V2[self.Nbasis:ld,self.Nbasis:ld,:self.Nbasis,:self.Nbasis] = self.g2e_site
                self.V2 = V2
                V_emb = utils.rot2el_chem_full(V2,R)
            else:
                ls = self.Nbasis
                lb = R.shape[-1]/2
                Rup = np.zeros((ls, lb))
                Rdn = np.zeros((ls, lb))
                Rup[:self.Nimp,:self.Nimp] = np.eye(self.Nimp)
                Rup[self.Nimp:,self.Nimp:] = R[self.Nimp:self.Nbasis, 2*self.Nimp:self.Nimp+lb]

                Rdn[:self.Nimp,:self.Nimp] = np.eye(self.Nimp)
                Rdn[self.Nimp:,self.Nimp:] = R[self.Nbasis+self.Nimp:, self.Nimp+lb:]
                
                Vaa = np.zeros((lb, lb, lb, lb))
                Vab = np.einsum('pi,iq,ri,is -> pqrs',Rup.conjugate().T,Rup,Rdn.conjugate().T,Rdn)*self.g2e_site[0]
                V_emb = (Vaa, Vab, Vaa)

        else:

            lb = R.shape[-1]/2
            V_ab = np.zeros((lb,lb,lb,lb), dtype=self.mtype)
            V_aa = np.zeros((lb,lb,lb,lb), dtype=self.mtype)
            for i in range(self.Nimp):
                V_ab[i,i,i,i] = self.g2e_site[0]
            V_emb = (V_aa, V_ab, V_aa)

        return h_emb, V_emb
    #####################################################################

    def get_impurityEnergy( self, h1_emb, V2_emb, corr1RDM, corr2RDM):

        #Note the 2RDM is expected to be in chemistry format
        #Note here Nimp is again specified in orbital size and NOT spon-orbital size 
        #Calculate Site Energy


        if(self.solverType=='MPS'):
            return self.ImpurityEnergy # DMET energy calculated inside the solver

        else:
            ld = h1_emb.shape[-1]

            E1 = 0.0
            E2 = 0.0
            Eclust = 0.0
            for orb1 in range(2*self.Nimp):
                for orb2 in range(ld):
                    E1 += corr1RDM[ orb1, orb2 ]*(h1_emb[ orb2, orb1 ])

            for i in range(len(corr2RDM)):
                for orb1 in range(self.Nimp):
                    for orb2 in range(ld/2):
                        for orb3 in range(ld/2):
                            for orb4 in range(ld/2):
                                E2 += corr2RDM[i][ orb1, orb2, orb3, orb4 ]*V2_emb[i][ orb2, orb1, orb4, orb3]
            rdm2diag = []
            for i in range(self.Nimp):
                rdm2diag.append(corr2RDM[1][i,i,i,i])
            rdm2diag = np.asarray(rdm2diag)
            docc = np.average(rdm2diag)
            print "RDM2 diagonal terms: ", rdm2diag
            netot = 0.
            for i in range(2*self.Nimp):
                netot += corr1RDM[i,i]

            print "T-Double occ (Iform):   %.2f      %.12f"%(self.T, docc)
            print "T-Ne on impurity:    %.2f      %.12f"%(self.T, netot)
            print '----------------------------------------------------------'

            print '----------------------------------------------------------'
            print '1 body energy contribution: ', E1/self.Nimp
            print '2 body energy contribution: ', E2/self.Nimp
            return E1+E2

    #####################################################################
    def get_impurityEnergyNI(self,h_emb,V_emb,rdm1,rdm2):
    
        h1 = ftmodules.perm1el(h_emb, self.Nimp)
    
        ld = h_emb.shape[-1]
        Norb = ld/2
    
        #d1 = ftutils.perm1el(rdm1)
        #print d1[Norb:,Norb:]
    
        li = self.Nimp
        E1 = 0.0
    
        for orb1 in range(2*self.Nimp):
            for orb2 in range(ld):
                E1 += rdm1[ orb1, orb2 ]*(h_emb[ orb2, orb1 ])
    
        E2 = 0.0
        for i in range(3):
            for orb1 in range(li):
                for orb2 in range(li):
                    for orb3 in range(li):
                        for orb4 in range(li):
                            E2 += rdm2[i][ orb1, orb2, orb3, orb4 ]*V_emb[i][ orb2, orb1, orb4, orb3]
    
    
        rdm2diag = []
        for i in range(self.Nimp):
            rdm2diag.append(rdm2[1][i,i,i,i])
        rdm2diag = np.asarray(rdm2diag)
        docc = np.average(rdm2diag)
        print "RDM2 diagonal terms: ", rdm2diag
        netot = 0.
        for i in range(2*self.Nimp):
            netot += rdm1[i,i]
        
        print "T-Double occ (NIform):   %.4f      %.12f"%(self.T, docc)
        print "T-Ne on impurity:    %.4f      %.12f"%(self.T, netot)
        print '----------------------------------------------------------'
        print '1 body energy contribution: ', E1/self.Nimp
        print '2 body energy contribution: ', E2/self.Nimp
        return E1 + E2

########################################################################################
    def matrix2array_fixdiag( self, mat ):
        
        #Expects input in [2*Nimp,2*Nimp] format
        
        if(self.constrainType == 'UHF'):        
            p = 0
            ls = self.Nimp*(self.Nimp-1)/2
            array = np.zeros([self.Nimp*(self.Nimp-1)],dtype=np.float64)
            for i in range(self.Nimp-1):
                for j in range(i+1,self.Nimp):
                    array[p] = mat[i,j]
                    array[p+ls] = mat[i+self.Nimp,j+self.Nimp]
                    p+=1

        elif(self.constrainType == 'RHF'):
            p = 0
            ls = self.Nimp*(self.Nimp-1)/2
            array = np.zeros([self.Nimp*(self.Nimp-1)/2],dtype=np.float64)
            for i in range(self.Nimp-1):
                for j in range(i+1,self.Nimp):
                    array[p] = mat[i,j]
                    p+=1
        else:
            raise RunTimeError("Invalid constraint option") 
    
        return array

########################################################################################
    def array2matrix_fixdiag( self, array ):

        mat = np.zeros([self.Nimp*2,self.Nimp*2],dtype=self.mtype)
        
        if(self.constrainType == 'UHF'):
            ls = self.Nimp*(self.Nimp-1)/2     
            p = 0
            for i in range(self.Nimp-1):
                for j in range(i+1,self.Nimp):
                    mat[i,j] = array[p]
                    mat[i+self.Nimp,j+self.Nimp] = array[p+ls]
                    p += 1
                    
            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] = self.grandmu
                
        elif(self.constrainType == 'RHF'):
            p = 0
            for i in range(self.Nimp-1):
                for j in range(i+1,self.Nimp):
                    mat[i,j] = array[p]
                    mat[i+self.Nimp,j+self.Nimp] = array[p]
                    p += 1
 
            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] = self.grandmu
            
        else:
            raise RuntimeError("Invalid constraint option")

        return mat


########################################################################################
    def generateBath(self):

        print
        print "========================================================================="
        print "STARTING MF PROBLEM:"
        print "========================================================================="
        print

        self.label = 0

        #Calculate modified 1e- terms in site basis
        h_site_mod = self.h1 + self.u_mat + self.fock2e

        '''
        if(itr == 1):
            import transform        
            transform.makePlot(h_site_mod,self.Nimp,self.Nelec_tot,'save_hf.png') 
        '''
        '''
        print "HF impurity Hamiltonian"
        eo = utils.extractImp(self.Nimp,h_site_mod)
        utils.displayMatrix(eo)

            print "HF Hamiltonian"
        utils.displayMatrix(h_site_mod)
        '''

        #Run HF using modified 1 e- and original 2 e- terms to generate HF orbitals in original site basis
        t1 = time.time()
        if(self.constrainType!='BCS'):
            self.hf1RDM_site, hforbs, hfE, hfevals = self.hfsolver(self,self.Nelec_tot, h_site_mod, self.g2e_site, needEnergy = True)
        else:
            if(self.critnorm > 5.0e-2):
                self.globalMu,self.hf1RDM_site, hforbs, hfE, hfevals = self.hfsolver(self,self.Nelec_tot, self.h1, self.u_mat, self.g2e_site, needEnergy = True, display=True,muFind = True)
            else:
                self.globalMu,self.hf1RDM_site, hforbs, hfE, hfevals = self.hfsolver(self,self.Nelec_tot, self.h1, self.u_mat, self.g2e_site, needEnergy = True, display=True,muFind = True,muguess = -self.globalMu[0,0])
        t2 = time.time()
        hftime = t2-t1

        #Make the fock term for HF if doing HF self-consistently
        '''
        if(self.scSolver):
            self.fock2e = hf.make_h2el(self.g2e_site, self.hf1RDM_site)
        else:
            self.fock2e = 0
        '''

        print
        #print "HF orbital energies: ",hfevals
        print "HF Time: ",hftime
        print "HF Energy: ",hfE
        print "Hartree Fock Energy per site: ", hfE/self.Nbasis
        print "Trace of RDM: ", self.hf1RDM_site.trace()
        print "Target trace of RDM: ", self.Nelec_tot
        print "Trace of alpha: ",self.hf1RDM_site[:self.Nbasis, :self.Nbasis].trace()
        print "Trace of beta: ",self.hf1RDM_site[self.Nbasis:, self.Nbasis:].trace()


        if(self.constrainType=='BCS'):
            hfgap = hfevals[self.Nbasis] - hfevals[self.Nbasis-1]
            print "LUMO Energy: ",hfevals[self.Nbasis]
            print "HOMO Energy: ",hfevals[self.Nbasis-1]
            print "LUMO-HOMO: ",hfgap
            self.hfgap = hfgap #Important will be used to prevent certain search valleys 
        else:
            hfgap = hfevals[self.Nelec_tot] - hfevals[self.Nelec_tot-1]
            print "LUMO Energy: ",hfevals[self.Nelec_tot]
            print "HOMO Energy: ",hfevals[self.Nelec_tot-1]
            print "LUMO-HOMO: ",hfgap
            self.hfgap = hfgap #Important will be used to prevent certain search valleys 
        sys.stdout.flush()

        #Calculate rotation matrix to embedding basis from HF 1RDM (same as calculating bath orbitals)
        R, rdmCore = self.get_rotmat( self.hf1RDM_site )
        self.rdmCore = rdmCore
        self.RotationMatrix = R

        #Calculate 1/2e- terms in embedding basis the correlation potential is added
        Rcpy = np.copy(R)
        h_emb, V_emb = self.get_embedding_ham(Rcpy)
        sys.stdout.flush()
        

        return h_emb,V_emb,R,rdmCore

    def solveCorrProb(self,h_emb,V_emb,R,rdmCore):
        
        print
        print "========================================================================="
        print "STARTING CORRELATED PROBLEM:"
        print "========================================================================="
        print
        print "Active space size: ",h_emb.shape[0]
        print "No. of electrons in space: ",self.actElCount 
        print "Size of reduced space: ",R.shape[1]
       
        self.label = 1
 
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #Do embedding problem
        #Need fock term for core
        self.jk_c = []
        if(self.iformalism):
            jk_c =  self.make_h2el(self.g2e_site,rdmCore)   
            jk_c = np.dot(R.conj().T,np.dot(jk_c,R))
            h1body = h_emb + jk_c
            self.jk_c = jk_c
        else:
            '''
            ucpy = np.copy(self.u_mat)
            ucpy[:self.Nimp,:self.Nimp] = 0
            ucpy[self.Nbasis:self.Nbasis+self.Nimp,:self.Nimp] = 0
            ucpy[:self.Nimp,self.Nbasis:self.Nbasis+self.Nimp] = 0
            ucpy[self.Nbasis:self.Nbasis+self.Nimp,self.Nbasis:self.Nbasis+self.Nimp] = 0
            '''

            uemb = np.dot(R.conj().T,np.dot(self.u_mat,R))
            le = R.shape[1]-self.Nimp*2
            uemb[:self.Nimp*2,:self.Nimp*2] = np.zeros([2*self.Nimp,2*self.Nimp])
            uemb[:self.Nimp*2,self.Nimp*2:] = np.zeros([2*self.Nimp,le])
            uemb[self.Nimp*2:,:self.Nimp*2] = np.zeros([le,2*self.Nimp])
            
            h1body = h_emb + uemb                         

        #Store MU for next iteration
        gtol = 1.0e-4
        if(self.itr == 1):
            lastmu = self.startMu 
        else:
            lastmu = self.muCollection[self.itr-2]
                 
        #Remember filling = 1/2 of no. of particles per site
        if(self.muSolverSimple):    
            lastmu = self.misolver(self,[lastmu],R,h1body, V_emb,self.Nimp*self.filling*2,self.hfgap,gtol,self.corrIter,fitmu=self.fitmu)
        else:
            lastmu = self.misolver(self,lastmu,R,h1body, V_emb, self.Nimp*self.filling*2,gtol,self.corrIter,fitmu=self.fitmu)
        self.muCollection.append(lastmu)

        #Calculate the DMET site energy and density of bath
        h_emb = utils.rot1el(self.h1, R)
        if(self.iformalism):
            Efrag = self.get_impurityEnergy(h_emb+0.5*jk_c, V_emb, self.IRDM1, self.IRDM2).real
        else:
            Efrag = self.get_impurityEnergyNI(h_emb+uemb, V_emb, self.IRDM1, self.IRDM2).real

        #if(self.constrainType == 'BCS'):
        #   Efrag += self.Nimp/2*self.g2e_site[0]

        dg = self.IRDM1.diagonal()
        if(self.constrainType != 'BCS'):
            impdensity = sum(dg[:2*self.Nimp])
        else:
            impnup = sum(dg[:self.Nimp])
            impndn = self.Nimp - sum(dg[self.Nimp:self.Nimp*2])
            impdensity = impnup+impndn
            
            nb = (R.shape[1] - 2*self.Nimp)/2
            bathdensity = sum(dg[2*self.Nimp:])
            coredensity = np.trace(self.rdmCore)
                                       
        print 
        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print 'ENERGY Statistics:'
        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print
        print 'FCI variational energy: ',self.ImpurityEnergy
        print 'DMET Energy per site: ',Efrag/self.Nimp           
        print "DMET Total Energy: ",Efrag*self.Nbasis/self.Nimp
        print 'T-E-INFO      %0.4f        %0.12f'%(self.T, Efrag/self.Nimp)
        print
        print "Total density: ",sum(dg)              
        print "Density on impurity: ",impdensity
        print

        if(self.constrainType=='BCS'):
            print "Bath density: ",bathdensity," Core density: ",coredensity              
            print "Density on impurity: up = ",impnup," dn = ",impndn," total = ",impdensity
            print            

        hf1RDM_b = np.dot(R.conjugate().T,np.dot(self.hf1RDM_site - rdmCore,R))
        rdmdiff = (hf1RDM_b-self.IRDM1)[:2*self.Nimp,:2*self.Nimp]           
        impuritynorm = np.linalg.norm(rdmdiff)
        fullnorm = np.linalg.norm(hf1RDM_b-self.IRDM1) 
        critnorm = np.linalg.norm((hf1RDM_b-self.IRDM1)[:self.fitIndex,:self.fitIndex])
        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print "RDM Statistics:"
        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        print
        print "Difference in HF RDM and Corr RDM before fit: ",impuritynorm
        print "Bath + Impurity HF RDM and Corr RDM before fit: ",fullnorm
        print 

        sys.stdout.flush() 

        print "HF Impurity RDM"
        ro = utils.extractImp(self.Nimp,self.hf1RDM_site)
        utils.displayMatrix(ro)
        print

        print "I-RDM:"  
        utils.displayMatrix(self.IRDM1[:2*self.Nimp,:2*self.Nimp])
        print
        magorder = abs(self.IRDM1[1,1]-self.IRDM1[0,0])/2.  
        sz = (np.sum(np.diag(self.IRDM1)[:self.Nimp])-np.sum(np.diag(self.IRDM1)[self.Nimp:2*self.Nimp]))/self.Nimp
        nfill = np.sum(np.diag(self.IRDM1)[:2*self.Nimp])/self.Nimp
        print "T-Magnetic order:    %0.4f      %0.12f"%(self.T, magorder)
        print "T-Sz:   %.4f     %0.12f "%(self.T, sz)
        print "T-filling: %.4f      %0.12f"%(self.T, nfill)

        '''
        #Do SITE based energy calculation
        Rt = R.transpose()
        rdm2site = utils.rot2el_chem_full(self.IRDM2,Rt) 
        Efrag = self.get_impurityEnergySite(rdm1site, rdm2site)
        print 'DMET Energy per site: ',Efrag/self.Nimp           
        print "DMET Total Energy: ",Efrag*self.Nbasis/self.Nimp
        '''
        self.critnorm = critnorm
        self.Efrag = Efrag 





if __name__ == '__main__':
    h1e_site = np.zeros((4,4))
    g2e_site = np.zeros(2)
    obj = ftdmet(4,4,2,h1e_site,g2e_site,T=4)
    obj.updateParams()
    obj.displayParameters()
    RDM1 = np.random.rand(8,8)
    RDM1 = RDM1 + RDM1.T
    a,b = obj.get_rotmat(RDM1)
    print a[:,4:]
