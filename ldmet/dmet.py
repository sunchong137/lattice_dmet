# Copyright 2024 Lattice DMET developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from ldmet import routine, corrpot
from ldmet.helpers import utils, diis

class dmet(object):
    '''
    Define the DMET class.
    '''
    def __init__(self, norb, nelec, nimp, h1e, g2e, init_uimp=None, solver_type="FCI", 
                 utol=5e-6, etol=1e-5, ntol=1e-6, MaxIter=100):
    # def __init__( self, Nbasis, Nelec_tot, nimp, h1e_site, g2e_site, h1esoc_site=None, SolverType='FCI', u_matrix=None, 
    #                     mtype = np.float64, ctype = 'SOC', globalMu=None, utol=5e-6, etol=1e-5, ntol=1e-6):
        '''
        Initialization with parameters defining the system. 
        The simulation breaks the S^2 symmetry. 
        Args:
            norb: int, number of physical orbitals, or sites.
            nelec: int, number of total electrons.
            nimp: int, number of impurity orbitals.
            h1e: 2D array, one-body Hamiltonian matrix.
            h2e: 4D array, two-body Hamiltonian tensor. 
        Kwargs:
            init_uimp: if not None, 2D or 3D array.
            solver_type: string, points to the solvers.
            global_mu: global chemical potential.
            utol: convergence threshold for correlation potential.
            etol: convergence threshold for DMET energy.
            ntol: convergence threshold for electron number.
           

        TODO: make norb and nimp tuples for 2D.
        TODO: check if we should make h1e and h2e tuples.
        TODO: see if global mu can be deleted.
        '''
        self.norb = norb   
        self.nelec_tot  = nelec 
        self.h1e = h1e
        self.g2e = g2e
        self.nimp = nimp 
        self.solver_type = solver_type
        if init_uimp is None:
            self.umat_imp = np.zeros((self.nimp, self.nimp))
        else:
            self.umat_imp = init_uimp
        # self.init_umat(init_uimp)
        # self.umat_new = np.copy(self.umat)
        self.filling = 0.5 * (nelec / norb) 

        self.utol  = utol
        self.etol  = etol
        self.ntol  = ntol
        self.ZeroValue = 1.0e-9 # for orbital selection

        self.dmetitrmax = MaxIter
        self.corrIter = 40

        self.interact_form = True # Interacting Formalism
        self.fitBath = False 
        #Interacting/Non-Interacting Formalism
        self.paramss = 0
        if(self.fitBath):
            self.fitIndex = nimp * 2
        else:
            self.fitIndex = nimp

        #DIIS
        self.doDIIS = True 
        self.diisStart = 4
        self.diisDim = 4    

        #DEBUG OPTIONS
        self.debugPrintRDMDiff = True
        self.dlog = []
 
        self.startMuDefault = True
        if self.startMuDefault:
            self.startMu = self.g2e[0] * (1 - 2*self.filling) 

    def solveFitProb(self, diis_solver):

        print()
        print("=========================================================================")
        print("STARTING FITTING PROBLEM:")
        print("=========================================================================")

        mintol = 1.0e-6
        maxiterformin = 200
        
        umat_new = corrpot.minimize_lsq(self, rotmat, mintol, maxiterformin)

        #DIIS       
        if(self.doDIIS):
            vcor = self.umat_imp.flatten() 
            vcor_new = self.umat_imp.flatten()
            diffcorr = vcor_new - vcor
            skipDiis = not (self.itr >= self.diisStart and np.linalg.norm(diffcorr) < 0.01*len(diffcorr) and self.critnorm < 1e-2*len(diffcorr))
            pvcor, _, _ = diis_solver.Apply(vcor_new, diffcorr, Skip=skipDiis)
            if(not skipDiis):
                print("DIIS GUESS USED")
                self.umat_new = routine.expand_u(utils.vec2mat(pvcor))

        #calculate frobenius norm of difference between old and new u-matrix (np.linalg.norm)
        udiff = utils.extractImp(self.nimp, self.u_mat - umat_new)

        #update u-matrix with most recent guess
        self.u_mat = umat_new     
        eo = utils.extractImp(self.nimp, self.u_mat)
        
        #Remove diagonal contribution #TODO rewrite
        nomdg = np.array([udiff[i,j] for i in range(2*self.nimp) for j in range(i+1,2*self.nimp)])
        nomdg_norm = np.linalg.norm(nomdg)
        u_mat_diff = nomdg_norm

        return u_mat_diff

   
    def solve_groundstate(self):
        #DMET loop to converge u-matrix
        #u tolerances
        tol        = self.utol * self.nimp
        u_mat_diff = tol + 1
        minutol    = self.utol * self.nimp #Atleast this tolerance is needed in u no matter what

        #rho tolerances
        ntol       = self.ntol * self.fitIndex
        self.critnorm   = ntol + 1

        #energy tolerances
        Efrag0     = 0.0
        etol       = self.etol #Energy tolerance in per site        
 
        lastmu = 0.0 
        self.itr = 0
        self.muCollection = []

        if(self.doDIIS):
            diis_solver = diis.FDiisContext(self.diisDim)         

        self.paramss = np.zeros([2, self.fitIndex, self.fitIndex]).flatten() 

        while(self.itr < self.dmetitrmax):

            self.itr += 1
            
            print()
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Starting DMET iteration: ",self.itr)

            #DMET 1: Generate bath
            h_emb, v_emb, R, rdmCore = routine.gen_emb_ham(self.h1e, self.umat_imp) 

            #DMET 2: Do correlated problem
            energy_imp, rdm1_emb, rdm2_emb = corrpot.solve_corr(h_emb, v_emb, R, rdmCore)
            self.ediff = abs(energy_imp-Efrag0)/self.nimp
            Efrag0 = energy_imp
                                
            #DMET 3: Do fitting
            u_mat_diff = self.solveFitProb(diis_solver)            
          
            print()
            print("=========================================================================")
            print('Convergence Criteria Observables:')
            print("=========================================================================")
            print('*UMatrix Difference = ',u_mat_diff)
            print('*RDM Difference (Bath Fit=%d) = %10.6e' %(self.fitBath, self.critnorm))
            print('*ENERGY Difference of Fragment = ',self.ediff)
            print()
              

            print("Umatrix diff: {:%0.4e}".format(u_mat_diff))
            # print('Convergence Critera: UMATRIX=%d RDM=%d ENERGY=%d' %(umatrixcriteria, rdmcriteria, energycriteria))
            print("=========================================================================")
            print('FINISHED DMET ITERATION ', self.itr)
            print()

    
            #If fit is super accurate then no point in continuing with u_matrix update      
            if(u_mat_diff < minutol):
                break
       
        if(self.itr >= self.dmetitrmax ):
            print("UMATRIX did not converge in less than", self.dmetitrmax, "iterations")
            
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
        u_mat_imp = utils.extractImp(self.nimp,self.u_mat)
        return energy_imp, u_mat_imp, rdm1_emb, rdm2_emb




