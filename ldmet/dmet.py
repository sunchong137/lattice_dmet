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
import minimizers as mmizer
import solvers as solver

class dmet(object):
    '''
    Define the DMET class.
    '''
    def __init__(self, norb, nelec, nimp, h1e, g2e, init_uimp=None, solver_type="FCI", 
                 utol=5e-6, etol=1e-5, ntol=1e-6, MaxIter=100):
    # def __init__( self, Nbasis, Nelec_tot, Nimp, h1e_site, g2e_site, h1esoc_site=None, SolverType='FCI', u_matrix=None, 
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
        self.minimize_u_matrix = mmizer.minimizelsq # TODO replace this less efficient one.
        #self.minimize_u_matrix = mmizer.minimizeBFGS
        self.make_h2el = hf.make_h2el # TODO replace

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
            self.fitIndex = norb * 2
        else:
            self.fitIndex = nimp * 2

        #DIIS
        self.doDIIS = True 
        self.diisStart = 4
        self.diisDim = 4    

        #DAMP
        self.doDAMP = True
        self.dampStart = 4
        self.dampFactor = 0.25
        self.dampTol = 0.005

        #Zero Value for Orbital Selection
        
        #DEBUG OPTIONS
        self.debugPrintRDMDiff = True
        self.dlog = []
 
        #Mu solver
        self.muSolverSimple = True 
        if(self.muSolverSimple):    
            self.misolver = solver.microIterationSimple
        else: 
            self.misolver = solver.microIteration
        self.startMuDefault = True
        if self.startMuDefault:
            self.startMu = self.g2e[0] * (1 - 2*self.filling) 

        self.hfsolver = hf.hf_hubbard_calc 
        self.mhfsolver = hf.hf_hubbard_calc
 


    def solveFitProb(self, dc):

        print()
        print("=========================================================================")
        print("STARTING FITTING PROBLEM:")
        print("=========================================================================")

        self.label = 2        

          

        mintol = 1.0e-6
        maxiterformin = 200
        
        self.minimize_u_matrix(self, self.RotationMatrix, mintol, maxiterformin)

        #DIIS       
        if(self.doDIIS):
            vcor = self.umat_imp.flatten() 
            vcor_new = self.umat_imp.flatten()
            diffcorr = vcor_new-vcor
            skipDiis = not (self.itr>=self.diisStart and np.linalg.norm(diffcorr) < 0.01*len(diffcorr) and self.critnorm < 1e-2*len(diffcorr))
            pvcor, _, _ = dc.Apply(vcor_new,diffcorr, Skip = skipDiis)
            if(not skipDiis):
                print("DIIS GUESS USED")
                self.u_mat_new = self.replicate_u_matrix(self.array2matrix(pvcor))  

        #DAMP
        if(self.doDAMP):
            vcor = self.umat_imp.flatten()  
            vcor_new = self.umat_imp.flatten()
            diffcorr = vcor_new-vcor
            if(self.itr>=self.dampStart and np.linalg.norm(diffcorr)/len(diffcorr) < self.dampTol):
                self.u_mat_new = self.replicate_u_matrix(self.array2matrix(diffcorr*self.dampFactor + vcor))


        #calculate frobenius norm of difference between old and new u-matrix (np.linalg.norm)
        udiff = utils.extractImp(self.Nimp,self.u_mat - self.u_mat_new)

        #update u-matrix with most recent guess
        self.u_mat = self.u_mat_new     
        eo = utils.extractImp(self.Nimp,self.u_mat)
        
        #Remove diagonal contribution 
        nomdg = np.array([udiff[i,j] for i in range(2*self.Nimp) for j in range(i+1,2*self.Nimp)])
        nomdg_norm = np.linalg.norm(nomdg)
        
        #Also remove SOC diagonal contribution
        nosocdg = np.array([udiff[i,j] for i in range(2*self.Nimp) for j in range(i+1,2*self.Nimp) if (j-self.Nimp != i)])
        nosocdg_norm = np.linalg.norm(nosocdg)

        #u_mat_diff = np.linalg.norm(nodgs) + np.linalg.norm(dgs)
        u_mat_diff = nomdg_norm

        print()
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('UMATRIX Statistics')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('Current value of difference in UMATRIX = ',u_mat_diff)
        print('Off-diagonal difference in UMATRIX = ',nomdg_norm)
        print('SOC difference w/o diagonal = ',nosocdg_norm)
        print('Full difference in UMATRIX = ',np.linalg.norm(udiff))
        print('Time to find new UMATRIX = ',time_for_umat)
        print()
        print('New UMATRIX: ')
        utils.displayMatrix(eo)
        print('T-Umatrix: alpha= %.12f     beta=%.12f'%(eo[0,0], eo[self.Nimp, self.Nimp]))

        return u_mat_diff

   
    def solve_groundstate( self ):

        self.updateParams()
        self.displayParameters()

        #DMET loop to converge u-matrix
        #u tolerances
        tol        = self.utol * self.Nimp
        u_mat_diff = tol+1
        minutol    = self.utol * self.Nimp #Atleast this tolerance is needed in u no matter what

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
            dc = diis.FDiisContext(self.diisDim)         


        self.initializeFock()       

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Calculate parameter space size << IMPORTANT
        self.paramss = np.zeros([self.fitIndex,self.fitIndex]).flatten() 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               
 
        while( self.itr < self.dmetitrmax ):

            self.itr += 1
            
            print()
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Starting DMET iteration: ",self.itr)
            print("Date time: ", str(datetime.datetime.now())[:-7])
                           

            if(not restartStatus):
                #DMET 1: Generate bath
                h_emb, v_emb, R, rdmCore = routine.gen_emb_ham(self.h1e, self.umat_imp) 

                #DMET 2: Do correlated problem
                corrpot.solve_corr(h_emb, v_emb, R, rdmCore)
                self.ediff = abs(self.Efrag-Efrag0)/self.Nimp
                Efrag0 = self.Efrag
                                  
                #DMET 3: Do fitting
                u_mat_diff = self.solveFitProb(dc)            
            else:           
                restartStatus = False
                
                if(self.label == 2):
                    #DMET 1: Generate bath
                    h_emb,v_emb,R,rdmCore = routine.gen_emb_ham(self.h1e, self.umat_imp) 

                    #DMET 2: Do correlated problem
                    corrpot.solve_corr(h_emb, v_emb, R, rdmCore)
                    self.ediff = abs(self.Efrag-Efrag0)/self.Nimp
                    Efrag0 = self.Efrag
  
                if(self.label >= 1): 
                    #DMET 3: Do fitting
                    u_mat_diff = self.solveFitProb(dc)            
 
            ##############################################################################################
            # TOLERANCE CHECKING
            ##############################################################################################

            print()
            print("=========================================================================")
            print('Convergence Criteria Observables:')
            print("=========================================================================")
            print('*UMatrix Difference = ',u_mat_diff)
            print('*RDM Difference (Bath Fit=%d) = %10.6e' %(self.fitBath,self.critnorm))
            print('*ENERGY Difference of Fragment = ',self.ediff)
            print()
              
        
            rdmcriteria = False
            umatrixcriteria = False
            energycriteria = False

            if(self.critnorm < ntol):
                rdmcriteria = True

            if(u_mat_diff < minutol):
                umatrixcriteria = True

            if(self.ediff < etol):
                energycriteria = True

            print('Convergence Critera: UMATRIX=%d RDM=%d ENERGY=%d' %(umatrixcriteria,rdmcriteria,energycriteria))
            print("=========================================================================")
            print('FINISHED DMET ITERATION ',self.itr)
            print()

    
            #If fit is super accurate then no point in continuing with u_matrix update      
            if(umatrixcriteria):
                break
       
        if( self.itr >= self.dmetitrmax ):
            print("UMATRIX did not converge in less than", self.dmetitrmax, "iterations")
            
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

        ##############################################################
        #Checkpoint again
        ##############################################################
        self.checkPoint()
        u_mat_imp = utils.extractImp(self.Nimp,self.u_mat)
        return self.Efrag, u_mat_imp, self.IRDM1, self.IRDM2

    #########################################################################



