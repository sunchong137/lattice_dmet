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
# from ldmet.helpers import math_helpers
from ldmet import routine
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
            
        # TODO clean the following
        self.doRestart = False
        self.chkPoint = True 
        self.chkPointInterval = 1
        self.chkPointFile = 'chkpoint.hdf5' 
        self.resFile = self.chkPointFile
        self.tableFile = 'tmp.txt'


    def get_impurityEnergy( self, h1_emb, V2_emb, corr1RDM, corr2RDM):

        #Note the 2RDM is expected to be in chemistry format
        #Note here Nimp is again specified in orbital size and NOT spon-orbital size 
        #Calculate Site Energy
        ld = h1_emb.shape[0]
       
        E1 = 0.0
        E2 = 0.0 
        Eclust = 0.0
        for orb1 in range(2*self.Nimp):
            for orb2 in range(ld):
                E1 += corr1RDM[ orb1, orb2 ]*(h1_emb[ orb2, orb1 ])
                for orb3 in range(ld):
                    for orb4 in range(ld):
                        E2 += 0.5*corr2RDM[ orb1, orb2, orb3, orb4 ]*V2_emb[ orb2, orb1, orb4, orb3]                   
        #print "1e ",E1," 2e ",E2
        return E1+E2

    def get_impurityEnergyNI( self, h1_emb, V2_emb, corr1RDM, corr2RDM):

        #Note the 2RDM is expected to be in chemistry format
        #Note here Nimp is again specified in orbital size and NOT spon-orbital size 
        #Calculate Site Energy
        ld = h1_emb.shape[0]
       
        E1 = 0.0
        E2 = 0.0 
        Eclust = 0.0
        for orb1 in range(2*self.Nimp):
            for orb2 in range(ld):
                E1 += corr1RDM[ orb1, orb2 ]*(h1_emb[ orb2, orb1 ])

        lis = 2*self.Nimp
        for orb1 in range(lis):
            for orb2 in range(lis):
                for orb3 in range(lis):
                    for orb4 in range(lis):
                        E2 += corr2RDM[ orb1, orb2, orb3, orb4 ]*V2_emb[ orb2, orb1, orb4, orb3]                   

        #print "1e ",E1," 2e ",E2
        return E1+0.5*E2
    #####################################################################

    def get_impurityEnergySite( self, corr1RDM, corr2RDM):

         #Note here Nimp is again specified in orbital size and NOT spon-orbital size 
        #Calculate Site Energy
        ld = self.h1.shape[0]
       
        E1 = 0.0
        E2 = 0.0 
        Eclust = 0.0
    
        impsites = range(self.Nimp) + range(self.Nbasis,self.Nbasis+self.Nimp)
    
        if(self.g2e_site.shape[0]!=1):
            for orb1 in impsites:
                for orb2 in range(ld):
                    E1 += corr1RDM[ orb1, orb2 ]*(self.h1[ orb2, orb1 ])
                    for orb3 in range(ld):
                        for orb4 in range(ld):
                            E2 += 0.5*corr2RDM[ orb1, orb2, orb3, orb4 ]*self.V2[ orb2, orb1, orb4, orb3]
        else:
            for orb1 in impsites:
                for orb2 in range(ld):
                    E1 += corr1RDM[ orb1, orb2 ]*(self.h1[ orb2, orb1 ])
        
            for orb1 in range(self.Nimp):   
                E2 += 0.5*corr2RDM[ orb1, orb1, orb1, orb1 ]*self.g2e_site[0]
                E2 += 0.5*corr2RDM[ orb1, orb1, self.Nbasis+orb1, self.Nbasis+orb1 ]*self.g2e_site[0]
                E2 += 0.5*corr2RDM[ self.Nbasis+orb1, self.Nbasis+orb1, orb1, orb1 ]*self.g2e_site[0]
                E2 += 0.5*corr2RDM[ self.Nbasis+orb1, self.Nbasis+orb1, self.Nbasis+orb1, self.Nbasis+orb1 ]*self.g2e_site[0]

        #print "1e ",E1," 2e ",E2
        return E1+E2

    #####################################################################
    def getOrderParams(self,Nimp,rdm,rdm2,R):
                
        #Calculate Double Occupancy
        dbl_occ = 0.0
        for orb1 in range(self.Nimp):
            dbl_occ += (rdm2[ orb1, orb1, self.Nimp+orb1, self.Nimp+orb1] + rdm2[self.Nimp+orb1,self.Nimp+orb1,orb1,orb1])
        dbl_occ /= self.Nimp*2
       
        #Anti-feromagnetic order
        afmorder = 0.0
        for orb1 in range(self.Nimp-1):
            afmorder += 0.5*(rdm2[orb1, orb1, self.Nimp+orb1+1, self.Nimp+orb1+1] +     
                         rdm2[orb1+1, orb1+1, self.Nimp+orb1, self.Nimp+orb1])    
     
        #dbl_occ = dbl_occ / Nimp 
        return dbl_occ, afmorder        

    #####################################################################
    def matrix2array( self, mat ):
        
        #Expects input in [2*Nimp,2*Nimp] format
        
        if(self.constrainType=='BCS'):
            array = np.array([mat[i,j] for i in range(2*self.Nimp) for j in range(i,2*self.Nimp)],dtype=np.float64)

            #array = ([mat[i,j] for i in range(self.Nimp) for j in range(i,self.Nimp)])
            #array += ([mat[i,j+self.Nimp] for i in range(self.Nimp) for j in range(self.Nimp)])
            array = np.array(array,dtype=np.float64)

        elif(self.constrainType == 'SOC'):
            array = [mat[i,j].real for i in range(2*self.Nimp) for j in range(i,2*self.Nimp)]
            #Add the imaginary parts only for off-diagonal pieces
            #array += [mat[i,j].imag for i in range(2*self.Nimp) for j in range(i+1,2*self.Nimp)]
            array = np.array(array,dtype=np.float64)

        elif(self.constrainType == 'SOCTR'):
            #SOC with Time-Reversal Symmetry
            #Same spin sector must be h and h*
            #In SOC sector must be anti-symmetric 
            array = [mat[i,i].real for i in range(self.Nimp)]
            array += [mat[i,j].real for i in range(self.Nimp) for j in range(i+1,self.Nimp)] #keep off-diagonal separate
            array += [mat[i,j].imag for i in range(self.Nimp) for j in range(i+1,self.Nimp)] #Since diagonal can't be imaginary
            array += [mat[i,j].real for i in range(self.Nimp) for j in range(i+1+self.Nimp,2*self.Nimp)]
            array += [mat[i,j].imag for i in range(self.Nimp) for j in range(i+1+self.Nimp,2*self.Nimp)]        
            array = np.array(array,dtype=np.float64)

        elif(self.constrainType == 'UHF'):        
            p = 0
            ls = self.Nimp*(self.Nimp+1)/2
            array = np.zeros([self.Nimp*(self.Nimp+1)],dtype=np.float64)
            for i in range(self.Nimp):
                for j in range(i,self.Nimp):
                    array[p] = mat[i,j]
                    array[p+ls] = mat[i+self.Nimp,j+self.Nimp]
                    p+=1

        elif(self.constrainType == 'RHF'):
            p = 0
            ls = self.Nimp*(self.Nimp+1)/2
            array = np.zeros([self.Nimp*(self.Nimp+1)/2],dtype=np.float64)
            for i in range(self.Nimp):
                for j in range(i,self.Nimp):
                    array[p] = mat[i,j]
                    p+=1
        else:
            raise RunTimeError("Invalid constraint option") 
    
        return array

    #####################################################################

    def array2matrix( self, array ):

        mat = np.zeros([self.Nimp*2,self.Nimp*2],dtype=self.mtype)
        
        if(self.constrainType == 'BCS'):

            p = 0            
            for i in range(2*self.Nimp):
                for j in range(i,2*self.Nimp):
                    mat[i,j] = array[p]
                    p += 1

            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] *= 0.5
            
        elif(self.constrainType == 'SOC'):
            p = 0
            for i in range(2*self.Nimp):
                for j in range(i,2*self.Nimp):
                    mat[i,j] = array[p]
                    p += 1

            for i in range(2*self.Nimp):                
                for j in range(i+1,2*self.Nimp):
                    mat[i,j] = complex(mat[i,j],array[p])
                    p += 1

            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] *= 0.5

        elif(self.constrainType == 'SOCTR'):
        
            p = 0
            #In same spin sector we must have h_up = h_dn*
            for i in range(self.Nimp):
                mat[i,i] = complex(array[p],0.0)
                mat[i+self.Nimp,i+self.Nimp] = complex(array[p],0.0)
                p += 1

            offset = self.Nimp*(self.Nimp-1)/2
            for i in range(self.Nimp):
                for j in range(i+1,self.Nimp):
                    mat[i,j] = complex(array[p],array[p+offset])
                    mat[i+self.Nimp,j+self.Nimp] = complex(array[p],-array[p+offset])
                    p += 1
        
            ''' 
            for i in range(self.Nimp):              
            for j in range(i,self.Nimp):
                    mat[i,j] = complex(mat[i,j],array[p])
                mat[i+self.Nimp,j+self.Nimp] = complex(mat[i+self.Nimp,j+self.Nimp],-array[p])
                p += 1
            '''
            p += offset
    
            #In SOC sector must have anti-symmetry
            offset = self.Nimp*(self.Nimp-1)/2
            for i in range(self.Nimp):
                mat[i,i+self.Nimp] = 0.0
                for j in range(i+1+self.Nimp,2*self.Nimp):
                    mat[i,j] = complex(array[p],array[p+offset])    
                    mat[j-self.Nimp,i+self.Nimp] = complex(-array[p],-array[p+offset])  
                    p += 1
    
            '''
            for i in range(self.Nimp):  
                for j in range(i+1+self.Nimp,2*self.Nimp):
                    mat[i,j] = complex(mat[i,j],array[p])
                    mat[j,i] = complex(mat[j,i],-array[p])  
                    p += 1
            '''
            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] *= 0.5
 
        elif(self.constrainType == 'UHF'):
        
            ls = self.Nimp*(self.Nimp+1)/2     
            p = 0
            for i in range(self.Nimp):
                for j in range(i,self.Nimp):
                    mat[i,j] = array[p]
                    mat[i+self.Nimp,j+self.Nimp] = array[p+ls]
                    p += 1
                    
            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] *= 0.5        
                
        elif(self.constrainType == 'RHF'):
            
            ls = self.Nimp*(self.Nimp+1)/2     
            p = 0
            for i in range(self.Nimp):
                for j in range(i,self.Nimp):
                    mat[i,j] = array[p]
                    mat[i+self.Nimp,j+self.Nimp] = array[p]
                    p += 1
 
            mat = mat + mat.conj().T 
            for i in range(2*self.Nimp):
                mat[i,i] *= 0.5            
            
        else:
            raise RuntimeError("Invalid constraint option")

        return mat



    #####################################################################
    
    def checkPoint(self,chkpfile = None):
        
        if(chkpfile is None):
            chkpfile = self.chkPointFile

        def writeComplex(f,datasetname,cobj):
            #if(np.iscomplexobj(cobj):
            robj = [x for x in cobj.real.flatten()]
            iobj = [x for x in cobj.imag.flatten()]
            dset = f.create_dataset(datasetname+".s",data=cobj.shape)       
            dset = f.create_dataset(datasetname+".r",data=robj)
            dset = f.create_dataset(datasetname+".i",data=iobj)
            #else:
            #   dset = f.create_dataset(datasetname,data=cobj)
    
        def writeReal(f,datasetname,cobj):
            obj = [x for x in cobj.flatten()]   
            dset = f.create_dataset(datasetname+'.s',data=cobj.shape)
            dset = f.create_dataset(datasetname+'.r',data=obj)
    
        f = h5py.File(chkpfile,"w")
        umi = utils.extractImp(self.Nimp,self.u_mat)
        
        if(np.iscomplexobj(umi[0][0]) ):
            writeComplex(f,"umatrix",umi)
        else:
            writeReal(f,"umatrix",umi)

        if(np.iscomplexobj(self.IRDM1[0][0]) ):
            writeComplex(f,"rdm1",self.IRDM1)
        else:
            writeReal(f,"rdm1",self.IRDM1)
        
        if(np.iscomplexobj(self.IRDM2[0][0][0][0]) ):
            writeComplex(f,"rdm2",self.IRDM2)
        else:
            writeReal(f,"rdm2",self.IRDM2)
        
        if(np.iscomplexobj(self.RotationMatrix[0][0]) ):
            writeComplex(f,"rotationmatrix",self.RotationMatrix)
        else:
            writeReal(f,"rotationmatrix",self.RotationMatrix)

        if(np.iscomplexobj(self.rdmCore[0][0]) ):
            writeComplex(f,"rdmcore",self.rdmCore)
        else:
            writeReal(f,"rdmcore",self.rdmCore)
      
        if(np.iscomplexobj(self.h1[0][0])):
            writeComplex(f,"h1",self.h1)
        else:
            writeReal(f,"h1",self.h1)
                      
        writeReal(f,"globalMu",self.globalMu)
 
        fmtd = np.array(self.muCollection)
        writeReal(f,"muc",fmtd)
       
        dset4 = f.create_dataset("simparams",data=[self.Nimp,self.Nbasis,self.fitIndex])
        dset4 = f.create_dataset("simstate",data=[self.itr,self.label])
        dset5 = f.create_dataset("observables",data=[self.critnorm,self.Efrag,self.ediff])
        f.close()   

    def restart(self,fname = None):

        if(fname is None):
            fname = self.resFile
        
        print("Attempting restart with: ",fname)

        if(os.path.isfile(fname)): 
            print('===================================================')
            print('Restarting')
            print('===================================================')
         
            f = h5py.File(fname,'r')

            def readMatrix(basename):
                s = f.get(basename+'.s')[:]
                l = np.prod(s)
                try:
                    mmr = f.get(basename + '.r')[:]
                    mmi = f.get(basename + '.i')[:]
                    mm = np.array(map(lambda a,b: a+1j*b,zip(mmr,mmi)))
                except TypeError:
                    mm = np.array(f.get(basename + '.r')[:]) 
                mm = np.reshape(mm,s)
                return mm
    
            um = readMatrix('umatrix')
            self.u_mat     = self.replicate_u_matrix(um)
            self.u_mat_new = np.copy( self.u_mat )     
        
            s = f.get('muc.s')[:] 
            l = np.prod(s)
            self.muCollection = list(f.get('muc.r')[:])

            self.itr = int(f.get('simstate')[0])
            self.label = int(f.get('simstate')[1])

            self.critnorm = f.get('observables')[0]
            self.Efrag = f.get('observables')[1]
            self.ediff = f.get('observables')[2]
            
            self.IRDM1 = readMatrix('rdm1')
            self.IRDM2 = readMatrix('rdm2')
            self.rdmCore = readMatrix('rdmcore')
            self.RotationMatrix = readMatrix('rotationmatrix')
            self.globalMu = readMatrix('globalMu')
 
            f.close()   
 
            print('Iteration: ',self.itr)
            print('Label: ',self.label)
            print('Correlation Potential: ')
            eo = utils.extractImp(self.Nimp,self.u_mat)
            utils.displayMatrix(eo)
            print('===================================================' )

            #Do adjustments for correct restart
            '''
                label = 0 means finished constructing
                label = 1 means it finished impurity problem but terminated right after.
                label = 2 means it finished fitting problem but terminated right after.
            '''
            if(self.label == 1):
                self.itr -= 1 #Loading will do += 1

            return True

        return False  


    def generateBath(self):
            
        print()
        print("=========================================================================")
        print("STARTING MF PROBLEM:")
        print("=========================================================================")
        print()

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
            self.hf1RDM_site, hforbs, hfE, hfevals = self.hfsolver(self.Nelec_tot, h_site_mod, self.g2e_site, itrmax = self.mfIter, needEnergy = True, assertcheck = False, doDIIS = True, hfoscitr = self.oscIter)
        else:
            self.globalMu,self.hf1RDM_site, hforbs, hfE, hfevals = self.hfsolver(self.Nelec_tot, self.h1, self.u_mat, self.g2e_site, needEnergy = True, display=True,muFind = True)

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
        print("HF Time: ",hftime)
        print("HF Energy: ",hfE)
        print("Hartree Fock Energy per site: ", hfE/self.Nbasis)
        print("Trace of RDM: ", self.hf1RDM_site.trace())
        print("Target trace of RDM: ", self.Nelec_tot)

        if(self.constrainType=='BCS'):
            hfgap = hfevals[self.Nbasis] - hfevals[self.Nbasis-1]
            print("LUMO Energy: ",hfevals[self.Nbasis])
            print("HOMO Energy: ",hfevals[self.Nbasis-1]) 
            print("LUMO-HOMO: ",hfgap)
            self.hfgap = hfgap #Important will be used to prevent certain search valleys 
        else: 
            hfgap = hfevals[self.Nelec_tot] - hfevals[self.Nelec_tot-1]
            print("LUMO Energy: ",hfevals[self.Nelec_tot])
            print("HOMO Energy: ",hfevals[self.Nelec_tot-1]) 
            print("LUMO-HOMO: ",hfgap)
            self.hfgap = hfgap #Important will be used to prevent certain search valleys 

        #Calculate rotation matrix to embedding basis from HF 1RDM (same as calculating bath orbitals)
        R = routine.gen_rotmat_from_mix(self.norb, self.nimp, self.hf1RDM_site) # TODO change the name
        rdmCore = routine.get_core() # TODO implement
        self.rdmCore = rdmCore
        self.RotationMatrix = R

        #Calculate 1/2e- terms in embedding basis the correlation potential is added
        Rcpy = np.copy(R)
        h_emb, V_emb = routine.gen_ham_embedding(self.h1e, self.g2e, self.RotationMatrix, self.nimp, iform=self.interact_form)  
        sys.stdout.flush() 

        return h_emb,V_emb,R,rdmCore

    def solveCorrProb(self,h_emb,V_emb,R,rdmCore):
    
        print()
        print("=========================================================================")
        print("STARTING CORRELATED PROBLEM:")
        print("=========================================================================")
        print()
        print("Active space size: ",h_emb.shape[0])
        print("No. of electrons in space: ",self.actElCount)
        print("Size of reduced space: ",R.shape[1])
       
        self.label = 1
 
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #Do embedding problem
        #Need fock term for core
        self.jk_c = []
        if(self.interact_form):
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
            lastmu = self.misolver(self,[lastmu],R,h1body, V_emb,self.Nimp*self.filling*2,self.hfgap,gtol,self.corrIter)
        else:
            lastmu = self.misolver(self,lastmu,R,h1body, V_emb, self.Nimp*self.filling*2,gtol,self.corrIter)
        self.muCollection.append(lastmu)

        #Calculate the DMET site energy and density of bath
        if(self.interact_form):
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
                                       
        print()
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('ENERGY Statistics:')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print()
        print('FCI variational energy: ',self.ImpurityEnergy)
        print('DMET Energy per site: ',Efrag/self.Nimp)    
        print("DMET Total Energy: ",Efrag*self.Nbasis/self.Nimp)
        print()
        print("Total density: ",sum(dg))
        print("Density on impurity: ",impdensity)
        print()

        if(self.constrainType=='BCS'):
            print("Bath density: ",bathdensity," Core density: ",coredensity)
            print("Density on impurity: up = ",impnup," dn = ",impndn," total = ",impdensity)
            print()

        hf1RDM_b = np.dot(R.conjugate().T,np.dot(self.hf1RDM_site - rdmCore,R))
        rdmdiff = (hf1RDM_b-self.IRDM1)[:2*self.Nimp,:2*self.Nimp]           
        impuritynorm = np.linalg.norm(rdmdiff)
        fullnorm = np.linalg.norm(hf1RDM_b-self.IRDM1) 
        critnorm = np.linalg.norm((hf1RDM_b-self.IRDM1)[:self.fitIndex,:self.fitIndex])
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print("RDM Statistics:")
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print()
        print("Difference in HF RDM and Corr RDM before fit: ",impuritynorm)
        print("Bath + Impurity HF RDM and Corr RDM before fit: ",fullnorm)
        print()

        # sys.stdout.flush() 

        print("HF Impurity RDM")
        ro = utils.extractImp(self.Nimp,self.hf1RDM_site)
        utils.displayMatrix(ro)
        print()

        print("I-RDM:")
        utils.displayMatrix(self.IRDM1[:2*self.Nimp,:2*self.Nimp])  

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
        

    def solveFitProb(self, dc):

        print()
        print("=========================================================================")
        print("STARTING FITTING PROBLEM:")
        print("=========================================================================")

        self.label = 2        

        sys.stdout.flush() 

        mintol = 1.0e-6
        maxiterformin = 200
        
        t1 = time.time()
        self.minimize_u_matrix(self, self.RotationMatrix, mintol, maxiterformin)
        t2 = time.time()
        time_for_umat = t2 - t1

        #DIIS       
        if(self.doDIIS):
            vcor = self.matrix2array(utils.extractImp(self.Nimp,self.u_mat))
            vcor_new = self.matrix2array(utils.extractImp(self.Nimp,self.u_mat_new))
            diffcorr = vcor_new-vcor
            skipDiis = not (self.itr>=self.diisStart and np.linalg.norm(diffcorr) < 0.01*len(diffcorr) and self.critnorm < 1e-2*len(diffcorr))
            pvcor, _, _ = dc.Apply(vcor_new,diffcorr, Skip = skipDiis)
            if(not skipDiis):
                print("DIIS GUESS USED")
                self.u_mat_new = self.replicate_u_matrix(self.array2matrix(pvcor))  

        #DAMP
        if(self.doDAMP):
            vcor = self.matrix2array(utils.extractImp(self.Nimp,self.u_mat))
            vcor_new = self.matrix2array(utils.extractImp(self.Nimp,self.u_mat_new))
            diffcorr = vcor_new-vcor
            if(self.itr>=self.dampStart and np.linalg.norm(diffcorr)/len(diffcorr) < self.dampTol):
                self.u_mat_new = self.replicate_u_matrix(self.array2matrix(diffcorr*self.dampFactor + vcor))


        #calculate frobenius norm of difference between old and new u-matrix (np.linalg.norm)
        udiff = utils.extractImp(self.Nimp,self.u_mat - self.u_mat_new)

        #update u-matrix with most recent guess
        self.u_mat = self.u_mat_new     
        eo = utils.extractImp(self.Nimp,self.u_mat)
        
        ''' 
        dgs = np.array([x for x in udiff.diagonal()])
        dgu = np.average(dgs[:self.Nimp])
        dgd = np.average(dgs[self.Nimp:])
        dgs[:self.Nimp] = dgs[:self.Nimp] - dgu
        dgs[self.Nimp:] = dgs[self.Nimp:] - dgd
        '''

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

    def initializeFock(self):
        
        #Do one set of HF calculations
        if (False): 
            print()
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Zero-th HF iteration: ")
            
            h_site_mod = self.h1 + self.u_mat 
            self.hf1RDM_site, hforbs, hfE, hfevals = hf.hf_calc(self.Nelec_tot, h_site_mod, self.g2e_site, itrmax = self.mfIter, needEnergy = True, assertcheck = False, doDIIS = True, hfoscitr = self.oscIter)

            self.fock2e = hf.make_h2el(self.g2e_site, self.hf1RDM_site)

            print("HF Energy: ",hfE)
            print("Hartree Fock Energy per site: ", hfE/self.Nbasis)   
            print("Trace of RDM: ", self.hf1RDM_site.trace())
         
            hfgap = hfevals[self.Nelec_tot] - hfevals[self.Nelec_tot-1]
            print("LUMO Energy: ",hfevals[self.Nelec_tot])
            print("HOMO Energy: ",hfevals[self.Nelec_tot-1]) 
            print("LUMO-HOMO: ",hfgap)

            import transform        
            transform.makePlot(h_site_mod,self.Nimp,self.Nelec_tot,'save_hf.png') 
            print( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        else:
            self.fock2e = 0
        
   
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

        restartStatus = False       
        if(self.doRestart):
            restartStatus = self.restart()

        self.initializeFock()       

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Calculate parameter space size << IMPORTANT
        self.paramss = len(self.matrix2array(np.zeros([self.fitIndex,self.fitIndex])))      
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #Prepare table file
        if(not os.path.exists(self.tableFile)):
            ftbl = open(self.tableFile,'a')
            print >>ftbl,'%4s\t%16s\t%16s\t%16s\t%16s' %('ITR.','DMET Energy','Energy Diff.','RDM Diff.','UMatrix Diff.')
            ftbl.close()           
 
        while( self.itr < self.dmetitrmax ):

            self.itr += 1
            
            print()
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("Starting DMET iteration: ",self.itr)
            print("Date time: ", str(datetime.datetime.now())[:-7])
            sys.stdout.flush()              

            if(not restartStatus):
                #DMET 1: Generate bath
                h_emb,v_emb,R,rdmCore = self.generateBath()

                #DMET 2: Do correlated problem
                self.solveCorrProb(h_emb,v_emb,R,rdmCore)
                self.ediff = abs(self.Efrag-Efrag0)/self.Nimp
                Efrag0 = self.Efrag
       
                #Checkpoint 
                if(self.chkPoint and self.itr%self.chkPointInterval == 0):
                    self.checkPoint()
                                       
                #DMET 3: Do fitting
                u_mat_diff = self.solveFitProb(dc)            
            else:           
                restartStatus = False
                
                if(self.label == 2):
                    #DMET 1: Generate bath
                    h_emb,v_emb,R,rdmCore = self.generateBath()

                    #DMET 2: Do correlated problem
                    self.solveCorrProb(h_emb,v_emb,R,rdmCore)
                    self.ediff = abs(self.Efrag-Efrag0)/self.Nimp
                    Efrag0 = self.Efrag
       
                    #Checkpoint 
                    if(self.chkPoint and self.itr%self.chkPointInterval == 0):
                        self.checkPoint()
                                           
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
            sys.stdout.flush() 
            
            ######################################################################
            # CHECKPOINT
            ###################################################################### 
            if(self.chkPoint and self.itr%self.chkPointInterval == 0):
                self.checkPoint()

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

            #Write a table
            ftbl = open(self.tableFile,'a')
            print >>ftbl,'%3d\t% 16.9e\t%16.9e\t%16.9e\t%16.9e' %(self.itr,self.Efrag/self.Nimp,self.ediff,self.critnorm,u_mat_diff)
            ftbl.close()

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



