import sys
import numpy as np
import scipy.linalg as la
from lnsrch import *

def bfgs_min(x0,func,jac,gtol=1.0e-6,itrmax=200,tolx=1.0e-18,astpmax=100.0):

    eps = np.finfo(x0.dtype).eps
    n = x0.size   
    H = np.eye(n)
    
    fp = func(x0)
    g = jac(x0)
    xi = -g 

    nrm = la.norm(x0)
    stpmax=astpmax*max(nrm,n*1.0)

    x = np.copy(x0)
    for itr in range(itrmax):
        
        #First do line search
        xn,fp,status,msg = lnsrch(x,fp,func,g,xi,stpmax)      
     
        #Update
        xi = xn-x
        x = xn 

        #Check convergence on function value
        test = 0.0
        for i in range(n):
             temp = abs(xi[i])/max(abs(x[i]),1.0)
             if(temp > test):
                test = temp
        
        abtest = la.norm(xi)/n
        dg = np.copy(g)
        g = jac(x)
        
        if(test<tolx or abtest<tolx):
            msg = 'Function change tolerance reached.'
            return x,g,1,msg

        #Check convergence on gradient
        den = max(fp,1.0)
        test = 0.0
        for i in range(n):
            temp = abs(g[i])*max(abs(x[i]),1.0)/den
            if(temp>test):
                test = temp
        if(test<gtol):
            msg = 'Gradient tolerance reached.'
            return x,g,1,msg

        dg = g - dg
        hdg = np.dot(H,dg)
            
        #Calculate denominators
        fac=np.dot(dg,xi)
        fae=np.dot(dg,hdg)
        sumdg = np.sum(np.sqrt(dg))
        sumxi = np.sum(np.sqrt(xi))
        
        if(fac>np.sqrt(eps*sumdg*sumxi)):
            fac=1.0/fac
            fad=1.0/fae
            dg = fac*xi - fad*hdg
            H = H + np.multiply(fac,np.outer(xi,xi)) - np.multiply(fad,np.outer(hdg,hdg)) + np.multiply(fae,np.outer(dg,dg)) 

            #Need to enforce Hermiticity
            H = np.triu(H) + np.triu(H,1).T
            
        xi -= np.dot(H,g) 

    msg = 'Max iterations reached.'
    return x,g,0,msg


      
