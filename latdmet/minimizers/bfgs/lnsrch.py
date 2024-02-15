import numpy as np
import scipy.linalg as la

def lnsrch(x0,f0,f,g,p,stpmax,tolx=1.0e-10,alpha=1.0e-4,itrmax=100):
    '''
    Theory: x = x0 + \lambda*p
    return: x,f(x),status,msg
    status:
        -2 - Roundoff problem (Severe error)
        -1 - itrmax reached
        0  - x is too close to x0
        1  - success
    '''
    nrm = la.norm(p)

    if(nrm>stpmax):
        x0 = stpmax/nrm

    slope = np.dot(g,p)

    if(slope>0):
        msg = 'Roundoff problem in line search'
        return x0,f0,-2,msg

    test=0.0
    for i in range(len(x0)):
        temp=abs(p[i])/max(abs(x0[i]),1.0)
        if(temp>test):
            test=temp

    alamin = tolx/test
    alam=1.0
    f2 = 0.0
    alam2 = 0.0

    for itr in range(itrmax):
        x = x0 + alam*p
        fval = f(x)
        if(alam<alamin):
            x=x0
            msg = 'Convergence on dx. Verify convergence'
            return x,fval,0,msg
        elif(fval <= f0+alpha*alam*slope):
            msg = 'Sufficient function decrease. Backtrack if needed.'
            return x,fval,1,msg
        else:
            if(abs(alam-1.0)<1.0e-16):
                tmplam = -slope/(2.0*(fval-f0-slope))
            else:
                rhs1=fval-f0-alam*slope
                rhs2=f2-f0-alam2*slope
                a=(rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2)
                b=(-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2)
                if(abs(a)<1.0e-16):
                    tmplam = -slope/(2.0*b)
                else:
                    disc=b*b-3.0*a*slope
                    if(disc<0.0):
                        tmplam=0.5*alam
                    elif(b<=0.0):
                        tmplam=(-b+disc**0.5)/(3.0*a)
                    else:
                        tmplam=-slope/(b+sqrt(disc))
                
                if(tmplam>0.5*alam):
                    tmplam=0.5*alam
        alam2 = alam
        f2 = f
        alam=max(tmplam,0.1*alam)
    
    msg = 'Did not converge normally. Terminated because reached max. iterations'
    return x,fval,-1,msg
