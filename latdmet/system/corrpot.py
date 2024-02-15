import numpy as np


def corrpot1d(nimp, U, AFMorder, cmtype="UHF"):
    u0 = U/2.
    hn = nimp/2
    if cmtype == 'RHF':
        umg = np.eye(2*nimp)*u0
    else:
        if (nimp%2 == 0):
            a = [u0+AFMorder, u0-AFMorder]*hn
            b = [u0-AFMorder, u0+AFMorder]*hn
            umg = np.diag(a+b)

        else:
            umg = np.eye(2*nimp)*u0

    return umg


def corrpot2d():
    pass


