import numpy as np

def ham_hubbard1d(L, t=1.0, tp=0.0,bc='pbc'):
    h = np.zeros((L,L))
    for i in xrange(L):
        h[i,(i+1)%L] = -1.0*t
        h[i,(i-1)%L] = -1.0*t
        h[i,(i+2)%L] = -1.0*tp
        h[i,(i-2)%L] = -1.0*tp

    if bc == 'apbc':
        h[0,-1] = t
        h[-1,0] = t
        h[0,-2] = tp
        h[1,-1] = tp
        h[-2,0] = tp
        h[-1,1] = tp

    return h

def ham_hubbard2d(Nx, Ny, Ix, Iy, t=1.0, tp=0.0, bc='pbc'):
    Lx   = Nx * Ix
    Ly   = Ny * Iy
    norb = Lx * Ly
    nimp = Ix * Iy
    h1 = np.zeros((norb,norb))
    for j in range(Ny):
      for i in range(Nx):
        for y in range(Iy):
          for x in range(Ix):
            site = (Nx*j + i)*nimp + Ix*y + x

            # Nearest-Neighbor
            nny = (j+(y+1)//Iy)%Ny
            cy  = (y+1)%Iy
            nnx = (i+(x+1)//Ix)%Nx
            cx  = (x+1)%Ix
            bnx = (i-(Ix-x)//Ix)%Nx
            bx  = (x-1)%Ix
            bny = (j-(Iy-y)//Iy)%Ny
            by  = (y-1)%Iy
 
            sited = (Nx*nny + i)*nimp + Ix*cy + x
            siter = (Nx*j + nnx)*nimp + Ix*y + cx
            siteu = (Nx*bny + i)*nimp + Ix*by + x
            sitel = (Nx*j + bnx)*nimp + Ix*y + bx
 
            h1[site,siteu] = -1.0*t
            h1[site,siter] = -1.0*t
            h1[site,sited] = -1.0*t
            h1[site,sitel] = -1.0*t

            # Next to Nearest-Neighbor
            site0 = (Nx*bny + bnx)*nimp + Ix*by + bx
            site1 = (Nx*bny + nnx)*nimp + Ix*by + cx
            site2 = (Nx*nny + bnx)*nimp + Ix*cy + bx
            site3 = (Nx*nny + nnx)*nimp + Ix*cy + cx
            h1[site,site0] = -1.0*tp
            h1[site,site1] = -1.0*tp
            h1[site,site2] = -1.0*tp
            h1[site,site3] = -1.0*tp

            # Third to Nearest-Neighbor
            #nny = (j+(y+2)//Iy)%Ny
            #cy  = (y+2)%Iy
            #nnx = (i+(x+2)//Ix)%Nx
            #cx  = (x+2)%Ix
            #bnx = (i-(Ix+1-x)//Ix)%Nx
            #bx  = (x-2)%Ix
            #bny = (j-(Iy+1-y)//Iy)%Ny
            #by  = (y-2)%Iy
 
            #sited = (Nx*nny + i)*nimp + Ix*cy + x
            #siter = (Nx*j + nnx)*nimp + Ix*y + cx
            #siteu = (Nx*bny + i)*nimp + Ix*by + x
            #sitel = (Nx*j + bnx)*nimp + Ix*y + bx
            
    return h1

def ham_hubbard2d_n(Nx, Ny, Ix, Iy, t=1.0, tp=0.0, bc='pbc'):
    Lx   = Nx * Ix
    Ly   = Ny * Iy
    norb = Lx * Ly
    nimp = Ix * Iy
    h1 = np.zeros((norb,norb))
    for j in range(Ny):
      for i in range(Nx):
        for y in range(Iy):
          for x in range(Ix):
            site = (Nx*j + i)*nimp + Ix*y + x

            # Nearest-Neighbor
            scuy = (j-(Iy-y)//Iy)%Ny  #up supercell
            iuy  = (y-1)%Iy           #up impurity site
            scdy = (j+(y+1)//Iy)%Ny   #down supercell
            idy  = (y+1)%Iy           #down impurity site
            sclx = (i+(x+1)//Ix)%Nx   #left supercell
            ilx  = (x+1)%Ix           #left impurity site
            scrx = (i-(Ix-x)//Ix)%Nx  #right supercell
            irx  = (x-1)%Ix           #right impurity site
 
            sited = (Nx*nny + i)*nimp + Ix*cy + x
            siter = (Nx*j + nnx)*nimp + Ix*y + cx
            siteu = (Nx*bny + i)*nimp + Ix*by + x
            sitel = (Nx*j + bnx)*nimp + Ix*y + bx
 
            h1[site,siteu] = -1.0*t
            h1[site,siter] = -1.0*t
            h1[site,sited] = -1.0*t
            h1[site,sitel] = -1.0*t

            # Next to Nearest-Neighbor
            site0 = (Nx*bny + bnx)*nimp + Ix*by + bx
            site1 = (Nx*bny + nnx)*nimp + Ix*by + cx
            site2 = (Nx*nny + bnx)*nimp + Ix*cy + bx
            site3 = (Nx*nny + nnx)*nimp + Ix*cy + cx
            h1[site,site0] = -1.0*tp
            h1[site,site1] = -1.0*tp
            h1[site,site2] = -1.0*tp
            h1[site,site3] = -1.0*tp

    return h1

                

if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    h =  ham_hubbard2d(1,1,3,3, t=-1, tp=-2)
    print h
    print np.linalg.norm(h-h.T)
