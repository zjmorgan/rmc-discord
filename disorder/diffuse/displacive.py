#!/usr/bin/env python

import numpy as np

from scipy import spatial
from scipy.optimize import fsolve

from disorder.diffuse.powder import displacive

def f(x,y):
    return (x-np.sin(x))/np.pi-y

def g(x,y):
    return (1-np.cos(x))/np.pi

def expansion(nu, nv, nw, n_atm, value=1, structure=None, fixed=True): 
    """
    Generate random displacement vectors.
    
    Parameters
    ----------
    nu : int
        :math:`N_1` number of grid points along the :math:`a`-axis of the \
        supercell
    nv : int
        :math:`N_2` number of grid points along the :math:`b`-axis of the \
        supercell
    nw : int
        :math:`N_3` number of grid points along the :math:`c`-axis of the \
        supercell
    n_atm : int
        Number of atoms in the unit cell
    value : ndarray, optional
        Magnitude of displacement vector, default ``value=1``
    
    Returns
    -------
    Ux, Uy, Uz : ndarray
        Each array has a flattened shape of size ``nu*nv*nw*n_atm``
        
    """
    
    theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm)
    phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm))
    
    nx = np.sin(phi)*np.cos(theta)
    ny = np.sin(phi)*np.sin(theta)
    nz = np.cos(phi) 
    
    if (len(np.shape(value)) == 0):
        Vxx = Vyy = Vzz = np.full(n_atm, value)
        Vyz = Vxz = Vxy = np.full(n_atm, 0)
    elif (len(np.shape(value)) == 1):
        Vxx = Vyy = Vzz = value
        Vyz = Vxz = Vxy = np.full(n_atm, 0)
    else:
        Vxx, Vyy, Vzz = value[0], value[1], value[2]
        Vyz, Vxz, Vxy = value[3], value[4], value[5]
    
    U = 1 if fixed else np.random.rand(nu,nv,nw,n_atm)
    
    ms = np.sqrt(Vxx*nx*nx+Vyy*ny*ny+Vzz*nz*nz\
             +2*(Vxz*nx*nz+Vyz*ny*nz+Vxy*nx*ny))
            
    Ux = U*ms*nx
    Uy = U*ms*ny
    Uz = U*ms*nz
    
    return Ux.flatten(), Uy.flatten(), Uz.flatten()

def rotation(ux, uy, uz, nu, nv, nw, n_atm, value, structure): 
    
    rx = (ux[structure].T-ux[structure[:,0]]).T
    ry = (uy[structure].T-uy[structure[:,0]]).T
    rz = (uz[structure].T-uz[structure[:,0]]).T
    
    Rx = np.zeros((nu*nv*nw*n_atm))
    Ry = np.zeros((nu*nv*nw*n_atm))
    Rz = np.zeros((nu*nv*nw*n_atm))
    
    Px = np.zeros((nu*nv*nw*n_atm))
    Py = np.zeros((nu*nv*nw*n_atm))
    Pz = np.zeros((nu*nv*nw*n_atm))
    
    Rx[structure] = rx
    Ry[structure] = ry
    Rz[structure] = rz
    
    m = structure.shape[0]
    n = structure.shape[1]
    
    psi = np.zeros((m,n))
    
    theta = np.repeat(2*np.pi*np.random.rand(m),n).reshape(m,n)
    phi = np.repeat(np.arccos(1-2*np.random.rand(m)),n).reshape(m,n)
    
    for i in range(m):    
        psi[i,:] = fsolve(f, np.pi/2, args=(np.random.rand()), fprime=g)
    
    u = np.sin(phi)*np.cos(theta)
    v = np.sin(phi)*np.sin(theta)
    w = np.cos(phi)
    
    theta = np.repeat(2*np.pi*np.random.rand(m),n).reshape(m,n)
    phi = np.repeat(np.arccos(1-2*np.random.rand(m)),n).reshape(m,n)
    
    a = np.sin(phi)*np.cos(theta)
    b = np.sin(phi)*np.sin(theta)
    c = np.cos(phi)
    
    Rxx = np.cos(psi)+u**2*(1-np.cos(psi))
    Rxy = u*v*(1-np.cos(psi))-w*np.sin(psi)
    Rxz = u*w*(1-np.cos(psi))+v*np.sin(psi)

    Ryx = v*u*(1-np.cos(psi))+w*np.sin(psi)    
    Ryy = np.cos(psi)+v**2*(1-np.cos(psi))
    Ryz = v*w*(1-np.cos(psi))-u*np.sin(psi)
 
    Rzx = w*u*(1-np.cos(psi))-v*np.sin(psi)
    Rzy = w*v*(1-np.cos(psi))+u*np.sin(psi)
    Rzz = np.cos(psi)+w**2*(1-np.cos(psi))
    
    Px[structure] = Rxx*Rx[structure]+Rxy*Ry[structure]+Rxz*Rz[structure]
    Py[structure] = Ryx*Rx[structure]+Ryy*Ry[structure]+Ryz*Rz[structure]
    Pz[structure] = Rzx*Rx[structure]+Rzy*Ry[structure]+Rzz*Rz[structure]
    
    V = np.random.rand(m,n)*value
    
    Px[structure] += V*a
    Py[structure] += V*b
    Pz[structure] += V*c
        
    Ux = Px-Rx
    Uy = Py-Ry
    Uz = Pz-Rz
    
    return Ux, Uy, Uz, Px, Py, Pz, Rx, Ry, Rz

def number(m):
    
    return (m+1)*(m+2) // 2

def factorial(n):
    
    if (n == 1 or n == 0):
        return 1
    else:
        return n*factorial(n-1)
    
def coefficients(p):
        
    coeffs = np.zeros(np.sum(number(np.arange(p+1))), dtype=complex)
    
    j = 0
    for i in range(p+1):
        for w in range(i+1):
            nw = factorial(w)
            for v in range(i+1):
                nv = factorial(v)
                for u in range(i+1):
                    nu = factorial(u)
                    if (u+v+w == i):
                        coeffs[j] = 1j**i/(nu*nv*nw)
                        j += 1        
                        
    return coeffs

def products(Vx, Vy, Vz, p):
    
    if (type(Vx) is np.ndarray):    
        n = Vx.shape[0]
    else:
        n = 1
    
    V = np.ones((np.sum(number(np.arange(p+1))),n))

    j = 0
    for i in range(p+1):
        for w in range(i+1):
            for v in range(i+1):
                for u in range(i+1):
                    if (u+v+w == i):
                        V[j,:] = Vx**u*Vy**v*Vz**w
                        j += 1
    
    return V.flatten()

def transform(U, 
              H,
              K,
              L,
              nu, 
              nv, 
              nw, 
              n_atm):
    """
    Discrete Fourier transform of Taylor expansion displacement products.

    Parameters
    ----------
    U : ndarray
        Displacement parameter :math:`U` (in Cartesian coordinates)
    H : ndarray, int
        Supercell index along the :math:`a^*`-axis in reciprocal space
    K : ndarray, int
        Supercell index along the :math:`b^*`-axis in reciprocal space
    L : ndarray, int
        Supercell index along the :math:`c^*`-axis in reciprocal space
    nu : int
        :math:`N_1` number of grid points along the :math:`a`-axis of the \
        supercell
    nv : int
        :math:`N_2` number of grid points along the :math:`b`-axis of the \
        supercell
    nw : int
        :math:`N_3` number of grid points along the :math:`c`-axis of the \
        supercell
    n_atm : int
        Number of atoms in the unit cell

    Returns
    -------
    U_k : ndarray
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
    i_dft : ndarray, int
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
        
    """
    
    n_prod = U.shape[0] // (nu*nv*nw*n_atm)
    
    U_k = np.fft.ifftn(U.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)
    
    i_dft = Kw+nw*(Kv+nv*Ku)
             
    return U_k.flatten(), i_dft

def intensity(U_k, 
              Q_k,
              coeffs,
              cond,
              p,
              i_dft,
              factors,
              subtract=True):    
    """
    Displacive scattering intensity.

    Parameters
    ----------
    U_k : ndarray
        Fourier transform of Taylor expansion displacement products
    Q_k : ndarray
        Fourier transform of Taylor expansion wavevector products
    coeffs : ndarray
        Taylor expansion coefficients
    cond : ndarray
        Array indices corresponding to nuclear Bragg peaks
    p : int
        Order of Taylor expansion
    i_dft : ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors : ndarray
        Prefactors of form factors, phase factors, and composition factors
    subtract : boolean, optional
       Optionally subtract the Bragg intensity or return the Bragg structure \
       factor
       
    Returns
    -------
    I : ndarray
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``
    F_bragg : ndarray
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``
        
    """
    
    n_prod = coeffs.shape[0]
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = U_k.shape[0] // n_prod // n_atm        
    
    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_peaks)
        
    V_k = np.zeros(factors.shape, dtype=np.complex)
    V_k_nuc = np.zeros((cond.sum(),n_atm), dtype=np.complex)
    
    start = (np.cumsum(number(np.arange(p+1)))-number(np.arange(p+1)))[::2]
    end = np.cumsum(number(np.arange(p+1)))[::2]
    
    even = []
    
    for k in range(len(end)):
        even += range(start[k], end[k])
    
    even = np.array(even)
    #even = np.arange(coeffs.size)
    
    for j in range(n_atm):
        V_k[:,j] = coeffs.dot(U_k[:,i_dft,j]*Q_k[:,:])
        V_k_nuc[:,j] = coeffs[even].dot(U_k[:,i_dft,j][even,:][:,cond]\
                                                  *Q_k[even,:][:,cond])
        
    prod = factors*V_k
    prod_nuc = factors[cond,:]*V_k_nuc

    F = np.sum(prod, axis=1)
    F_nuc = np.sum(prod_nuc, axis=1)
    
    if (subtract):
        
        F[cond] -= F_nuc
                      
        I = np.real(F)**2+np.imag(F)**2
         
        return I/(n_uvw*n_atm)
    
    else:
        
        F_bragg = np.zeros(F.shape, dtype=np.complex)
        F_bragg[cond] = F_nuc
                              
        I = np.real(F)**2+np.imag(F)**2
         
        return I/(n_uvw*n_atm), F_bragg

def structure(U_k, 
              Q_k, 
              coeffs, 
              cond,
              p,
              i_dft,
              factors):
    """
    Partial displacive structure factor.

    Parameters
    ----------
    U_k : ndarray
        Fourier transform of Taylor expansion displacement products
    Q_k : ndarray
        Fourier transform of Taylor expansion wavevector products
    coeffs : ndarray
        Taylor expansion coefficients
    cond : ndarray
        Array indices corresponding to nuclear Bragg peaks
    p : int
        Order of Taylor expansion
    i_dft : ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors : ndarray
        Prefactors of scattering lengths, phase factors, and occupancies

    Returns
    -------
    F : ndarray
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``
    F_nuc : ndarray
        Array has a flattened shape of size ``cond.sum()*i_dft.shape[0]``
    prod : ndarray
        Array has a flattened shape of size 
        ``coeffs.shape[0]*i_dft.shape[0]*n_atm``
    prod_nuc : ndarray
        Array has a flattened shape of size 
        ``coeffs.sum()*i_dft.shape[0]*n_atm``
    V_k : ndarray
        Array has a flattened shape of size 
        ``coeffs.shape[0]*i_dft.shape[0]*n_atm``
    V_k_nuc : ndarray
        Array has a flattened shape of size 
        ``coeffs.sum()*i_dft.shape[0]*n_atm``
    even : ndarray, int
        Array indices of the even Taylor expandion coefficients
    bragg : ndarray, int
        Array has a flattened shape of size ``coeffs.sum()``
        
    """
    
    n_prod = coeffs.shape[0]
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = U_k.shape[0] // n_prod // n_atm
    
    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_peaks)
        
    V_k = np.zeros(factors.shape, dtype=np.complex)
    V_k_nuc = np.zeros((cond.sum(),n_atm), dtype=np.complex)
    
    start = (np.cumsum(number(np.arange(p+1)))-number(np.arange(p+1)))[::2]
    end = np.cumsum(number(np.arange(p+1)))[::2]
    
    even = []
    
    for k in range(len(end)):
        even += range(start[k], end[k])
    
    even = np.array(even)
    #even = np.arange(coeffs.size)
    
    for j in range(n_atm):
        V_k[:,j] = coeffs.dot(U_k[:,i_dft,j]*Q_k[:,:])
        V_k_nuc[:,j] = coeffs[even].dot(U_k[:,i_dft,j][even,:][:,cond]\
                                                  *Q_k[even,:][:,cond])
        
    prod = factors*V_k
    prod_nuc = factors[cond,:]*V_k_nuc

    F = np.sum(prod, axis=1)
    F_nuc = np.sum(prod_nuc, axis=1)
    
    bragg = np.arange(n_peaks)[cond]
     
    return F, \
           F_nuc, \
           prod.flatten(), \
           prod_nuc.flatten(), \
           V_k.flatten(), \
           V_k_nuc.flatten(), \
           even, \
           bragg

def powder(Q, 
           Ux, 
           Uy, 
           Uz, 
           rx, 
           ry, 
           rz, 
           scattering_length, 
           fract=0.5):
    
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    coordinate = np.array(list(pairs))
            
    i = coordinate[:,0].copy()
    j = coordinate[:,1].copy()
        
    n_hkl = Q.shape[0]
    n_xyz = Ux.shape[0]
    
    n_pairs = i.shape[0]
    n_atm = scattering_length.shape[0] // n_hkl
        
    k = np.mod(i,n_atm)
    l = np.mod(j,n_atm)
        
    m = np.arange(n_xyz)
    n = np.mod(m,n_atm)
        
    summation = np.zeros(Q.shape[0])
    
    auto = np.zeros(Q.shape[0])

    displacive(summation, 
               auto,
               Q,
               Ux,
               Uy,
               Uz,
               rx,
               ry,
               rz,
               scattering_length,
               i,
               j,
               k,
               l,
               n,
               n_xyz,
               n_atm)    
    
    scale = n_xyz/((np.sqrt(8*n_pairs+1)+1)/2)
            
    I = (auto/scale+2*summation)/n_xyz
    
    return I

def debye_waller(Q, Ux, Uy, Uz):
            
    U = np.sqrt(Ux**2+Uy**2+Uz**2)
        
    QU = np.kron(Q,U)
    
    T = np.exp(-QU.reshape(Q.shape[0],U.shape[0])**2/3)/U.shape[0]
        
    return np.sum(T,axis=1).flatten()