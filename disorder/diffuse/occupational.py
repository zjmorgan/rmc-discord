#!/usr/bin/env python

import numpy as np

from scipy import spatial

from disorder.diffuse.powder import occupational

def composition(nu, nv, nw, n_atm, value=0.5, structure=None):
    """
    Generate random relative site occupancies.
    
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
    value : float, ndarray, optional
        Average of site occupancies, defualt ``value=0.5``
    
    Returns
    -------
    A : ndarray
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
        
    """
    
    if (structure is None):
                
        A = (np.random.random((nu,nv,nw,n_atm))<=value)/value-1
        
        A = A.flatten()
        
    else:
        
        A = np.zeros((nu*nv*nw*n_atm))
        
        if (np.size(value) > 1):
        
            atms = np.mod(structure, n_atm)
            
            m, n = atms.shape[0], atms.shape[1]
            
            values = value[atms]
                                    
            A[structure] = (np.random.random((m,n))<=values)/values-1
                
        else:
            
            m, n = structure.shape[0], structure.shape[1]
            
            a = np.random.random(m)
                            
            A[structure] = (np.repeat(a,n).reshape(m,n)<=value)/value-1
                
    return A

def transform(A, 
              H,
              K,
              L,
              nu, 
              nv, 
              nw, 
              n_atm):
    """
    Discrete Fourier transform of relative occupancy parameter.

    Parameters
    ----------
    A : ndarray
        Relative occupancy parameter :math:`A` 
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
    A_k : ndarray
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
    i_dft : ndarray, int
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
        
    """
    
    A_k = np.fft.ifftn(A.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)
    
    i_dft = Kw+nw*(Kv+nv*Ku)
         
    return A_k.flatten(), i_dft

def intensity(A_k, 
              i_dft,
              factors):
    """
    Chemical scattering intensity.

    Parameters
    ----------
    A_k : ndarray
        Fourier transform of relative site occupancies
    i_dft: ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: ndarray
        Prefactors of form factors, phase factors, and composition factors

    Returns
    -------
    I : ndarray
        Array has a flattened shape of size ``i_dft.shape[0]``

    """
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = A_k.shape[0] // n_atm
            
    A_k = A_k.reshape(n_uvw,n_atm)
    
    prod = factors*A_k[i_dft,:]

    F = np.sum(prod, axis=1)
                  
    I = np.real(F)**2+np.imag(F)**2
     
    return I/(n_uvw*n_atm)

def structure(A_k,
              i_dft,
              factors):
    """
    Partial chemical structure factor.

    Parameters
    ----------
    A_k : ndarray
        Fourier transform of relative site occupancies
    i_dft: ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: ndarray
        Prefactors of form factors, phase factors, and composition factors

    Returns
    -------
    F : ndarray
        Array has a flattened shape of size ``i_dft.shape[0]``
    prod : ndarray
        Array has a flattened shape of size 
        ``i_dft.shape[0]*n_atm``

    """
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = A_k.shape[0] // n_atm
    
    A_k = A_k.reshape(n_uvw,n_atm)
    
    prod = factors*A_k[i_dft,:]

    F = np.sum(prod, axis=1)
     
    return F, prod.flatten()

def powder(Q, A, rx, ry, rz, scattering_length, fract=0.5):
    
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    coordinate = np.array(list(pairs))
            
    i = coordinate[:,0]
    j = coordinate[:,1]
        
    n_hkl = Q.shape[0]
    n_xyz = A.shape[0]
    
    n_pairs = i.shape[0]
    n_atm = scattering_length.shape[0] // n_hkl
    
    k = np.mod(i,n_atm)
    l = np.mod(j,n_atm)
    
    rx_ij = rx[j]-rx[i]
    ry_ij = ry[j]-ry[i]
    rz_ij = rz[j]-rz[i]
    
    r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)
    
    m = np.arange(n_xyz)
    n = np.mod(m,n_atm)
    
    delta_i_delta_i = (1+A[m])**2
    delta_ij = (1+A[i])*(1+A[j])
    
    summation = np.zeros(Q.shape[0])
    
    auto = np.zeros(Q.shape[0])

    occupational(summation, 
                 auto,
                 Q,
                 r_ij,
                 scattering_length,
                 delta_ij,
                 delta_i_delta_i,
                 k,
                 l,
                 n,
                 n_xyz,
                 n_atm)
    
    scale = n_xyz/((np.sqrt(8*n_pairs+1)+1)/2)
        
    I = (auto/scale+2*summation)/n_xyz
    
    return I