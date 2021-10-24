#!/usr/bin/env python

import numpy as np

from scipy import spatial

def composition(nu, nv, nw, n_atm, value=0.5):
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
                    
    A_r = (np.random.random((nu,nv,nw,n_atm))<=value)/value-1
                        
    return A_r.flatten()

def transform(A_r, H, K, L, nu, nv, nw, n_atm):
    """
    Discrete Fourier transform of relative occupancy parameter.

    Parameters
    ----------
    A_r : ndarray
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
    
    A_k = np.fft.ifftn(A_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)
    
    i_dft = Kw+nw*(Kv+nv*Ku)
         
    return A_k.flatten(), i_dft

def intensity(A_k, i_dft, factors):
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

def structure(A_k, i_dft, factors):
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