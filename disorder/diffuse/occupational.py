#!/usr/bin/env python

import numpy as np

def composition(nu, nv, nw, n_atm, value=0.5):
    """
    Generate random relative site occupancies.
    
    Parameters
    ----------
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the \
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell
    n_atm : int
        Number of atoms in the unit cell
    value : float, 1d array, optional
        Average of site occupancies, defualt ``value=0.5``
    
    Returns
    -------
    A : 1d array
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
        
    """
                    
    A_r = (np.random.random((nu,nv,nw,n_atm))<=value)/value-1
                        
    return A_r.flatten()

def transform(A_r, H, K, L, nu, nv, nw, n_atm):
    """
    Discrete Fourier transform of relative occupancy parameter.

    Parameters
    ----------
    A_r : 1d array
          Relative occupancy parameter :math:`A` 
    H, K, L : 1d array, int
        Supercell index along the :math:`a^*`, :math:`b^*`, and \
        :math:`c^*`-axis in reciprocal space
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the \
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell
    n_atm : int
        Number of atoms in the unit cell

    Returns
    -------
    A_k : 1d array
        Array has a flattened shape of size ``nu*nw*nv*n_atm``
    i_dft : 1d array, int
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
    A_k : 1d array
        Fourier transform of relative site occupancies
    i_dft: 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: 1d array
        Prefactors of form factors, phase factors, and composition factors

    Returns
    -------
    I : 1d array
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
    A_k : 1d array
        Fourier transform of relative site occupancies
    i_dft: 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: 1d array
        Prefactors of form factors, phase factors, and composition factors

    Returns
    -------
    F : 1d array
        Array has a flattened shape of size ``i_dft.shape[0]``
    prod : 1d array
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