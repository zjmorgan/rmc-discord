#!/usr/bin/env python

import numpy as np

from disorder.diffuse.displacive import number

def transform(U_r, A_r, H, K, L, nu, nv, nw, n_atm):
    """
    Discrete Fourier transform of Taylor expansion displacement products and \
    relative occupancy parameter.

    Parameters
    ----------
    U_r : 1d array
          Displacement parameter :math:`U` (in Cartesian coordinates).
    A_r : 1d array
          Relative occupancy parameter :math:`A`.
    H, K, L : 1d array, int
        Supercell index along the :math:`a^*`, :math:`b^*`, and
        :math:`c^*`-axis in reciprocal space.
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell.
    n_atm : int
        Number of atoms in the unit cell.

    Returns
    -------
    U_k : 1d array
        Array has a flattened shape of size ``nu*nw*nv*n_atm``.
    A_k : 1d array
        Array has a flattened shape of size ``nu*nw*nv*n_atm``.
    i_dft : 1d array, int
        Array has a flattened shape of size ``nu*nw*nv*n_atm``.

    """

    n_prod = U_r.shape[0] // (nu*nv*nw*n_atm)

    U_r = U_r.reshape(n_prod,nu,nv,nw,n_atm)

    U_k = np.fft.ifftn(U_r, axes=(1,2,3))*nu*nv*nw

    A_r = np.tile(A_r, n_prod).reshape(n_prod,nu,nv,nw,n_atm)

    A_k = np.fft.ifftn(A_r*U_r, axes=(1,2,3))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)

    i_dft = Kw+nw*(Kv+nv*Ku)

    return U_k.flatten(), A_k.flatten(), i_dft

def intensity(U_k, A_k, Q_k, coeffs, cond, p, i_dft, factors, subtract=True):
    """
    Chemical scattering intensity.

    Parameters
    ----------
    U_k : 1d array
        Fourier transform of Taylor expansion displacement products.
    A_k : 1d array
        Fourier transform of relative site occupancies.
    Q_k : 1d array
        Fourier transform of Taylor expansion wavevector products.
    coeffs : 1d array
        Taylor expansion coefficients.
    cond : 1d array
        Array indices corresponding to nuclear Bragg peaks.
    p : int
        Order of Taylor expansion.
    i_dft: 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space.
    factors: 1d array
        Prefactors of form factors, phase factors, and composition factors.

    Returns
    -------
    I : 1d array
        Array has a flattened shape of size ``i_dft.shape[0]``.

    """

    n_prod = coeffs.shape[0]

    n_peaks = i_dft.shape[0]

    n_atm = factors.shape[0] // n_peaks

    factors = factors.reshape(n_peaks,n_atm)

    n_uvw = U_k.shape[0] // n_prod // n_atm

    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    A_k = A_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_peaks)

    start = (np.cumsum(number(np.arange(p+1)))-number(np.arange(p+1)))[::2]
    end = np.cumsum(number(np.arange(p+1)))[::2]

    even = []

    for k in range(len(end)):
        even += range(start[k], end[k])

    even = np.array(even)

    V_k = np.einsum('ijk,kj->ji', coeffs*(U_k[:,i_dft,:]+\
                                          A_k[:,i_dft,:]).T, Q_k)
    V_k_nuc = np.einsum('ijk,kj->ji',
                        (coeffs[even]*(U_k[:,i_dft,:][even,:]+\
                                       A_k[:,i_dft,:][even,:]).T),
                        Q_k[even,:])[cond]

    prod = factors*V_k
    prod_nuc = factors[cond,:]*V_k_nuc

    F = np.sum(prod, axis=1)
    F_nuc = np.sum(prod_nuc, axis=1)

    if subtract:
        F[cond] -= F_nuc

        I = np.real(F)**2+np.imag(F)**2
        return I/(n_uvw*n_atm)
    else:
        F_bragg = np.zeros(F.shape, dtype=complex)
        F_bragg[cond] = F_nuc

        I = np.real(F)**2+np.imag(F)**2
        return I/(n_uvw*n_atm), F_bragg

def structure(U_k, A_k, Q_k, coeffs, cond, p, i_dft, factors):
    """
    Partial displacive structure factor.

    Parameters
    ----------
    U_k : 1d array
        Fourier transform of Taylor expansion displacement products.
    A_k : 1d array
        Fourier transform of relative site occupancies times Taylor expansion
        displacement products.
    Q_k : 1d array
        Fourier transform of Taylor expansion wavevector products.
    coeffs : 1d array
        Taylor expansion coefficients.
    cond : 1d array
        Array indices corresponding to nuclear Bragg peaks.
    p : int
        Order of Taylor expansion.
    i_dft: 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space.
    factors: 1d array
        Prefactors of scattering lengths, phase factors, and occupancies.

    Returns
    -------
    F : 1d array
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``.
    F_nuc : 1d array
        Array has a flattened shape of size ``cond.sum()*i_dft.shape[0]``.
    prod : 1d array
        Array has a flattened shape of size
        ``coeffs.shape[0]*i_dft.shape[0]*n_atm``.
    prod_nuc : 1d array
        Array has a flattened shape of size
        ``coeffs.sum()*i_dft.shape[0]*n_atm``.
    V_k : 1d array
        Array has a flattened shape of size
        ``coeffs.shape[0]*i_dft.shape[0]*n_atm``.
    V_k_nuc : 1d array
        Array has a flattened shape of size
        ``coeffs.sum()*i_dft.shape[0]*n_atm``.
    even : 1d array, int
        Array indices of the even Taylor expandion coefficients.
    bragg : 1d array, int
        Array has a flattened shape of size ``coeffs.sum()``.

    """

    n_prod = coeffs.shape[0]

    n_peaks = i_dft.shape[0]

    n_atm = factors.shape[0] // n_peaks

    factors = factors.reshape(n_peaks,n_atm)

    n_uvw = U_k.shape[0] // n_prod // n_atm

    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    A_k = A_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_peaks)

    start = (np.cumsum(number(np.arange(p+1)))-number(np.arange(p+1)))[::2]
    end = np.cumsum(number(np.arange(p+1)))[::2]

    even = []

    for k in range(len(end)):
        even += range(start[k], end[k])

    even = np.array(even)

    V_k = np.einsum('ijk,kj->ji', coeffs*(U_k[:,i_dft,:]+\
                                          A_k[:,i_dft,:]).T, Q_k)
    V_k_nuc = np.einsum('ijk,kj->ji',
                        (coeffs[even]*(U_k[:,i_dft,:][even,:]+\
                                       A_k[:,i_dft,:][even,:]).T),
                        Q_k[even,:])[cond]

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