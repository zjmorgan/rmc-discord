#!/usr/bin/env python3

import numpy as np

from scipy import constants
from disorder.material import tables

c = (0.5*constants.value('classical electron radius')*
      constants.value('neutron mag. mom. to nuclear magneton ratio')*
  0.1/constants.femto)**2

def j0(Q, A, a, B, b, C, c, D):
    """
    Appoximation of the spherical Bessesl function :math:`j_0(Q)`.

    Parameters
    ----------
    Q : ndarray
        Magnitude of wavevector :math:`Q`
    A : float
        :math:`A_0` constant
    a : float
        :math:`a_0` constant
    B : float
        :math:`B_0` constant
    b : float
        :math:`b_0` constant
    C : float
        :math:`C_0` constant
    c : float
        :math:`c_0` constant
    D : float
        :math:`D_0` constant

    Returns
    -------
    j0 : ndarray
        Has the same shape as the input wavevector

    """

    s = Q/4/np.pi

    return A*np.exp(-a*s**2)+B*np.exp(-b*s**2)+C*np.exp(-c*s**2)+D

def j2(Q, A, a, B, b, C, c, D):
    """
    Appoximation of the spherical Bessesl function :math:`j_2(Q)`.

    Parameters
    ----------
    Q : ndarray
        Magnitude of wavevector :math:`Q`
    A : float
        :math:`A_2` constant
    a : float
        :math:`a_2` constant
    B : float
        :math:`B_2` constant
    b : float
        :math:`b_2` constant
    C : float
        :math:`C_2` constant
    c : float
        :math:`c_2` constant
    D : float
        :math:`D_2` constant

    Returns
    -------
    j2 : ndarray
        Has the same shape as the input wavevector

    """

    s = Q/4/np.pi

    return A*s**2*np.exp(-a*s**2)+\
           B*s**2*np.exp(-b*s**2)+\
           C*s**2*np.exp(-c*s**2)+\
           D*s**2

def f(Q, j0, j2=0, K2=0):
    """
    Magnetic form factor :math:`f(Q)`

    Parameters
    ----------
    Q : ndarray
        Magnitude of wavevector
    j0, j2 : ndarray
        :math:`j_0` and :math:`j_2` constant with same shape as wavevector
    K2 : ndarray, optional
        Coupling constant, defualt ``K2=0``

    Returns
    -------
    f : ndarray
        Has the same shape as the input wavevector

    """

    return j0+K2*j2

def form(Q, ions, g=2):
    """
    Magnetic form factor :math:`f(Q)`

    Parameters
    ----------
    Q : 1d array
        Magnitude of wavevector
    ions : 1d array
        Magnetic ions
    g : float, ndarray, optional
       :math:`g` factor of the spins, defualt ``g=2``

    Returns
    -------
    f : ndarray
        Has the same shape as the input wavevector

    """

    k = 2/g-1

    n_hkl = Q.shape[0]
    n_atm = len(ions)

    factor = np.zeros(n_hkl*n_atm)

    for i, ion in enumerate(ions):

        if (tables.j0.get(ion) is None):
            A0, a0, B0, b0, C0, c0, D0 = 0, 0, 0, 0, 0, 0, 0
            A2, a2, B2, b2, C2, c2, D2 = 0, 0, 0, 0, 0, 0, 0
        else:
            A0, a0, B0, b0, C0, c0, D0 = tables.j0.get(ion)
            A2, a2, B2, b2, C2, c2, D2 = tables.j2.get(ion)

        if (np.size(k) > 1):
            K = k[i]
        else:
            K = k

        factor[i::n_atm] = f(Q, j0(Q, A0, a0, B0, b0, C0, c0, D0),\
                                j2(Q, A2, a2, B2, b2, C2, c2, D2), K)

        factor[factor < 0] = 0

    return factor

def spin(nu, nv, nw, n_atm, value=1, fixed=True):
    """
    Generate random spin vectors.

    Parameters
    ----------
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the \
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell
    n_atm : int
        Number of atoms in the unit cell

    Returns
    -------
    Sx, Sy, Sz : ndarray
        Each array has a flattened shape of size ``nu*nw*nv*n_atm``

    """

    if (len(np.shape(value)) <= 1):

        if (len(np.shape(value)) == 0):
            V = np.full(n_atm, value)
        elif (len(np.shape(value)) == 1):
            V = value

        theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm)
        phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm))

        Sx = V*np.sin(phi)*np.cos(theta)
        Sy = V*np.sin(phi)*np.sin(theta)
        Sz = V*np.cos(phi)

    else:

        sign = 2*(np.random.rand(nu,nv,nw,n_atm) < 0.5)-1

        Sx, Sy, Sz = sign*value[0], sign*value[1], sign*value[2]

    if not fixed:

        U = np.random.rand(nu,nv,nw,n_atm)

        Sx *= U
        Sy *= U
        Sz *= U

    return Sx.flatten(), Sy.flatten(), Sz.flatten()

def transform(Sx, Sy, Sz, H, K, L, nu, nv, nw, n_atm):
    """
    Discrete Fourier transform of spin vectors.

    Parameters
    ----------
    Sx, Sy, Sz : 1d array
        Spin vector component :math:`S_x`, :math:`S_y`, and :math:`S_z` in \
        Cartesian components along the :math:`x`, :math:`y`, and \
        :math:`z`-direction
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
    Sx_k, Sy_k, Sz_k : ndarray
        Each array has a flattened shape of size ``nu*nw*nv*n_atm``
    i_dft : ndarray, int
        Array has a flattened shape of size ``nu*nw*nv*n_atm``

    """

    Sx_k = np.fft.ifftn(Sx.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw
    Sy_k = np.fft.ifftn(Sy.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw
    Sz_k = np.fft.ifftn(Sz.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)

    i_dft = Kw+nw*(Kv+nv*Ku)

    return Sx_k.flatten(), Sy_k.flatten(), Sz_k.flatten(), i_dft

def intensity(Qx_norm, Qy_norm, Qz_norm, Sx_k, Sy_k, Sz_k, i_dft, factors):
    """
    Magnetic scattering intensity.

    Parameters
    ----------
    Qx_norm, Qy_norm, Qz_norm : 1d array
        Normalized wavevector component :math:`\hat{Q}_x`, :math:`\hat{Q}_y`, \
        and :math:`\hat{Q}_z` 
    Sx_k, Sy_k, Sz_k : 1d array
        Fourier transform of the spin vector component :math:`S_x`, \
        :math:`S_y`, and :math:`S_z` component 
    i_dft: 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: ndarray
        Prefactors of form factors and phase factors

    Returns
    -------
    I : 1d array
        Array has a flattened shape of size ``i_dft.shape[0]``

    """

    n_peaks = i_dft.shape[0]

    n_atm = factors.shape[0] // n_peaks

    factors = factors.reshape(n_peaks,n_atm)

    n_uvw = Sx_k.shape[0] // n_atm

    Sx_k = Sx_k.reshape(n_uvw,n_atm)
    Sy_k = Sy_k.reshape(n_uvw,n_atm)
    Sz_k = Sz_k.reshape(n_uvw,n_atm)

    prod_x = factors*Sx_k[i_dft,:]
    prod_y = factors*Sy_k[i_dft,:]
    prod_z = factors*Sz_k[i_dft,:]

    Fx = np.sum(prod_x, axis=1)
    Fy = np.sum(prod_y, axis=1)
    Fz = np.sum(prod_z, axis=1)

    Q_norm_dot_F = Qx_norm*Fx+Qy_norm*Fy+Qz_norm*Fz

    Fx_perp = Fx-Q_norm_dot_F*Qx_norm
    Fy_perp = Fy-Q_norm_dot_F*Qy_norm
    Fz_perp = Fz-Q_norm_dot_F*Qz_norm

    I = np.real(Fx_perp)**2+np.imag(Fx_perp)**2\
      + np.real(Fy_perp)**2+np.imag(Fy_perp)**2\
      + np.real(Fz_perp)**2+np.imag(Fz_perp)**2

    return I/(n_uvw*n_atm)

def structure(Qx_norm, Qy_norm, Qz_norm, Sx_k, Sy_k, Sz_k, i_dft, factors):
    """
    Partial magnetic structure factor.

    Parameters
    ----------
    Qx_norm, Qy_norm, Qz_norm : 1d array
        Normalized wavevector component :math:`\hat{Q}_x`, :math:`\hat{Q}_y`, \
        and :math:`\hat{Q}_z` 
    Sx_k, Sy_k, Sz_k : 1d array
        Fourier transform of the spin vector component :math:`S_x`, \
        :math:`S_y`, and :math:`S_z` component
    i_dft: ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: ndarray
        Prefactors of form factors and phase factors

    Returns
    -------
    Fx, Fy, Fz : ndarray
        Each array has a flattened shape of size ``i_dft.shape[0]``
    prod_x, prod_y, prod_z : ndarray
        Each array has a flattened shape of size ``i_dft.shape[0]*n_atm``

    """

    n_peaks = i_dft.shape[0]

    n_atm = factors.shape[0] // n_peaks

    factors = factors.reshape(n_peaks,n_atm)

    n_uvw = Sx_k.shape[0] // n_atm

    Sx_k = Sx_k.reshape(n_uvw,n_atm)
    Sy_k = Sy_k.reshape(n_uvw,n_atm)
    Sz_k = Sz_k.reshape(n_uvw,n_atm)

    prod_x = factors*Sx_k[i_dft,:]
    prod_y = factors*Sy_k[i_dft,:]
    prod_z = factors*Sz_k[i_dft,:]

    Fx = np.sum(prod_x, axis=1)
    Fy = np.sum(prod_y, axis=1)
    Fz = np.sum(prod_z, axis=1)

    return Fx, Fy, Fz, prod_x.flatten(), prod_y.flatten(), prod_z.flatten()

def magnitude(mu1, mu2, mu3, C):

    M = np.array([mu1,mu2,mu3])
    n = np.size(mu1)

    M = M.reshape(3,n)

    mu = []
    for i in range(n):
        mu.append(np.linalg.norm(np.dot(C, M[:,i])))

    return np.array(mu)

def cartesian(mu1, mu2, mu3, C):

    M = np.array([mu1,mu2,mu3])
    n = np.size(mu1)

    M = M.reshape(3,n)

    mux, muy, muz = [], [], []
    for i in range(n):
        Mp = np.dot(C, M[:,i])
        mux.append(Mp[0])
        muy.append(Mp[1])
        muz.append(Mp[2])

    return np.array(mux), np.array(muy), np.array(muz)