#!/usr/bin/env python3

import numpy as np

from scipy import constants, spatial
from disorder.material import tables

from disorder.diffuse.powder import magnetic

c = (0.5*constants.value('classical electron radius')*
      constants.value('neutron mag. mom. to nuclear magneton ratio')*
  0.1/constants.femto)**2

def j0(Q, A, a, B, b, C, c, D):
    """
    Appoximation of the spherical Bessesl function :math:`j_0`.

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
    Appoximation of the spherical Bessesl function :math:`j_2`.

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
    Magnetic form factor :math:`f`

    Parameters
    ----------
    Q : ndarray
        Magnitude of wavevector
    j0 : ndarray
        :math:`j_0` constant with same shape as wavevector
    j2 : ndarray, optional
        :math:`j_2` constant with same shape as wavevector
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
    Magnetic form factor :math:`f`

    Parameters
    ----------
    Q : ndarray
        Magnitude of wavevector
    ions : ndarray
        :math:`j_0` constant with same shape as wavevector
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

def spin(nu, nv, nw, n_atm, value=1):
    """
    Generate random spin vectors.

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

    Returns
    -------
    Sx, Sy, Sz : ndarray
        Each array has a flattened shape of size ``nu*nw*nv*n_atm``

    """

    theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm)
    phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm))

    Sx = value*np.sin(phi)*np.cos(theta)
    Sy = value*np.sin(phi)*np.sin(theta)
    Sz = value*np.cos(phi)

    return Sx.flatten(), Sy.flatten(), Sz.flatten()

def transform(Sx,
              Sy,
              Sz,
              H,
              K,
              L,
              nu,
              nv,
              nw,
              n_atm):
    """
    Discrete Fourier transform of spin vectors.

    Parameters
    ----------
    Sx : ndarray
        Spin vector component :math:`S_x` in Cartesian components along the \
        `x`-direction
    Sy : ndarray
        Spin vector component :math:`S_y` in Cartesian components along the \
        `y`-direction
    Sz : ndarray
        Spin vector component :math:`S_z` in Cartesian components along the \
        `z`-direction
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

def intensity(Qx_norm,
              Qy_norm,
              Qz_norm,
              Sx_k,
              Sy_k,
              Sz_k,
              i_dft,
              factors):
    """
    Magnetic scattering intensity.

    Parameters
    ----------
    Qx_norm : ndarray
        :math:`\hat{Q}_x` component of normalized wavevector
    Qy_norm : ndarray
        :math:`\hat{Q}_y` component of normalized wavevector
    Qz_norm : ndarray
        :math:`\hat{Q}_z` component of normalized wavevector
    Sx_k : ndarray
        Fourier transform of :math:`S_x` component of the spin vector
    Sy_k : ndarray
        Fourier transform of :math:`S_y` component of the spin vector
    Sz_k : ndarray
        Fourier transform of :math:`S_z` component of the spin vector
    i_dft: ndarray, int
        Array indices of Fourier transform corresponding to reciprocal space
    factors: ndarray
        Prefactors of form factors and phase factors

    Returns
    -------
    I : ndarray
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

def structure(Qx_norm,
              Qy_norm,
              Qz_norm,
              Sx_k,
              Sy_k,
              Sz_k,
              i_dft,
              factors):
    """
    Partial magnetic structure factor.

    Parameters
    ----------
    Qx_norm : ndarray
        :math:`\hat{Q}_x` component of normalized wavevector
    Qy_norm : ndarray
        :math:`\hat{Q}_y` component of normalized wavevector
    Qz_norm : ndarray
        :math:`\hat{Q}_z` component of normalized wavevector
    Sx_k : ndarray
        Fourier transform of :math:`S_x` component of the spin vector
    Sy_k : ndarray
        Fourier transform of :math:`S_y` component of the spin vector
    Sz_k : ndarray
        Fourier transform of :math:`S_z` component of the spin vector
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

def powder(Q, Sx, Sy, Sz, rx, ry, rz, form_factor, fract=0.5):

    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)

    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)

    pairs = tree.query_pairs(fract*r_max)

    coordinate = np.array(list(pairs))

    i = coordinate[:,0]
    j = coordinate[:,1]

    n_hkl = Q.shape[0]
    n_xyz = Sx.shape[0]

    n_atm = form_factor.shape[0] // n_hkl

    k = np.mod(i,n_atm)
    l = np.mod(j,n_atm)

    rx_ij = rx[j]-rx[i]
    ry_ij = ry[j]-ry[i]
    rz_ij = rz[j]-rz[i]

    r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)

    m = np.arange(n_xyz)
    n = np.mod(m,n_atm)

    S_i_dot_S_i = Sx[m]*Sx[m]+Sy[m]*Sy[m]+Sz[m]*Sz[m]

    S_i_dot_S_j = Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j]

    S_i_dot_r_ij = (Sx[i]*rx_ij[:]+Sy[i]*ry_ij[:]+Sz[i]*rz_ij[:])/r_ij[:]
    S_j_dot_r_ij = (Sx[j]*rx_ij[:]+Sy[j]*ry_ij[:]+Sz[j]*rz_ij[:])/r_ij[:]

    S_i_dot_r_ij_S_j_dot_r_ij = S_i_dot_r_ij*S_j_dot_r_ij

    A_ij = S_i_dot_S_j-S_i_dot_r_ij_S_j_dot_r_ij
    B_ij = 3*S_i_dot_r_ij_S_j_dot_r_ij-S_i_dot_S_j

    summation = np.zeros(Q.shape[0])
    
    auto = np.zeros(Q.shape[0])

    magnetic(summation, 
             auto,
             Q,
             r_ij,
             form_factor,
             A_ij,
             B_ij,
             S_i_dot_S_i,
             k,
             l,
             n,
             n_xyz,
             n_atm)

    I = 2*(auto/3+summation)/n_xyz

    return I