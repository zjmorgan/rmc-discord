#!/usr/bin/env python

import numpy as np

def expansion(nu, nv, nw, n_atm, value=1, fixed=True):
    """
    Generate random displacement vectors.

    Parameters
    ----------
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell.
    n_atm : int
        Number of atoms in the unit cell.
    value : 1d array, optional
        Magnitude of displacement vector, default ``value=1``.

    Returns
    -------
    Ux, Uy, Uz : 1d array
        Each array has a flattened shape of size ``nu*nv*nw*n_atm``.

    """

    if (len(np.shape(value)) == 0):
        Vxx = Vyy = Vzz = np.full(n_atm, value)
        Vyz = Vxz = Vxy = np.full(n_atm, 0)
    elif (len(np.shape(value)) == 1):
        Vxx = Vyy = Vzz = value
        Vyz = Vxz = Vxy = np.full(n_atm, 0)
    else:
        Vxx, Vyy, Vzz = value[0], value[1], value[2]
        Vyz, Vxz, Vxy = value[3], value[4], value[5]

    if fixed:

        theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm)
        phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm))

        nx = np.sin(phi)*np.cos(theta)
        ny = np.sin(phi)*np.sin(theta)
        nz = np.cos(phi)

        U = np.sqrt(Vxx*nx*nx+Vyy*ny*ny+Vzz*nz*nz\
                +2*(Vxz*nx*nz+Vyz*ny*nz+Vxy*nx*ny))

        Ux = U*nx
        Uy = U*ny
        Uz = U*nz

    else:

        L, V = np.zeros((3,3,n_atm)), np.zeros((3,3,n_atm))

        V[0,0,:] = Vxx
        V[1,1,:] = Vyy
        V[2,2,:] = Vzz
        V[1,2,:] = V[2,1,:] = Vyz
        V[0,2,:] = V[2,0,:] = Vxz
        V[0,1,:] = V[1,0,:] = Vxy

        for i in range(n_atm):
            if np.all(np.linalg.eigvals(V[...,i]) > 0):
                L[...,i] = np.linalg.cholesky(V[...,i])

        U = np.random.normal(loc=0,
                             scale=1,
                             size=3*nu*nv*nw*n_atm).reshape(3,nu,nv,nw,n_atm)

        Ux = U[0,...]*L[0,0,:]
        Uy = U[0,...]*L[1,0,:]+U[1,...]*L[1,1,:]
        Uz = U[0,...]*L[2,0,:]+U[1,...]*L[2,1,:]+U[2,...]*L[2,2,:]

    return Ux.flatten(), Uy.flatten(), Uz.flatten()

def number(n):
    """
    :math:`n`-th triangular number.

    Parameters
    ----------
    n : int
        Number.

    Returns
    -------
    int
       Triangular number.

    """

    return (n+1)*(n+2) // 2

def numbers(n):
    """
    Cumulative sum of :math:`0\dots n` triangular numbers.

    Parameters
    ----------
    n : int
        Number.

    Returns
    -------
    int
       Cumulative sum.

    """

    return (n+1)*(n+2)*(n+3) // 6

def indices(p):
    """
    Even and odd indices for the Taylor expansion.

    Parameters
    ----------
    p : int
        Order of the Taylor expansion.

    Returns
    -------
    even, odd : 1d array, int
       Indices for the even and odd terms.

    """

    tri_numbers = number(np.arange(p+1))

    total_terms = numbers(np.arange(p+1))

    first_index = total_terms-tri_numbers

    split = [np.arange(j,k) for j, k in zip(first_index,total_terms)]

    return np.concatenate(split[0::2]), np.concatenate(split[1::2])

def factorial(n):
    """
    Factorial :math:`n!`.

    Parameters
    ----------
    n : int
        Number.

    Returns
    -------
    int
        Factorial of the number.

    """

    if (n == 1 or n == 0):
        return 1
    else:
        return n*factorial(n-1)

def coefficients(p):
    """
    Coefficients for the Taylor expansion product.

    Parameters
    ----------
    p : int
        Order of the Taylor expansion.

    Returns
    -------
    coeffs : 1d array, complex
        Array of coefficients

    """

    coeffs = np.zeros(numbers(p), dtype=complex)

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

    V = np.ones((numbers(p),n))

    j = 0
    for i in range(p+1):
        for w in range(i+1):
            for v in range(i+1):
                for u in range(i+1):
                    if (u+v+w == i):
                        V[j,:] = Vx**u*Vy**v*Vz**w
                        j += 1

    return V.flatten()

def transform(U_r, H, K, L, nu, nv, nw, n_atm):
    """
    Discrete Fourier transform of Taylor expansion displacement products.

    Parameters
    ----------
    U_r : 1d array
          Displacement parameter :math:`U` (in Cartesian coordinates).
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
    i_dft : 1d array, int
        Array has a flattened shape of size ``nu*nw*nv*n_atm``.

    """

    n_uvw = nu*nv*nw

    n_prod = U_r.shape[0] // (n_uvw*n_atm)

    U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))*n_uvw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)

    i_dft = Kw+nw*(Kv+nv*Ku)

    return U_k.flatten(), i_dft

def intensity(U_k, Q_k, coeffs, cond, p, i_dft, factors, subtract=True):
    """
    Displacive scattering intensity.

    Parameters
    ----------
    U_k : 1d array
        Fourier transform of Taylor expansion displacement products.
    Q_k : 1d array
        Fourier transform of Taylor expansion wavevector products.
    coeffs : 1d array
        Taylor expansion coefficients
    cond : 1d array
        Array indices corresponding to nuclear Bragg peaks.
    p : int
        Order of Taylor expansion
    i_dft : 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space.
    factors : 1d array
        Prefactors of form factors, phase factors, and composition factors.
    subtract : boolean, optional
       Optionally subtract the Bragg intensity or return the Bragg structure 
       factor.

    Returns
    -------
    I : 1d array
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``.
    F_bragg : 1d array
        Array has a flattened shape of size ``coeffs.shape[0]*i_dft.shape[0]``.

    """

    n_prod = coeffs.shape[0]

    n_hkl = i_dft.shape[0]

    n_atm = factors.shape[0] // n_hkl

    factors = factors.reshape(n_hkl,n_atm)

    n_uvw = U_k.shape[0] // n_prod // n_atm

    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_hkl)

    even, odd = indices(p)

    V_k = np.einsum('ijk,kj->ji', coeffs*U_k[:,i_dft,:].T, Q_k)
    V_k_nuc = np.einsum('ijk,kj->ji', (coeffs[even]*U_k[:,i_dft,:][even,:].T),
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

def structure(U_k, Q_k, coeffs, cond, p, i_dft, factors):
    """
    Partial displacive structure factor.

    Parameters
    ----------
    U_k : 1d array
        Fourier transform of Taylor expansion displacement products.
    Q_k : 1d array
        Fourier transform of Taylor expansion wavevector products.
    coeffs : 1d array
        Taylor expansion coefficients.
    cond : 1d array
        Array indices corresponding to nuclear Bragg peaks.
    p : int
        Order of Taylor expansion.
    i_dft : 1d array, int
        Array indices of Fourier transform corresponding to reciprocal space.
    factors : 1d array
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

    n_hkl = i_dft.shape[0]

    n_atm = factors.shape[0] // n_hkl

    factors = factors.reshape(n_hkl,n_atm)

    n_uvw = U_k.shape[0] // n_prod // n_atm

    U_k = U_k.reshape(n_prod,n_uvw,n_atm)
    Q_k = Q_k.reshape(n_prod,n_hkl)

    even, odd = indices(p)

    V_k = np.einsum('ijk,kj->ji', coeffs*U_k[:,i_dft,:].T, Q_k)
    V_k_nuc = np.einsum('ijk,kj->ji', (coeffs[even]*U_k[:,i_dft,:][even,:].T),
                        Q_k[even,:])[cond]

    prod = factors*V_k
    prod_nuc = factors[cond,:]*V_k_nuc

    F = np.sum(prod, axis=1)
    F_nuc = np.sum(prod_nuc, axis=1)

    bragg = np.arange(n_hkl)[cond]

    return F, F_nuc, prod.flatten(), prod_nuc.flatten(), \
           V_k.flatten(), V_k_nuc.flatten(), even, bragg

def parameters(Ux, Uy, Uz, D, n_atm):

    Uxx = np.mean((Ux**2).reshape(Ux.size // n_atm, n_atm), axis=0)
    Uyy = np.mean((Uy**2).reshape(Uy.size // n_atm, n_atm), axis=0)
    Uzz = np.mean((Uz**2).reshape(Ux.size // n_atm, n_atm), axis=0)
    Uyz = np.mean((Uy*Uz).reshape(Ux.size // n_atm, n_atm), axis=0)
    Uxz = np.mean((Ux*Uz).reshape(Uy.size // n_atm, n_atm), axis=0)
    Uxy = np.mean((Ux*Uy).reshape(Uz.size // n_atm, n_atm), axis=0)

    U11 = np.zeros(n_atm)
    U22 = np.zeros(n_atm)
    U33 = np.zeros(n_atm)
    U23 = np.zeros(n_atm)
    U13 = np.zeros(n_atm)
    U12 = np.zeros(n_atm)

    D_inv = np.linalg.inv(D)

    for i in range(n_atm):

        Up = np.array([[Uxx[i], Uxy[i], Uxz[i]],
                       [Uxy[i], Uyy[i], Uyz[i]],
                       [Uxz[i], Uyz[i], Uzz[i]]])

        U = np.dot(np.dot(D_inv, Up), D_inv.T)

        U11[i] = U[0,0]
        U22[i] = U[1,1]
        U33[i] = U[2,2]
        U23[i] = U[1,2]
        U13[i] = U[0,2]
        U12[i] = U[0,1]

    return U11, U22, U33, U23, U13, U12

def equivalent(Uiso, D):
    """
    Components of atomic displacement parameters in crystal coordiantes
    :math:`U_{11}`, :math:`U_{22}`, :math:`U_{33}`, :math:`U_{23}`,
    :math:`U_{13}`, and :math:`U_{12}`.

    Parameters
    ----------
    Uiso : 1d array
        Isotropic atomic displacement parameters :math:`U_\mathrm{iso}`.
    D : 2d array, 3x3 
        Transform matrix from crystal axis to Cartesian coordiante system.

    Returns
    -------
    U11, U22, U33, U23, U13, U12 : float or 1d array
        Has same size as input isotropic atomic displacement parameters.

    """

    uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))

    U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
    U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]

    return U11, U22, U33, U23, U13, U12

def isotropic(U11, U22, U33, U23, U13, U12, D):
    """
    Equivalent isotropic displacement parameters :math:`U_\mathrm{iso}`.

    Parameters
    ----------
    U11, U22, U33, U23, U13, U12 : float or 1d array
        Components of atomic displacement parameters :math:`U_{11}`, 
        :math:`U_{22}`, :math:`U_{33}`, :math:`U_{23}`, :math:`U_{13}`, 
        and :math:`U_{12}`.
    D : 2d array, 3x3 
        Transform matrix from crystal axis to Cartesian coordiante system.

    Returns
    -------
    Uiso : 1d array
        Has same size as input atomic displacement parameter components.

    """

    U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
    n = np.size(U11)

    U = U.reshape(3,3,n)

    Uiso = []
    for i in range(n):
        Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D.T))
        Uiso.append(np.mean(Up).real)

    return np.array(Uiso)

def principal(U11, U22, U33, U23, U13, U12, D):
    """
    Principal atmoic displacement parameters :math:`U_\mathrm{1}, 
    :math:`U_\mathrm{2}`, and :math:`U_\mathrm{3}`.

    Parameters
    ----------
    U11, U22, U33, U23, U13, U12 : float or 1d array
        Components of atomic displacement parameters :math:`U_{11}`, 
        :math:`U_{22}`, :math:`U_{33}`, :math:`U_{23}`, :math:`U_{13}`, 
        and :math:`U_{12}`.
    D : 2d array, 3x3 
        Transform matrix from crystal axis to Cartesian coordiante system.

    Returns
    -------
    U1, U2, U3 : 1d array
        Has same size as input atomic displacement parameter components.

    """

    U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
    n = np.size(U11)

    U = U.reshape(3,3,n)

    U1, U2, U3 = [], [], []
    for i in range(n):
        Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D.T))
        Up.sort()
        U1.append(Up[0].real)
        U2.append(Up[1].real)
        U3.append(Up[2].real)

    return np.array(U1), np.array(U2), np.array(U3)

def cartesian(U11, U22, U33, U23, U13, U12, D):
    """
    Components of atomic displacement parameters in Cartesian coordiantes
    :math:`U_{xx}`, :math:`U_{yy}`, :math:`U_{zz}`, :math:`U_{yz}`,
    :math:`U_{xz}`, and :math:`U_{xy}`.

    Parameters
    ----------
    U11, U22, U33, U23, U13, U12 : float or 1d array
        Components of atomic displacement parameters :math:`U_{11}`, 
        :math:`U_{22}`, :math:`U_{33}`, :math:`U_{23}`, :math:`U_{13}`, 
        and :math:`U_{12}`.
    D : 2d array, 3x3
        Transform matrix from crystal axis to Cartesian coordiante system.

    Returns
    -------
    Uxx, Uyy, Uzz, Uyz, Uxz, Uxy : 1d array
        Has same size as input atomic displacement parameter components.

    """

    U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
    n = np.size(U11)

    U = U.reshape(3,3,n)

    Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []
    for i in range(n):
        Up = np.dot(np.dot(D, U[...,i]), D.T)
        Uxx.append(Up[0,0])
        Uyy.append(Up[1,1])
        Uzz.append(Up[2,2])
        Uyz.append(Up[1,2])
        Uxz.append(Up[0,2])
        Uxy.append(Up[0,1])

    return np.array(Uxx), np.array(Uyy), np.array(Uzz), \
           np.array(Uyz), np.array(Uxz), np.array(Uxy)