#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport sqrt, fabs

from disorder.material import crystal, symmetry

cdef bint sign(double a) nogil:

    return (a > 0)-(a < 0)

cdef bint iszero(double a) nogil:

    cdef double atol = 1e-08

    return fabs(a) <= atol

def vector_correlation(double [::1] Sx,
                       double [::1] Sy,
                       double [::1] Sz,
                       signed long long [::1] counts,
                       signed long long [::1] search,
                       long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_corr_np = np.zeros(N+1)
    C_corr_np[N] = 1

    cdef double [::1] C_corr = C_corr_np

    cdef Py_ssize_t r, s, i, j

    cdef double U, V

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            U = sqrt(Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i])
            V = sqrt(Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j])

            if not (iszero(U) or iszero(V)):
                C_corr[r] += (Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j])/U/V
            else:
                count -= 1

        if (count > 0):
            C_corr[r] /= count
        else:
            C_corr[r] = 0

    return C_corr_np

def vector_average(double [::1] Sx,
                   double [::1] Sy,
                   double [::1] Sz,
                   signed long long [::1] counts,
                   signed long long [::1] search,
                   long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_corr_np = np.zeros(N+1)
    C_corr_np[N] = 1

    cdef double [::1] C_corr = C_corr_np

    cdef Py_ssize_t r, s, i, j

    cdef double U, V

    cdef double Sx_i, Sy_i, Sz_i
    cdef double Sx_j, Sy_j, Sz_j

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        Sx_i, Sy_i, Sz_i = 0, 0, 0
        Sx_j, Sy_j, Sz_j = 0, 0, 0

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            U = sqrt(Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i])
            V = sqrt(Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j])

            if not (iszero(U) or iszero(V)):
                Sx_i += Sx[i]/U
                Sy_i += Sy[i]/U
                Sz_i += Sz[i]/U

                Sx_j += Sx[j]/V
                Sy_j += Sy[j]/V
                Sz_j += Sz[j]/V
            else:
                count -= 1

        if (count > 0):
            C_corr[r] = (Sx_i*Sx_j+Sy_i*Sy_j+Sz_i*Sz_j)/count**2
        else:
            C_corr[r] = 0

    return C_corr_np

def vector_collinearity(double [::1] Sx,
                        double [::1] Sy,
                        double [::1] Sz,
                        signed long long [::1] counts,
                        signed long long [::1] search,
                        long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_coll_np = np.zeros(N+1)
    C_coll_np[N] = 1

    cdef double [::1] C_coll = C_coll_np

    cdef Py_ssize_t r, s, i, j

    cdef double U, V

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            U = Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i]
            V = Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j]

            if not (iszero(U) or iszero(V)):
                C_coll[r] += (Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j])**2/U/V
            else:
                count -= 1

        if (count > 0):
            C_coll[r] /= count
        else:
            C_coll[r] = 0

    return C_coll_np

def scalar_correlation(double [::1] S,
                       signed long long [::1] counts,
                       signed long long [::1] search,
                       long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_corr_np = np.zeros(N+1)
    C_corr_np[N] = 1

    cdef double [::1] C_corr = C_corr_np

    cdef Py_ssize_t r, s, i, j

    cdef double U, V

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            U = fabs(S[i])
            V = fabs(S[j])

            if not (iszero(U) or iszero(V)):
                C_corr[r] += (S[i]*S[j])/U/V
            else:
                count -= 1

        if (count > 0):
            C_corr[r] /= count
        else:
            C_corr[r] = 0

    return C_corr_np

def scalar_average(double [::1] S,
                   signed long long [::1] counts,
                   signed long long [::1] search,
                   long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_corr_np = np.zeros(N+1)
    C_corr_np[N] = 1

    cdef double [::1] C_corr = C_corr_np

    cdef Py_ssize_t r, s, i, j

    cdef double U, V

    cdef double S_i, S_j

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        S_i, S_j = 0, 0

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            U = fabs(S[i])
            V = fabs(S[j])

            if not (iszero(U) or iszero(V)):
                S_i += S[i]/U
                S_j += S[j]/V
            else:
                count -= 1

        if (count > 0):
            C_corr[r] = S_i*S_j/count**2
        else:
            C_corr[r] = 0

    return C_corr_np

def scalar_vector_correlation(double [::1] S,
                              double [::1] Sx,
                              double [::1] Sy,
                              double [::1] Sz,
                              double [::1] rx,
                              double [::1] ry,
                              double [::1] rz,
                              signed long long [::1] counts,
                              signed long long [::1] search,
                              long long [:,::1] coordinate):

    cdef Py_ssize_t N = counts.shape[0]

    C_corr_np = np.zeros(N+1)
    C_corr_np[N] = 0

    cdef double [::1] C_corr = C_corr_np

    cdef Py_ssize_t r, s, i, j

    cdef double rx_ij, ry_ij, rz_ij, Sx_ij, Sy_ij, Sz_ij

    cdef double U, V, metric

    cdef Py_ssize_t count

    for r in range(N):

        count = counts[r]

        for s in range(search[r],search[r+1]):

            i, j = coordinate[s,0], coordinate[s,1]

            rx_ij = rx[j]-rx[i]
            ry_ij = ry[j]-ry[i]
            rz_ij = rz[j]-rz[i]

            Sx_ij = Sx[j]-Sx[i]
            Sy_ij = Sy[j]-Sy[i]
            Sz_ij = Sz[j]-Sz[i]

            U = sqrt(rx_ij*rx_ij+ry_ij*ry_ij+rz_ij*rz_ij)
            V = sqrt(Sx_ij*Sx_ij+Sy_ij*Sy_ij+Sz_ij*Sz_ij)

            metric = 0.25*((1+sign(S[i]))*(1-sign(S[j]))
                          +(1+sign(S[j]))*(1-sign(S[i])))

            if not (iszero(U) or iszero(V)) and metric > 0.5:
                C_corr[r] += (rx_ij*Sx_ij+ry_ij*Sy_ij+rz_ij*Sz_ij)/U/V
            else:
                count -= 1

        if (count > 0):
            C_corr[r] /= count
        else:
            C_corr[r] = 0

    return C_corr_np

@cython.binding(True)
def pairs1d(rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Generate spherically averaged pairs.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    d : 1d array
        Separation distance magnitude.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.
    counts : 1d array, int
        Number of pair counts.
    search : 1d array, int
        Index of first appearance in sorted pairs.
    coordinate : 2d array, int
        Coordinate (i,j) pairs.
    unique_pairs : int
        Number of unique pairs.

    """

    n_atm = ion.shape[0] // (nu*nv*nw)

    mu = (nu+1) // 2
    mv = (nv+1) // 2
    mw = (nw+1) // 2

    m_uvw = mu*mv*mw
    n_uvw = nu*nv*nw

    m_xyz = mu*mv*mw*n_atm
    n_xyz = nu*nv*nw*n_atm

    c_uvw = np.arange(n_uvw, dtype=int)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    i = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm)).flatten()
    j = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm)).flatten()

    ic = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm)).flatten()
    jc = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm)).flatten()

    i, j = np.concatenate((i,ic)), np.concatenate((j,jc))

    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]

    du, dv, dw = crystal.transform(dx, dy, dz, np.linalg.inv(A))

    du[du < -mu] += nu
    dv[dv < -mv] += nv
    dw[dw < -mw] += nw

    du[du > mu] -= nu
    dv[dv > mv] -= nv
    dw[dw > mw] -= nw

    dx, dy, dz = crystal.transform(du, dv, dw, A)

    d = np.sqrt(dx**2+dy**2+dz**2)

    mask = d <= d.max()*fract

    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    d = d[mask]

    i = i[mask]
    j = j[mask]

    coordinate = np.column_stack((i,j))

    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)

    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])

    ions, ion_labels = np.unique(ion_ion, return_inverse=True)

    metric = np.stack((np.round(np.round(d/tol,1)).astype(int), \
                       ion_labels)).T

    sort = np.lexsort(np.fliplr(metric).T)

    d = d[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]

    ion_ion = ion_ion[sort]

    metric = metric[sort]

    unique, indices, counts = np.unique(metric,
                                        axis=0,
                                        return_index=True,
                                        return_counts=True)

    search = np.append(indices,len(d))

    d = d[indices]
    pairs = ion_ion[indices]

    d = np.append(d, 0)
    pairs = np.append(pairs, '0')

    return d, pairs, counts, search, coordinate, unique.shape[0]+1

@cython.binding(True)
def pairs3d(rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Generate three-dimensional pairs.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.
    counts : 1d array, int
        Number of pair counts.
    search : 1d array, int
        Index of first appearance in sorted pairs.
    coordinate : 2d array, int
        Coordinate (i,j) pairs.
    unique_pairs : int
        Number of unique pairs.

    """

    n_atm = ion.shape[0] // (nu*nv*nw)

    mu = (nu+1) // 2
    mv = (nv+1) // 2
    mw = (nw+1) // 2

    m_uvw = mu*mv*mw
    n_uvw = nu*nv*nw

    m_xyz = mu*mv*mw*n_atm
    n_xyz = nu*nv*nw*n_atm

    c_uvw = np.arange(n_uvw, dtype=int)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    i = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm)).flatten()
    j = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm)).flatten()

    ic = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm)).flatten()
    jc = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm)).flatten()

    i, j = np.concatenate((i,ic)), np.concatenate((j,jc))

    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]

    du, dv, dw = crystal.transform(dx, dy, dz, np.linalg.inv(A))

    du[du < -mu] += nu
    dv[dv < -mv] += nv
    dw[dw < -mw] += nw

    du[du > mu] -= nu
    dv[dv > mv] -= nv
    dw[dw > mw] -= nw

    dx, dy, dz = crystal.transform(du, dv, dw, A)

    d = np.sqrt(dx**2+dy**2+dz**2)

    mask = d <= d.max()*fract

    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    d = np.stack((dx,dy,dz)).T

    i = i[mask]
    j = j[mask]

    coordinate = np.column_stack((i,j))

    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)

    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])

    ions, ion_labels = np.unique(ion_ion, return_inverse=True)

    metric = np.vstack((np.round(np.round(d.T/tol,1)).astype(int), \
                        ion_labels)).T

    sort = np.lexsort(np.fliplr(metric).T)

    d = d[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]

    ion_ion = ion_ion[sort]

    metric = metric[sort]

    unique, indices, counts = np.unique(metric,
                                        axis=0,
                                        return_index=True,
                                        return_counts=True)

    search = np.append(indices,len(d))

    dx = dx[sort][indices]
    dy = dy[sort][indices]
    dz = dz[sort][indices]
    pairs = ion_ion[indices]

    dx = np.append(dx, 0)
    dy = np.append(dy, 0)
    dz = np.append(dz, 0)
    pairs = np.append(pairs, '0')

    return dx, dy, dz, pairs, counts, search, coordinate, unique.shape[0]+1

@cython.binding(True)
def vector1d(Sx, Sy, Sz, rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Spherically averaged vector-pair correlations.

    Parameters
    ----------
    Sx, Sy, Sz : 1d array
        Vectors.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Correlated averge product function.
    C_coll : 1d array
        Collinear correlated averge product function.
    C_corr_ : 1d array
        Uncorrelated averge product function.
    C_coll_ : 1d array
        Collinear uncorrelated averge product function.
    d : 1d array
        Separation distance magnitude.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs1d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    d, pairs, counts, search, coordinate, N = data

    C_corr  = vector_correlation(Sx, Sy, Sz, counts, search, coordinate)
    C_coll  = vector_collinearity(Sx, Sy, Sz, counts, search, coordinate)
    C_corr_ = vector_average(Sx, Sy, Sz, counts, search, coordinate)

    C_coll_ = C_corr**2

    return C_corr, C_coll, C_corr_, C_coll_, d, pairs

@cython.binding(True)
def vector3d(Sx, Sy, Sz, rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Three-dimensional vector-pair correlations.

    Parameters
    ----------
    Sx, Sy, Sz : 1d array
        Vectors.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Correlated averge product function.
    C_coll : 1d array
        Collinear correlated averge product function.
    C_corr_ : 1d array
        Uncorrelated averge product function.
    C_coll_ : 1d array
        Collinear uncorrelated averge product function.
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs3d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    dx, dy, dz, pairs, counts, search, coordinate, N = data

    C_corr  = vector_correlation(Sx, Sy, Sz, counts, search, coordinate)
    C_coll  = vector_collinearity(Sx, Sy, Sz, counts, search, coordinate)
    C_corr_ = vector_average(Sx, Sy, Sz, counts, search, coordinate)

    C_coll_ = C_corr**2

    dx = np.concatenate((dx,-dx[:N-1]))
    dy = np.concatenate((dy,-dy[:N-1]))
    dz = np.concatenate((dz,-dz[:N-1]))

    pairs = np.concatenate((pairs,pairs[:N-1]))

    C_corr  = np.concatenate((C_corr,C_corr[:N-1]))
    C_coll  = np.concatenate((C_coll,C_coll[:N-1]))
    C_corr_ = np.concatenate((C_corr_,C_corr_[:N-1]))
    C_coll_ = np.concatenate((C_coll_,C_coll_[:N-1]))

    return C_corr, C_coll, C_corr_, C_coll_, dx, dy, dz, pairs

@cython.binding(True)
def scalar1d(S, rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Spherically averaged scalar-pair correlations.

    Parameters
    ----------
    S : 1d array
        Scalars.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Correlated averge product function.
    C_corr_ : 1d array
        Uncorrelated averge product function.
    d : 1d array
        Separation distance magnitude.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs1d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    d, pairs, counts, search, coordinate, N = data

    C_corr  = scalar_correlation(S, counts, search, coordinate)
    C_corr_ = scalar_average(S, counts, search, coordinate)

    return C_corr, C_corr_, d, pairs

@cython.binding(True)
def scalar3d(S, rx, ry, rz, ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Three-dimensional scalar-pair correlations.

    Parameters
    ----------
    S : 1d array
        Scalars.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Correlated averge product function.
    C_corr_ : 1d array
        Uncorrelated averge product function.
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs3d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    dx, dy, dz, pairs, counts, search, coordinate, N = data

    C_corr  = scalar_correlation(S, counts, search, coordinate)
    C_corr_ = scalar_average(S, counts, search, coordinate)

    dx = np.concatenate((dx,-dx[:N-1]))
    dy = np.concatenate((dy,-dy[:N-1]))
    dz = np.concatenate((dz,-dz[:N-1]))

    pairs = np.concatenate((pairs,pairs[:N-1]))

    C_corr  = np.concatenate((C_corr,C_corr[:N-1]))
    C_corr_ = np.concatenate((C_corr_,C_corr_[:N-1]))

    return C_corr, C_corr_, dx, dy, dz, pairs

@cython.binding(True)
def scalar_vector1d(S, Sx, Sy, Sz, rx, ry, rz, ion,
                    nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Spherically averaged scalar-pair correlations.

    Parameters
    ----------
    S : 1d array
        Scalars.
    Sx, Sy, Sz : 1d array
        Vectors.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Cross correlated averge product function.
    d : 1d array
        Separation distance magnitude.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs1d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    d, pairs, counts, search, coordinate, N = data

    C_corr = scalar_vector_correlation(S, Sx, Sy, Sz, rx, ry, rz,
                                       counts, search, coordinate)

    return C_corr, d, pairs

@cython.binding(True)
def scalar_vector3d(S, Sx, Sy, Sz, rx, ry, rz,
                    ion, nu, nv, nw, A, fract=0.25, tol=1e-4):
    """
    Three-dimensional scalar-pair correlations.

    Parameters
    ----------
    S : 1d array
        Scalars.
    Sx, Sy, Sz : 1d array
        Vectors.
    rx, ry, rz : 1d array
        Atomic positions.
    ion : 1d array, str
        Atoms, ions, or isotopes.
    nu, nv, nw : int
        Supercell size.
    A : 2d array, 3x3
        Transformation matrix.
    fract : float, optional
        Fraction of longest distance for radial cutoff. Default is ``0.25``.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    C_corr : 1d array
        Cross correlated averge product function.
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.

    """

    data = pairs3d(rx, ry, rz, ion, nu, nv, nw, A, fract, tol)

    dx, dy, dz, pairs, counts, search, coordinate, N = data

    C_corr = scalar_vector_correlation(S, Sx, Sy, Sz, rx, ry, rz,
                                       counts, search, coordinate)

    dx = np.concatenate((dx,-dx[:N-1]))
    dy = np.concatenate((dy,-dy[:N-1]))
    dz = np.concatenate((dz,-dz[:N-1]))

    pairs = np.concatenate((pairs,pairs[:N-1]))

    C_corr  = np.concatenate((C_corr,C_corr[:N-1]))

    return C_corr, dx, dy, dz, pairs

@cython.binding(True)
def symmetrize(arrays, dx, dy, dz, pairs, A, laue, tol=1e-4):
    """
    Symmetrization of three-dimensional correlations.

    Parameters
    ----------
    arrays : tuple
        Arrays to average.
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.
    A : 2d array, 3x3
        Transformation matrix.
    laue : str
        Laue class.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    symmetrize : tuple
        Symmetrized of arrays.
    dx_ave, dy_ave, dz_ave : 1d array
        Symmetrized separation distance vector.
    pairs_ave : 1d array, str
        Symmetrized atom, ion, or isotope-pairs.

    """

    arrays = np.hstack(list((arrays,)))

    if (arrays.ndim == 1):

        arrays = arrays[np.newaxis,:]

    M = arrays.shape[0]

    symops = np.array(symmetry.laue(laue))

    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])

    N = arrays.shape[1]

    A_inv = np.linalg.inv(A)

    total = []

    arr = []

    pair_list = []
    pair_labels = []

    unique, labels = np.unique(pairs, return_inverse=True)

    for n in range(N):

        x = A_inv[0,0]*dx[n]+A_inv[0,1]*dy[n]+A_inv[0,2]*dz[n]
        y = A_inv[1,0]*dx[n]+A_inv[1,1]*dy[n]+A_inv[1,2]*dz[n]
        z = A_inv[2,0]*dx[n]+A_inv[2,1]*dy[n]+A_inv[2,2]*dz[n]

        displacement = []

        for symop in symops:

            transformed = symmetry.evaluate([symop], [x,y,z], translate=False)

            displacement.append(transformed)

        symmetries = np.unique(np.vstack(displacement), axis=0)

        total.append(symmetries)

        arr.append(np.tile(arrays[:,n], symmetries.shape[0]))

        pair_list.append(np.tile(pairs[n], symmetries.shape[0]))
        pair_labels.append(np.tile(labels[n], symmetries.shape[0]))

    total = np.vstack(np.array(total, dtype=object)).astype(float)

    arr = np.hstack(np.array(arr, dtype=object)).astype(float)

    arr = arr.reshape(arr.shape[0] // M, M)

    pairs = np.hstack(np.array(pair_list, dtype=object))
    pair_labels = np.hstack(np.array(pair_labels, dtype=object)).astype(int)

    metric = np.vstack((np.round(np.round( \
                        total.astype(float).T/tol,1)).astype(int), \
                        pair_labels)).T

    sort = np.lexsort(np.fliplr(metric).T)

    arr = arr[sort]

    pairs = pairs[sort]

    unique, indices, counts = np.unique(metric[sort],
                                        axis=0,
                                        return_index=True,
                                        return_counts=True)

    search = np.append(indices,len(arr))

    D = unique.shape[0]

    u_symm = total[sort][indices,0]
    v_symm = total[sort][indices,1]
    w_symm = total[sort][indices,2]
    pairs_symm = pairs[indices]

    arrays_ave = np.zeros((M,D))

    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):
                arrays_ave[i,r] += arr[s,i]/counts[r]

    dx_symm = (A[0,0]*u_symm+A[0,1]*v_symm+A[0,2]*w_symm).astype(float)
    dy_symm = (A[1,0]*u_symm+A[1,1]*v_symm+A[1,2]*w_symm).astype(float)
    dz_symm = (A[2,0]*u_symm+A[2,1]*v_symm+A[2,2]*w_symm).astype(float)

    arrays_ave = arrays_ave.flatten()

    output = tuple(np.split(arrays_ave, M))
    output = (*output, dx_symm, dy_symm, dz_symm, pairs_symm)

    return output

@cython.binding(True)
def average1d(arrays, d, pairs, tol=1e-4):
    """
    Average of one-dimensional correlations.

    Parameters
    ----------
    arrays : tuple
        Arrays to average.
    d : 1d array
        Separation distance magnitude.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    average : tuple
        Averaged of arrays.
    d : 1d array
        Averaged separation distance magnitude.
    pairs_ave : 1d array, str
        Averaged atom, ion, or isotope-pairs.

    """

    arrays = np.hstack(list((arrays,)))

    if (arrays.ndim == 1):

        arrays = arrays[np.newaxis,:]

    M = arrays.shape[0]

    metric = (np.round(np.round(d/tol,1))).astype(int)

    sort = np.argsort(metric)

    metric = metric[sort]
    d = d[sort]
    arrays = arrays[:,sort]
    pairs = pairs[sort]

    unique, indices, counts = np.unique(metric,
                                        axis=0,
                                        return_index=True,
                                        return_counts=True)

    search = np.append(indices,len(d))

    D = unique.shape[0]

    arrays_ave = np.zeros((M,D))

    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):
                arrays_ave[i,r] += arrays[i][s]/counts[r]

    d_ave = d[indices]
    pairs_ave = pairs[indices]

    arrays_ave = arrays_ave.flatten()

    output = tuple(np.split(arrays_ave, M))
    output = (*output, d_ave, pairs_ave)

    return output

@cython.binding(True)
def average3d(arrays, dx, dy, dz, pairs, tol=1e-4):
    """
    Average of three-dimensional correlations.

    Parameters
    ----------
    arrays : tuple
        Arrays to average.
    dx, dy, dz : 1d array
        Separation distance vector.
    pairs : 1d array, str
        Atom, ion, or isotope-pairs.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-4``.

    Returns
    -------
    average : tuple
        Averaged of arrays.
    dx_ave, dy_ave, dz_ave : 1d array
        Averaged separation distance vector.
    pairs_ave : 1d array, str
        Averaged atom, ion, or isotope-pairs.

    """

    arrays = np.hstack(list((arrays,)))

    if (arrays.ndim == 1):

        arrays = arrays[np.newaxis,:]

    M = arrays.shape[0]

    d = np.stack((dx,dy,dz)).T

    metric = (np.round(np.round(d.astype(float)/tol,1))).astype(int)

    sort = np.lexsort(np.fliplr(metric).T)

    metric = metric[sort]
    dx = dx[sort]
    dy = dy[sort]
    dz = dz[sort]
    arrays = arrays[:,sort]
    pairs = pairs[sort]

    unique, indices, counts = np.unique(metric,
                                        axis=0,
                                        return_index=True,
                                        return_counts=True)

    search = np.append(indices,len(d))

    D = unique.shape[0]

    arrays_ave = np.zeros((M,D))

    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):
                arrays_ave[i,r] += arrays[i][s]/counts[r]

    dx_ave = dx[indices].astype(float)
    dy_ave = dy[indices].astype(float)
    dz_ave = dz[indices].astype(float)
    pairs_ave = pairs[indices]

    arrays_ave = arrays_ave.flatten()

    output = tuple(np.split(arrays_ave, M))
    output = (*output, dx_ave, dy_ave, dz_ave, pairs_ave)

    return output
