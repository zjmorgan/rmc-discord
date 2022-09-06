#!/usr/bin/env python

import numpy as np

from scipy.special import erfc

from disorder.material import crystal, symmetry

def __A(alpha,r):

    c = 2*alpha/np.sqrt(np.pi)

    return -(erfc(alpha*r)/r-c*np.exp(-alpha**2*r**2))/r**2

def __B(alpha,r):

    c = 2*alpha/np.sqrt(np.pi)

    return (erfc(alpha*r)/r+c*np.exp(-alpha**2*r**2))/r**2

def __C(alpha,r):

    c = 2*alpha*(3+2*alpha**2*r**2)/np.sqrt(np.pi)

    return (3*erfc(alpha*r)/r+c*np.exp(-alpha**2*r**2))/r**4

def atom_pairs_distance(rx, ry, rz, nu, nv, nw, n_atm, A, tol=1e-3):
    """
    Atom pairs.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    nu, nv, nw : int
        Supercell size.
    n_atm : int
        Number of unit cell atoms.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-3``.

    Returns
    -------
    dx, dy, dz : 1d array
        Separation distance vector.
    i, j : 1d array, int
        Coordinate pairs.
    inverse : 1d array, int
        Indices of the unique pair distances array that reconstruct all pairs.

    """

    A_inv = np.linalg.inv(A)

    mu = (nu+1)//2
    mv = (nv+1)//2
    mw = (nw+1)//2

    m_uvw = mu*mv*mw
    n_uvw = nu*nv*nw

    c_uvw = np.arange(n_uvw, dtype=int)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu).flatten()
    iv = np.mod(iv+cv[:,None], nv).flatten()
    iw = np.mod(iw+cw[:,None], nw).flatten()

    ju = np.mod(ju+cu[:,None], nu).flatten()
    jv = np.mod(jv+cv[:,None], nv).flatten()
    jw = np.mod(jw+cw[:,None], nw).flatten()

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    pairs.sort(axis=0)

    uni, _, _ = symmetry.unique(pairs)

    i_lat, j_lat = uni

    # ---

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    du, dv, dw = ju-iu, jv-iv, jw-iw

    distance = np.stack((du,dv,dw))

    metric = np.vstack(distance).T

    _, index, inverse = symmetry.unique(metric)

    i_lat = np.ravel_multi_index((iu[index],iv[index],iw[index]), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju[index],jv[index],jw[index]), (nu,nv,nw))

    # ---

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    ux = rx.reshape(nu,nv,nw,n_atm)[0,0,0,:]
    uy = ry.reshape(nu,nv,nw,n_atm)[0,0,0,:]
    uz = rz.reshape(nu,nv,nw,n_atm)[0,0,0,:]

    dx = ux[j_atm]-ux[i_atm]
    dy = uy[j_atm]-uy[i_atm]
    dz = uz[j_atm]-uz[i_atm]

    distance = np.stack((dx,dy,dz))

    metric = np.vstack(np.round(distance/tol,0)).astype(int).T

    _, ind, inv = symmetry.unique(metric)

    i_atm, j_atm = i_atm[ind], j_atm[ind]

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    i = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm)).flatten()
    j = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm)).flatten()

    ic = np.ravel_multi_index((0,i_atm[:,None]), (n_uvw,n_atm)).flatten()
    jc = np.ravel_multi_index((0,j_atm[:,None]), (n_uvw,n_atm)).flatten()

    i, j = np.concatenate((ic,i)), np.concatenate((jc,j))

    # ---

    dx, dy, dz = rx[j]-rx[i], ry[j]-ry[i], rz[j]-rz[i]

    du, dv, dw = crystal.transform(dx, dy, dz, A_inv)

    du[du < -mu] += nu
    dv[dv < -mv] += nv
    dw[dw < -mw] += nw

    du[du > mu] -= nu
    dv[dv > mv] -= nv
    dw[dw > mw] -= nw

    dx, dy, dz = crystal.transform(du, dv, dw, A)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    i = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm)).flatten()
    j = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm)).flatten()

    ic = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm)).flatten()
    jc = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm)).flatten()

    i, j = np.concatenate((ic,i)), np.concatenate((jc,j))

    l, m = ind.size, index.size

    k = np.concatenate((inv,l+inv,2*l+np.arange(n_atm)))

    p = (np.arange(n_uvw)*0+inv[:,None]).flatten()
    q = l+(inverse+m*k[:,None]).flatten()

    inverse = np.concatenate((p,q))

    return dx, dy, dz, i, j, inverse

def spatial_wavevector(nu, nv, nw, n_atm, B, R):
    """
    Spatial wavevector.

    Parameters
    ----------
    nu, nv, nw : int
        Supercell size.
    n_atm : int
        Number of unit cell atoms.
    B : 2d array, 3x3
        Reciprocal-space crystal axis to Cartesian transformation matrix.
    R : 2d array, 3x3
        Rotation matrix between real and reciprocal-space Cartesian axes.

    Returns
    -------
    Gx, Gy, Gz : 1d array
        Wavevector components.

    """

    mu = (nu+1)//2
    mv = (nv+1)//2
    mw = (nw+1)//2

    ku = 2*np.pi*np.arange(mu)/nu
    kv = 2*np.pi*np.concatenate((np.arange(mv),np.arange(-mv+1,0)))/nv
    kw = 2*np.pi*np.concatenate((np.arange(mw),np.arange(-mw+1,0)))/nw

    ku, kv, kw = np.meshgrid(ku, kv, kw, indexing='ij')

    ku, kv, kw = ku.flatten(), kv.flatten(), kw.flatten()
    ku, kv, kw = np.delete(ku, 0), np.delete(kv, 0), np.delete(kw, 0)

    Gx, Gy, Gz = crystal.transform(ku, kv, kw, B)
    Gx, Gy, Gz = crystal.transform(Gx, Gy, Gz, R)

    return Gx, Gy, Gz

def charge_charge_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):
    """
    Charge-charge matrix.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    nu, nv, nw : int
        Supercell size.
    n_atm : int
        Number of unit cell atoms.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    B : 2d array, 3x3
        Reciprocal-space crystal axis to Cartesian transformation matrix.
    R : 2d array, 3x3
        Rotation matrix between real and reciprocal-space Cartesian axes.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-3``.

    Returns
    -------
    Qij : 1d array
        Charge-charge matrix. Array of size ``n*(n+1)//2``.

    """

    n = nu*nv*nw*n_atm

    Qij = np.zeros(n*(n+1)//2)

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    k = j+n*i-(i+1)*i//2

    l = np.arange(n)
    l = l+n*l-(l+1)*l//2

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    Qij[k] = (erfc(alpha*d)/d)[inverse]

    Qij[l] = -2*alpha/np.sqrt(np.pi)

    cos_d_dot_G = np.cos(np.kron(dx,Gx)+
                         np.kron(dy,Gy)+
                         np.kron(dz,Gz))

    cos_d_dot_G = cos_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-np.pi**2*G_sq/alpha**2)/G_sq

    Qij[k] += (factors*cos_d_dot_G).sum(axis=1)[inverse]

    return Qij

def charge_dipole_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):
    """
    Charge-dipole matrix.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    nu, nv, nw : int
        Supercell size.
    n_atm : int
        Number of unit cell atoms.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    B : 2d array, 3x3
        Reciprocal-space crystal axis to Cartesian transformation matrix.
    R : 2d array, 3x3
        Rotation matrix between real and reciprocal-space Cartesian axes.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-3``.

    Returns
    -------
    Qijk : 2d array
        Charge-dipole matrix. Array of shape ``n*(n+1)//2`` x3.

    """

    n = nu*nv*nw*n_atm

    Qijk = np.zeros((n*(n+1)//2,3))

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    k = j+n*i-(i+1)*i//2

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    a = __A(alpha, d)

    Qijk[k,0] = (a*dx)[inverse]
    Qijk[k,1] = (a*dy)[inverse]
    Qijk[k,2] = (a*dz)[inverse]

    sin_d_dot_G = np.sin(np.kron(dx,Gx)+
                         np.kron(dy,Gy)+
                         np.kron(dz,Gz))

    sin_d_dot_G = sin_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-np.pi**2*G_sq/alpha**2)/G_sq

    g = factors*sin_d_dot_G

    Qijk[k,0] += np.sum(g*Gx, axis=1)[inverse]
    Qijk[k,1] += np.sum(g*Gy, axis=1)[inverse]
    Qijk[k,2] += np.sum(g*Gz, axis=1)[inverse]

    return Qijk

def dipole_dipole_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):
    """
    Dipole-dipole matrix.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Atomic positions.
    nu, nv, nw : int
        Supercell size.
    n_atm : int
        Number of unit cell atoms.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    B : 2d array, 3x3
        Reciprocal-space crystal axis to Cartesian transformation matrix.
    R : 2d array, 3x3
        Rotation matrix between real and reciprocal-space Cartesian axes.
    tol : float, optional
        Tolerance of distances for unique pairs. Default is ``1e-3``.

    Returns
    -------
    Qijkl : 2d array,
        Dipole-dipole matrix. Array of shape ``n*(n+1)//2`` x6.

    """

    n = nu*nv*nw*n_atm

    Qijkl = np.zeros((n*(n+1)//2,6))

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    k = j+n*i-(i+1)*i//2

    l = np.arange(n)
    l = l+n*l-(l+1)*l//2

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    b, c = __B(alpha, d), __C(alpha, d)

    Qijkl[k,0] = (b-dx*dx*c)[inverse]
    Qijkl[k,1] = (b-dy*dy*c)[inverse]
    Qijkl[k,2] = (b-dz*dz*c)[inverse]

    Qijkl[k,3] = (-dy*dz*c)[inverse]
    Qijkl[k,4] = (-dx*dz*c)[inverse]
    Qijkl[k,5] = (-dx*dy*c)[inverse]

    Qijkl[l,0] = Qijkl[l,1] = Qijkl[l,2] = -4*alpha**3/(3*np.sqrt(np.pi))

    cos_d_dot_G = np.cos(np.kron(dx,Gx)+
                         np.kron(dy,Gy)+
                         np.kron(dz,Gz))

    cos_d_dot_G = cos_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-np.pi**2*G_sq/alpha**2)/G_sq

    g = factors*cos_d_dot_G

    Gxx, Gyy, Gzz = Gx*Gx, Gy*Gy, Gz*Gz
    Gxz, Gyz, Gxy = Gx*Gz, Gy*Gz, Gx*Gy

    Qijkl[k,0] += np.sum(g*Gxx, axis=1)[inverse]
    Qijkl[k,1] += np.sum(g*Gyy, axis=1)[inverse]
    Qijkl[k,2] += np.sum(g*Gzz, axis=1)[inverse]

    Qijkl[k,3] += np.sum(g*Gyz, axis=1)[inverse]
    Qijkl[k,4] += np.sum(g*Gxz, axis=1)[inverse]
    Qijkl[k,5] += np.sum(g*Gxy, axis=1)[inverse]

    return Qijkl

def pairs(u, v, w, atm, A, extend=False):
    """
    Generate pairs.

    Parameters
    ----------
    u, v, w : 1d array
        Fractional coordinates.
    atm : 1d array, str
        Atoms, ions, or isotopes.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    extend : bool, optional
        Extend beyond one unit cell. The default is ``False``.

    Returns
    -------
    pair_info : dict
        Pair information for constructing bonds.

    """

    n_atm = atm.shape[0]

    i, j = np.triu_indices(n_atm, k=1)
    i, j = np.concatenate((i,j)), np.concatenate((j,i))

    du = u[j]-u[i]
    dv = v[j]-v[i]
    dw = w[j]-w[i]

    u_img = -1*(du > 0.5)+(du < -0.5)
    v_img = -1*(dv > 0.5)+(dv < -0.5)
    w_img = -1*(dw > 0.5)+(dw < -0.5)

    du[du < -0.5] += 1
    dv[dv < -0.5] += 1
    dw[dw < -0.5] += 1

    du[du > 0.5] -= 1
    dv[dv > 0.5] -= 1
    dw[dw > 0.5] -= 1

    if extend:

        U, V, W = np.meshgrid(np.arange(-1,2),
                              np.arange(-1,2),
                              np.arange(-1,2), indexing='ij')

        U = U.flatten()[:,np.newaxis]
        V = V.flatten()[:,np.newaxis]
        W = W.flatten()[:,np.newaxis]

        u_img = (u_img+U).flatten()
        v_img = (v_img+V).flatten()
        w_img = (w_img+W).flatten()

        du, dv, dw = (du+U).flatten(), (dv+V).flatten(), (dw+W).flatten()

        U = np.repeat(np.delete(U.flatten(), 13), n_atm)
        V = np.repeat(np.delete(V.flatten(), 13), n_atm)
        W = np.repeat(np.delete(W.flatten(), 13), n_atm)

        du = np.concatenate((du,U))
        dv = np.concatenate((dv,V))
        dw = np.concatenate((dw,W))

        u_img = np.concatenate((u_img,U))
        v_img = np.concatenate((v_img,V))
        w_img = np.concatenate((w_img,W))

        i = np.tile(i, 27)
        j = np.tile(j, 27)

        i = np.concatenate((i,np.tile(np.arange(n_atm), 26)))
        j = np.concatenate((j,np.tile(np.arange(n_atm), 26)))

    dx, dy, dz = crystal.transform(du, dv, dw, A)

    d = np.sqrt(dx**2+dy**2+dz**2)

    atms = np.stack((atm[i],atm[j]))

    atm_pairs = np.array(['_'.join((a,b)) for a, b in zip(*atms)])
    atms, atm_labels = np.unique(atm_pairs, return_inverse=True)

    metric = np.stack((dx,dy,dz,d)).T

    pair_info = { }

    for k in range(n_atm):

        mask = i == k

        sort = np.lexsort(metric[mask].T)

        label_dict = { }

        l = j[mask][sort]
        lab = atm_labels[mask][sort]
        ref = [atm_pair.split('_')[1] for atm_pair in atm_pairs[mask][sort]]

        iu, iv, iw = u_img[mask][sort], v_img[mask][sort], w_img[mask][sort]

        c = 0
        m, c_uvw = [], []
        m.append(l[0])
        c_uvw.append((iu[0],iv[0],iw[0]))
        ind = -1
        for ind in range(lab.shape[0]-1):
            if lab[ind] != lab[ind+1]:
                key = c, ref[ind]
                label_dict[key] = m, c_uvw
                m, c_uvw = [], []
                c += 1
            m.append(l[ind+1])
            c_uvw.append((iu[ind+1],iv[ind+1],iw[ind+1]))
        key = c, ref[ind+1]
        label_dict[key] = m, c_uvw

        pair_info[k] = label_dict

    return pair_info

def bonds(pair_info, u, v, w, A, tol=1e-3):
    """
    Generate bond pairs.

    Parameters
    ----------
    pair_info : dict
        Pair information for constructing bonds.
    u, v, w : 1d array
        Fractional coordinates.
    atm : 1d array, str
        Atoms, ions, or isotopes.
    A : 2d array, 3x3
        Real space crystal axis to Cartesian transformation matrix.
    tol : float, optional
        Tolerance of distances for unique pairs. The default is ``1e-3``.

    Returns
    -------
    dx, dy, dz : 2d array
        Separation distance vector.
    img_i, img_j, img_k : 2d array, int
        Indices of cell-pairs.
    atm_ind : 2d array, int
        Indices of atom-pairs.
    pair_inv : 2d array, int
        Indices of inverse-pairs.
    pair_ind : 2d array, int
        Indices of bond-pairs.
    pair_trans : 2d array, int
        Indices of transpose-pairs.

    """

    n_atm = u.shape[0]

    indices = [[[a,*b] for sub in value.values()
                for a, b in zip(*sub)] for value in pair_info.values()]

    atm_ind, img_i, img_j, img_k = np.array(indices).T

    atm_ind, img_i, img_j, img_k = atm_ind.T, img_i.T, img_j.T, img_k.T

    du, dv, dw = (u[atm_ind].T-u).T, (v[atm_ind].T-v).T, (w[atm_ind].T-w).T

    du += img_i
    dv += img_j
    dw += img_k

    dx, dy, dz = crystal.transform(du, dv, dw, A)

    dx = dx.reshape(atm_ind.shape)
    dy = dy.reshape(atm_ind.shape)
    dz = dz.reshape(atm_ind.shape)

    d = np.sqrt(dx**2+dy**2+dz**2)

    n_pair = atm_ind.shape[1]

    dist = np.round(d/tol).astype(int)

    _, inv_ind = np.unique(dist, return_inverse=True)

    pair_ind = np.arange(n_pair+1)[inv_ind].reshape(n_atm,n_pair)

    d_xyz = np.round(np.stack((dx,dy,dz))/tol).astype(int)

    inv_d_xyz = -d_xyz

    p = np.repeat(np.arange(n_pair), n_atm*n_pair).reshape(n_pair,n_pair,n_atm)

    mask = (d_xyz.T == inv_d_xyz[:,atm_ind,:].T).all(axis=3)

    pair_inv = p.T[mask.T].reshape(n_atm,n_pair)

    pair_trans = np.zeros((n_atm,n_pair), dtype=np.int32)

    indices = atm_ind, pair_inv, pair_ind, pair_trans

    return dx, dy, dz, img_i, img_j, img_k, *indices

def anisotropy(dx, dy, dz):
    """
    Easy axes components.

    Parameters
    ----------
    dx, dy, dz : 2d array
        Separation distance vector.

    Returns
    -------
    uxx, uyy, uzz, uyz, uxz, uxy : 2d array
        Components of easy axes.

    """

    d = np.sqrt(dx**2+dy**2+dz**2)

    uxx = (dx*dx).mean(axis=1)/d.mean(axis=1)**2
    uyy = (dy*dy).mean(axis=1)/d.mean(axis=1)**2
    uzz = (dz*dz).mean(axis=1)/d.mean(axis=1)**2
    uyz = (dy*dz).mean(axis=1)/d.mean(axis=1)**2
    uxz = (dx*dz).mean(axis=1)/d.mean(axis=1)**2
    uxy = (dx*dy).mean(axis=1)/d.mean(axis=1)**2

    return uxx, uyy, uzz, uyz, uxz, uxy
