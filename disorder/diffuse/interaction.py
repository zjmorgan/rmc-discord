#!/usr/bin/env python

import numpy as np

from scipy.special import erfc

from disorder.material import crystal

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

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    # ---

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    du, dv, dw = ju-iu, jv-iv, jw-iw

    distance = np.stack((du,dv,dw))

    distance = np.stack((distance.T,-distance.T)).T

    sort = np.lexsort(distance, axis=1)[:,0]

    n_pairs = sort.size

    distance = distance.reshape(3,2*n_pairs)[:,sort+2*np.arange(n_pairs)]

    metric = np.vstack(distance).T

    u, index, inverse = np.unique(metric, return_index=True,
                                  return_inverse=True, axis=0)

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

    distance = np.stack((distance.T,-distance.T)).T

    sort = np.lexsort(distance, axis=1)[:,0]

    n_pairs = sort.size

    distance = distance.reshape(3,2*n_pairs)[:,sort+2*np.arange(n_pairs)]

    metric = np.vstack(np.round(distance/tol,0)).astype(int).T

    _, ind, inv = np.unique(metric, return_index=True,
                            return_inverse=True, axis=0)

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

    mu = (nu+1)//2
    mv = (nv+1)//2
    mw = (nw+1)//2

    ku = 2*np.pi*np.arange(mu)
    kv = 2*np.pi*np.concatenate((np.arange(mv),np.arange(-mv+1,0)))
    kw = 2*np.pi*np.concatenate((np.arange(mw),np.arange(-mw+1,0)))

    ku, kv, kw = np.meshgrid(ku, kv, kw, indexing='ij')

    ku, kv, kw = ku.flatten(), kv.flatten(), kw.flatten()
    ku, kv, kw = np.delete(ku, 0), np.delete(kv, 0), np.delete(kw, 0)

    Gx, Gy, Gz = crystal.transform(ku, kv, kw, B)
    Gx, Gy, Gz = crystal.transform(Gx, Gy, Gz, R)

    return Gx, Gy, Gz

def charge_charge_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):

    n = nu*nv*nw*n_atm

    Qij = np.zeros((n,n))

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    Qij[i,j] = (erfc(alpha*d)/d)[inverse]

    Qij[np.arange(n),np.arange(n)] = -2*alpha/np.sqrt(np.pi)

    cos_d_dot_G = np.cos(np.kron(dx,Gx)+np.kron(dy,Gy)+np.kron(dz,Gz))
    cos_d_dot_G = cos_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-G_sq/(4*alpha**2))/G_sq

    Qij[i,j] += (factors*cos_d_dot_G).sum(axis=1)[inverse]

    Qij[j,i] = Qij[i,j]

    return Qij

def charge_dipole_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):

    n = nu*nv*nw*n_atm

    Qijk = np.zeros((n,n,3))

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    a = __A(alpha, d)

    Qijk[i,j,0] = (a*dx)[inverse]
    Qijk[i,j,1] = (a*dy)[inverse]
    Qijk[i,j,2] = (a*dz)[inverse]

    sin_d_dot_G = np.sin(np.kron(dx,Gx)+np.kron(dy,Gy)+np.kron(dz,Gz))
    sin_d_dot_G = sin_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-G_sq/(4*alpha**2))/G_sq

    g = factors*sin_d_dot_G

    Qijk[i,j,0] += np.sum(g*Gx, axis=1)[inverse]
    Qijk[i,j,1] += np.sum(g*Gy, axis=1)[inverse]
    Qijk[i,j,2] += np.sum(g*Gz, axis=1)[inverse]

    Qijk[j,i,0] = Qijk[i,j,0]
    Qijk[j,i,1] = Qijk[i,j,1]
    Qijk[j,i,2] = Qijk[i,j,2]

    return Qijk

def dipole_dipole_matrix(rx, ry, rz, nu, nv, nw, n_atm, A, B, R, tol=1e-3):

    n = nu*nv*nw*n_atm

    Qijkl = np.zeros((n,n,3,3))

    dx, dy, dz, i, j, inverse = atom_pairs_distance(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, tol=tol)

    d = np.sqrt(dx**2+dy**2+dz**2)

    Gx, Gy, Gz = spatial_wavevector(nu, nv, nw, n_atm, B, R)

    G_sq = Gx**2+Gy**2+Gz**2

    u, v, w = np.dot(A, [nu,0,0]), np.dot(A, [0,nv,0]), np.dot(A, [0,0,nw])

    V = np.dot(u, np.cross(v, w))

    alpha = np.sqrt(2*np.pi*np.min([nu/np.linalg.norm(u)**2,
                                    nv/np.linalg.norm(v)**2,
                                    nw/np.linalg.norm(w)**2]))

    b, c = __B(alpha, d), __C(alpha, d)

    Qijkl[i,j,0,0] = (b-dx*dx*c)[inverse]
    Qijkl[i,j,1,1] = (b-dy*dy*c)[inverse]
    Qijkl[i,j,2,2] = (b-dz*dz*c)[inverse]

    Qijkl[i,j,0,2] = Qijkl[i,j,2,0] = (-dx*dz*c)[inverse]
    Qijkl[i,j,1,2] = Qijkl[i,j,2,1] = (-dy*dz*c)[inverse]
    Qijkl[i,j,0,1] = Qijkl[i,j,1,0] = (-dx*dy*c)[inverse]

    Qijkl[np.arange(n),np.arange(n),0,0] = -4*alpha**3/(3*np.sqrt(np.pi))
    Qijkl[np.arange(n),np.arange(n),1,1] = -4*alpha**3/(3*np.sqrt(np.pi))
    Qijkl[np.arange(n),np.arange(n),2,2] = -4*alpha**3/(3*np.sqrt(np.pi))

    cos_d_dot_G = np.cos(np.kron(dx,Gx)+np.kron(dy,Gy)+np.kron(dz,Gz))
    cos_d_dot_G = cos_d_dot_G.reshape(d.size, G_sq.size)

    factors = 4*np.pi/V*np.exp(-G_sq/(4*alpha**2))/G_sq

    g = factors*cos_d_dot_G

    Qijkl[i,j,0,0] += np.sum(g*Gx*Gx, axis=1)[inverse]
    Qijkl[i,j,1,1] += np.sum(g*Gy*Gy, axis=1)[inverse]
    Qijkl[i,j,2,2] += np.sum(g*Gz*Gz, axis=1)[inverse]

    fxz = np.sum(g*Gx*Gz, axis=1)[inverse]
    fyz = np.sum(g*Gy*Gz, axis=1)[inverse]
    fxy = np.sum(g*Gx*Gy, axis=1)[inverse]

    Qijkl[i,j,0,2] += fxz
    Qijkl[i,j,1,2] += fyz
    Qijkl[i,j,0,1] += fxy

    Qijkl[i,j,2,0] += fxz
    Qijkl[i,j,2,1] += fyz
    Qijkl[i,j,1,0] += fxy

    Qijkl[j,i,0,0] = Qijkl[i,j,0,0]
    Qijkl[j,i,0,1] = Qijkl[i,j,0,1]
    Qijkl[j,i,0,2] = Qijkl[i,j,0,2]

    Qijkl[j,i,1,0] = Qijkl[i,j,1,0]
    Qijkl[j,i,1,1] = Qijkl[i,j,1,1]
    Qijkl[j,i,1,2] = Qijkl[i,j,1,2]

    Qijkl[j,i,2,0] = Qijkl[i,j,2,0]
    Qijkl[j,i,2,1] = Qijkl[i,j,2,1]
    Qijkl[j,i,2,2] = Qijkl[i,j,2,2]

    return Qijkl