#!/usr/bin/env python

import numpy as np

from disorder.material import crystal
from disorder.material import symmetry

def reciprocal(h_range, k_range, l_range, mask, B, T=np.eye(3)):

    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]

    h_, k_, l_  = np.meshgrid(np.linspace(h_range[0],h_range[1],nh),
                              np.linspace(k_range[0],k_range[1],nk),
                              np.linspace(l_range[0],l_range[1],nl),
                              indexing='ij')

    h, k, l = crystal.transform(h_, k_, l_, T)

    Qh, Qk, Ql = crystal.vector(h, k, l, B)

    return Qh[~mask], Qk[~mask], Ql[~mask]

def cell(nu, nv, nw, A):

    i, j, k = np.meshgrid(np.arange(nu),
                          np.arange(nv),
                          np.arange(nw), indexing='ij')

    ix, iy, iz = crystal.transform(i, j, k, A)

    return ix.flatten(), iy.flatten(), iz.flatten()

def real(ux, uy, uz, ix, iy, iz, atm):

    rx = (ix[:,np.newaxis]+ux).flatten()
    ry = (iy[:,np.newaxis]+uy).flatten()
    rz = (iz[:,np.newaxis]+uz).flatten()

    ion = np.tile(atm, ix.shape[0])

    return rx, ry, rz, ion

def factor(nu, nv, nw):

    ku = 2*np.pi*np.fft.fftfreq(nu)
    kv = 2*np.pi*np.fft.fftfreq(nv)
    kw = 2*np.pi*np.fft.fftfreq(nw)

    ru = np.arange(nu)
    rv = np.arange(nv)
    rw = np.arange(nw)

    k_dot_r = np.kron(ku,ru)[:,np.newaxis,np.newaxis]+\
              np.kron(kv,rv)[:,np.newaxis]+\
              np.kron(kw,rw)

    pf = np.exp(1j*k_dot_r)

    return pf.flatten()

def unit(vx, vy, vz):

    v = np.sqrt(vx**2+vy**2+vz**2)

    mask = np.isclose(v, 0, rtol=1e-4)

    if (np.sum(mask) > 0):
        n = np.argwhere(mask)
        v[n] = 1

        vx, vy, vz = vx/v, vy/v, vz/v
        vx[n], vy[n], vz[n] = 0, 0, 0

        v[n] = 0
    else:
        vx, vy, vz = vx/v, vy/v, vz/v

    return vx, vy, vz, v

def indices(mask):

    i_mask = np.arange(mask.size)[mask.flatten()]

    i_unmask = np.arange(mask.size)[~mask.flatten()]

    return i_mask, i_unmask

def prefactors(scattering_length, phase_factor, occupancy, primitive=None):

    n_atm = occupancy.shape[0]

    n_hkl = scattering_length.shape[0] // n_atm

    scattering_length = scattering_length.reshape(n_hkl,n_atm)
    phase_factor = phase_factor.reshape(n_hkl,n_atm)

    factors = scattering_length*phase_factor*occupancy

    if (not primitive is None):
        return np.sum(factors[:,primitive],axis=2).flatten()
    else:
        return factors.flatten()

def transform(delta_r, H, K, L, nu, nv, nw, n_atm):

    delta_r = delta_r.reshape(nu,nv,nw,n_atm)

    delta_k = np.fft.ifftn(delta_r, axes=(0,1,2))*nu*nv*nw

    Ku = np.mod(H, nu).astype(int)
    Kv = np.mod(K, nv).astype(int)
    Kw = np.mod(L, nw).astype(int)

    i_dft = Kw+nw*(Kv+nv*Ku)

    return delta_k.flatten(), i_dft

def intensity(delta_k, i_dft, factors):

    n_hkl = i_dft.shape[0]

    n_atm = factors.shape[0] // n_hkl

    factors = factors.reshape(n_hkl,n_atm)

    n_uvw = delta_k.shape[0] // n_atm

    delta_k = delta_k.reshape(n_uvw,n_atm)

    prod = factors*delta_k[i_dft,:]

    F = np.sum(prod, axis=1)

    I = np.real(F)**2+np.imag(F)**2

    return I/(n_uvw*n_atm)

def structure(delta_k, i_dft, factors):

    n_hkl = i_dft.shape[0]

    n_atm = factors.shape[0] // n_hkl

    factors = factors.reshape(n_hkl,n_atm)

    n_uvw = delta_k.shape[0] // n_atm

    delta_k = delta_k.reshape(n_uvw,n_atm)

    prod = factors*delta_k[i_dft,:]

    F = np.sum(prod, axis=1)

    return F, prod.flatten()

def bragg(Qx, Qy, Qz, rx, ry, rz, factors, cond):

    n_hkl = cond.sum()
    n_xyz = factors.size // Qx.shape[0]

    factors = factors.reshape(factors.size // n_xyz,n_xyz)[cond,:]

    phase_factor = np.exp(1j*(np.kron(Qx[cond],rx)\
                             +np.kron(Qy[cond],ry)\
                             +np.kron(Qz[cond],rz)))

    phase_factor = phase_factor.reshape(n_hkl,n_xyz)

    return (factors*phase_factor).sum(axis=1)

def debye_waller(h_range, k_range, l_range, nh, nk, nl,
                 U11, U22, U33, U23, U13, U12,
                 a_, b_, c_, T=np.eye(3)):

    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh),
                             np.linspace(k_range[0],k_range[1],nk),
                             np.linspace(l_range[0],l_range[1],nl),
                             indexing='ij')

    h, k, l = crystal.transform(h_, k_, l_, T)

    h = h.flatten()
    k = k.flatten()
    l = l.flatten()

    n_hkl = nh*nk*nl
    n_atm = U11.shape[0]

    T = np.zeros((n_hkl,n_atm))

    for i in range(n_atm):

        T[:,i] = np.exp(-2*np.pi**2*(U11[i]*(h*a_)**2+
                                     U22[i]*(k*b_)**2+
                                     U33[i]*(l*c_)**2+
                                     U23[i]*k*l*b_*c_*2+
                                     U13[i]*h*l*a_*c_*2+
                                     U12[i]*h*k*a_*b_*2))

    return T.flatten()

def condition(H, K, L, nu=1, nv=1, nw=1, centering=None):

    iH = np.mod(H, nu)
    iK = np.mod(K, nv)
    iL = np.mod(L, nw)

    h = H // nu
    k = K // nv
    l = L // nw

    dft_cond = (iH == 0) & (iK == 0) & (iL == 0)

    if (centering is None):
        cond = dft_cond
    elif (centering == 'P'):
        cond = (h % 1 == 0) & (k % 1 == 0) & (l % 1 == 0) & (dft_cond)
    elif (centering == 'I'):
        cond = ((h+k+l) % 2 == 0) & (dft_cond)
    elif (centering == 'F'):
        cond = ((h+k) % 2 == 0) \
             & ((k+l) % 2 == 0) \
             & ((l+h) % 2 == 0) & (dft_cond)
    elif (centering == 'R(obv)'):
        cond = ((-h+k+l) % 3 == 0) & (dft_cond)
    elif (centering == 'R(rev)'):
        cond = ((h-k+l) % 3 == 0) & (dft_cond)
    elif (centering == 'C'):
        cond = ((h+k) % 2 == 0) & (dft_cond)
    elif (centering == 'A'):
        cond = ((k+l) % 2 == 0) & (dft_cond)
    elif (centering == 'B'):
        cond = ((l+h) % 2 == 0) & (dft_cond)
    elif (centering == 'H'):
        cond = ((h-k) % 3 == 0) & (dft_cond)
    elif (centering == 'D'):
        cond = ((h+k+l) % 3 == 0) & (dft_cond)

    return H[cond], K[cond], L[cond], cond

def mapping(h_range, k_range, l_range, nh, nk, nl,
            nu, nv, nw, T=np.eye(3), laue=None):

    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh),
                             np.linspace(k_range[0],k_range[1],nk),
                             np.linspace(l_range[0],l_range[1],nl),
                             indexing='ij')

    h_ = h_.flatten()
    k_ = k_.flatten()
    l_ = l_.flatten()

    h, k, l = crystal.transform(h_, k_, l_, T)

    H = np.round(h*nu).astype(int)
    K = np.round(k*nv).astype(int)
    L = np.round(l*nw).astype(int)

    iH = np.mod(H, nu)
    iK = np.mod(K, nv)
    iL = np.mod(L, nw)

    mask = (iH == 0) & (~np.isclose(np.mod(h*nu,nu),0))
    H[mask] += 1

    mask = (iK == 0) & (~np.isclose(np.mod(k*nv,nv),0))
    K[mask] += 1

    mask = (iL == 0) & (~np.isclose(np.mod(l*nw,nw),0))
    L[mask] += 1

    if (laue == None or laue == 'None'):

        index = np.arange(nh*nk*nl)

        return h, k, l, H, K, L, index, index, np.array([u'x,y,z'])

    symops = symmetry.inverse(symmetry.laue(laue))

    total = []

    coordinate = np.stack((H,K,L))

    cosymmetries, coindices, coinverses = np.unique(coordinate,
                                                    axis=1,
                                                    return_index=True,
                                                    return_inverse=True)

    for op in symops:

        transformed = symmetry.evaluate([op], cosymmetries, translate=False)
        total.append(transformed)

    index = np.arange(coordinate.shape[1])

    total = np.vstack(total)

    for i in range(cosymmetries.shape[1]):

        total[:,:,i] = total[np.lexsort(total[:,:,i].T),:,i]

    total = np.vstack(total)

    _, indices, inverses = np.unique(total,
                                     axis=1,
                                     return_index=True,
                                     return_inverse=True)

    reverses = np.arange(indices.shape[0])

    h = h[coindices][indices]
    k = k[coindices][indices]
    l = l[coindices][indices]

    H = H[coindices][indices]
    K = K[coindices][indices]
    L = L[coindices][indices]

    index = index[coindices][indices]
    reverses = reverses[inverses][coinverses]

    return h, k, l, H, K, L, index, reverses, symops

def reduced(h_range, k_range, l_range, nh, nk, nl,
            nu, nv, nw, T=np.eye(3), laue=None):

    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh),
                             np.linspace(k_range[0],k_range[1],nk),
                             np.linspace(l_range[0],l_range[1],nl),
                             indexing='ij')

    h, k, l = crystal.transform(h_, k_, l_, T)

    h = h.flatten()
    k = k.flatten()
    l = l.flatten()

    del h_, k_, l_

    if (nh > 1):
        h_max_res = (h_range[1]-h_range[0])/(nh-1)
    else:
        h_max_res = 0

    if (nk > 1):
        k_max_res = (k_range[1]-k_range[0])/(nk-1)
    else:
        k_max_res = 0

    if (nl > 1):
        l_max_res = (l_range[1]-l_range[0])/(nl-1)
    else:
        l_max_res = 0

    hkl_max_res = np.array([[h_max_res,0,0],[0,k_max_res,0],[0,0,l_max_res]])
    hkl_res = np.abs(np.dot(T, hkl_max_res))

    h_res, k_res, l_res = np.max(hkl_res, axis=0)

    if (h_res > 0 and h_res < 1/nu):
        Nu = int(1/h_res // nu)*nu
    else:
        Nu = nu

    if (k_res > 0 and k_res < 1/nv):
        Nv = int(1/k_res // nv)*nv
    else:
        Nv = nv

    if (l_res > 0 and l_res < 1/nw):
        Nw = int(1/l_res // nw)*nw
    else:
        Nw = nw

    H = np.round(h*Nu).astype(np.int16)

    iH = np.mod(H, Nu)
    del iH, h

    K = np.round(k*Nv).astype(np.int16)

    iK = np.mod(K, Nv)
    del iK, k

    L = np.round(l*Nw).astype(np.int16)

    iL = np.mod(L, Nw)
    del iL, l

    if (laue == None or laue == 'None'):

        index = np.arange(nh*nk*nl)

        return index, index, np.array([u'x,y,z']), Nu, Nv, Nw

    symops = np.array(symmetry.laue(laue))

    symops = symmetry.inverse(symops)

    coordinate = np.stack((H,K,L)).T
    n = coordinate.shape[0]
    
    del H, K, L
    
    coordinate = np.stack((coordinate,-coordinate)).T

    sort = np.lexsort(coordinate, axis=1)[:,0]
    pair = coordinate.reshape(3,2*n)[:,sort+2*np.arange(n)].T
    index = np.arange(n)
        
    del coordinate, sort

    cosymmetries, coindices, coinverses = symmetry.unique(pair)
    h_, k_, l_ = cosymmetries.T
    n = cosymmetries.shape[0]
 
    del cosymmetries
            
    sym, n_symops = symmetry.laue_id(symops)

    ops = np.zeros((3,n,n_symops), dtype=np.int16)

    for i in range(n_symops):

        h, k, l = symmetry.bragg(h_, k_, l_, sym, i)

        ops[0,:,i], ops[1,:,i], ops[2,:,i] = h, k, l
        
    sort = np.lexsort(ops, axis=1)[:,0]
    total = ops.reshape(3,n_symops*n)[:,sort+n_symops*np.arange(n)].T
    
    del ops, h, k, l

    _, indices, inverses = symmetry.unique(total)

    reverses = np.arange(indices.shape[0])
    
    index = index[coindices][indices]
    reverses = reverses[inverses][coinverses]

    return index, reverses, symops, Nu, Nv, Nw