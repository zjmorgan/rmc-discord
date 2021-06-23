#!/usr/bin/env python

import numpy as np

from scipy import spatial

from disorder.diffuse.scattering import gauss, blur
from disorder.diffuse.powder import average

def data(folder='ising/', directory='', filestr=None):
    
    if (filestr == None):
        I = np.load(directory+folder+'intensity.npy')
        sigma_sq = np.load(directory+folder+'error.npy')
        
    else:
        I = np.load(directory+folder+'intensity.'+filestr+'.npy')
        sigma_sq = np.load(directory+folder+'error.'+filestr+'.npy')       
    
    inv_sigma_sq = 1/sigma_sq
    
    mask = np.isnan(I)\
         + np.isinf(I)\
         + np.isnan(inv_sigma_sq)\
         + np.isinf(inv_sigma_sq)
        
    if (len(I.shape) == 2):
        nh, nk, nl = I.shape[0], I.shape[1], 1
        mask = mask.reshape(nh, nk, nl)
        I = I.reshape(nh, nk, nl)
        inv_sigma_sq = inv_sigma_sq.reshape(nh, nk, nl)
    else:
        nh, nk, nl = I.shape[0], I.shape[1], I.shape[2] 
            
    I = I[~mask]
    inv_sigma_sq = inv_sigma_sq[~mask]
    
    return I, inv_sigma_sq, mask, nh, nk, nl

def reciprocal(h_range, k_range, l_range, mask, B, T=np.eye(3)):
    
    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]
    
    h_, k_, l_  = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                              np.linspace(k_range[0],k_range[1],nk), 
                              np.linspace(l_range[0],l_range[1],nl), 
                              indexing='ij')
    
    h = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
    k = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
    l = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
    
    Qh, Qk, Ql = nuclear(h, k, l, B)
    
    return Qh[~mask], Qk[~mask], Ql[~mask]

def nuclear(h, k, l, B):    
        
    Qh = 2*np.pi*(B[0,0]*h+B[0,1]*k+B[0,2]*l)
    Qk = 2*np.pi*(B[1,0]*h+B[1,1]*k+B[1,2]*l)
    Ql = 2*np.pi*(B[2,0]*h+B[2,1]*k+B[2,2]*l)
    
    return Qh, Qk, Ql

def cell(nu, nv, nw, A):

    i, j, k = np.meshgrid(np.arange(nu), 
                          np.arange(nv), 
                          np.arange(nw), indexing='ij')
    
    ix = A[0,0]*i+A[0,1]*j+A[0,2]*k
    iy = A[1,0]*i+A[1,1]*j+A[1,2]*k
    iz = A[2,0]*i+A[2,1]*j+A[2,2]*k
    
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

def boxblur(sigma, n):
    
    sigma = np.asarray(sigma)
        
    l = np.floor(np.sqrt(12*sigma**2/n+1))
    
    if (np.size(sigma) == 1): 
        if (l % 2 == 0):
            l -= 1
            
    else:
        l[np.mod(l,2) == 0] -= 1
        
    u = l+2
    
    m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
    
    if (np.size(sigma) == 1): 
        
        sizes = np.zeros(n, dtype=int)
        
        for i in range(n):
       
            if (i < m):
                sizes[i] = l
            else:
                sizes[i] = u
    
    else:
        
        sizes = np.zeros((n, np.size(sigma)), dtype=int)
        
        for i in range(n):
            for j in range(np.size(sigma)):
            
                if (i < m[j]):
                    sizes[i,j] = l[j]
                else:
                    sizes[i,j] = u[j]
    
    return ((np.array(sizes)-1)/2).astype(int).flatten()
    
def gaussian(mask, sigma):
        
    v = np.ones(mask.shape)
    v[mask] = 0
    
    v = v.flatten()
    
    a = np.zeros(mask.size)
    b = np.zeros(mask.size)
    c = np.zeros(mask.size)
    d = np.zeros(mask.size)
    e = np.zeros(mask.size)
    f = np.zeros(mask.size)
    g = np.zeros(mask.size)
    h = np.zeros(mask.size)
    
    w = np.zeros(mask.size)
    
    boxes = boxblur(sigma, 3)
    
    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]
        
    if (np.isclose(sigma, 0).all()):
    
        return v
    
    else:
        
        gauss(w, v, boxes, a, b, c, d, e, f, g, h, nh, nk, nl) 
        
        veil = np.isclose(w,0)
        
        if (np.sum(veil) > 0):
            
            x = np.zeros(mask.size)
            
            gauss(x, v, boxes, a, b, c, d, e, f, g, h, nh, nk, nl) 
            
            w[veil] = x[veil].copy()
                    
        veil = np.isclose(w,0)
        
        w_inv = np.zeros(mask.size)
        w_inv[~veil] = 1/w[~veil]
               
        return w_inv
        
def boxfilter(I, mask, sigma, v_inv):
    
    a = np.zeros(mask.size)
    b = np.zeros(mask.size)
    c = np.zeros(mask.size)
    d = np.zeros(mask.size)
    e = np.zeros(mask.size)
    f = np.zeros(mask.size)
    g = np.zeros(mask.size)
    h = np.zeros(mask.size)
    
    i = np.zeros(mask.size)    
    
    boxes = boxblur(sigma, 3)
    
    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]
    
    if (np.isclose(sigma, 0).all()):
    
        return  I
    
    else:
        
        u = I.copy()
        u[mask] = 0
        
        gauss(i, u.flatten(), boxes, a, b, c, d, e, f, g, h, nh, nk, nl) 
        
        return (v_inv*i).reshape(nh,nk,nl)
    
def blurring(I, sigma):
    
    i = np.zeros(I.size)    
        
    boxes = boxblur(sigma, 3)
    
    nh, nk, nl = I.shape[0], I.shape[1], I.shape[2]
    
    if (np.isclose(sigma, 0).all()):
    
        return I
    
    else:
                
        blur(i, I.flatten(), boxes, nh, nk, nl) 
        
        return i.reshape(nh,nk,nl)
    
def indices(inverses, mask):
        
    i_mask = np.arange(mask.size)[mask.flatten()]
    
    i_unmask = np.arange(mask.size)[~mask.flatten()]
    
    return i_mask, i_unmask

def powder(Q, rx, ry, rz, scattering_length, fract=0.5):
    
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    coordinate = np.array(list(pairs))
            
    i = coordinate[:,0]
    j = coordinate[:,1]
        
    n_hkl = Q.shape[0]
    n_xyz = rx.shape[0]
    
    n_pairs = i.shape[0]
    n_atm = scattering_length.shape[0] // n_hkl
    
    k = np.mod(i,n_atm)
    l = np.mod(j,n_atm)
    
    #n_uvw = n_xyz // n_atm

    rx_ij = rx[j]-rx[i]
    ry_ij = ry[j]-ry[i]
    rz_ij = rz[j]-rz[i]
    
    r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)
    
    m = np.arange(n_xyz)
    n = np.mod(m,n_atm)
        
    summation = np.zeros(Q.shape[0])
    
    auto = np.zeros(Q.shape[0])

    average(summation, 
            auto,
            Q,
            r_ij,
            scattering_length,
            k,
            l,
            n,
            n_xyz,
            n_atm)
    
    scale = n_xyz/((np.sqrt(8*n_pairs+1)+1)/2)
        
    I = (auto/scale+2*summation)/n_xyz
   
    return I

def prefactors(scattering_length, phase_factor, occupancy, primitive=None):
    
    n_atm = occupancy.shape[0]
    
    n_peaks = scattering_length.shape[0] // n_atm
    
    scattering_length = scattering_length.reshape(n_peaks,n_atm)
    phase_factor = phase_factor.reshape(n_peaks,n_atm)
    
    factors = scattering_length*phase_factor*occupancy
    
    if (not primitive is None):
        
        return np.sum(factors[:,primitive],axis=2).flatten()
        
    else:
        
        return factors.flatten()

def transform(A, 
              H,
              K,
              L,
              nu, 
              nv, 
              nw, 
              n_atm):
    
    A_k = np.fft.ifftn(A.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*nu*nv*nw

    Kx = np.mod(H, nu).astype(int)
    Ky = np.mod(K, nv).astype(int)
    Kz = np.mod(L, nw).astype(int)
    
    i_dft = Kz+nw*(Ky+nv*Kx)
         
    return A_k.flatten(), i_dft

def intensity(A_k, 
              i_dft,
              factors):
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = A_k.shape[0] // n_atm
    
    A_k = A_k.reshape(n_uvw,n_atm)
    
    prod = factors*A_k[i_dft,:]

    F = np.sum(prod, axis=1)
                  
    I = np.real(F)**2+np.imag(F)**2
     
    return I/(n_uvw*n_atm)

def structure(A_k, 
              i_dft,
              factors):
    
    n_peaks = i_dft.shape[0]
    
    n_atm = factors.shape[0] // n_peaks
        
    factors = factors.reshape(n_peaks,n_atm)
    
    n_uvw = A_k.shape[0] // n_atm        
    
    A_k = A_k.reshape(n_uvw,n_atm)
    
    prod = factors*A_k[i_dft,:]

    F = np.sum(prod, axis=1)
     
    return F, prod.flatten()

def interpolate(G, mask, n=2, tol=0.0001):
        
    metric = np.round(G[mask]/tol).astype(int)
    
    u, indices, inverses = np.unique(metric, 
                                     return_index=True, 
                                     return_inverse=True)
    
    G_nuc = G[mask][indices]
    
    n_nuc = G_nuc.shape[0]
    
    diff = np.diff(G_nuc)/n
    
    Q = np.zeros((n_nuc-1)*n)
    
    for i in range(n):
        Q[i::n] = G_nuc[:n_nuc-1]+diff*i
        
    Q = np.append(Q,G_nuc[-1]) #np.append(G_nuc[0],Q)
    
    return Q, indices, inverses
        
def bragg(T, Qx, Qy, Qz, ux, uy, uz, scattering_length, cond, n_atm):
    
    phase_factor = np.exp(1j*(np.kron(Qx[cond],ux)\
                             +np.kron(Qy[cond],uy)\
                             +np.kron(Qz[cond],uz)))
    
    F = scattering_length*T*phase_factor
    
    return np.sum(F.reshape(cond.sum(),n_atm),axis=1)
