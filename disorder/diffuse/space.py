#!/usr/bin/env python

import numpy as np

from scipy import spatial

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
    
    n_uvw = nu*nv*nw
    
    delta_k = np.fft.ifftn(delta_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2))*n_uvw

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
        
def bragg(Qx, Qy, Qz, ux, uy, uz, factors, cond, n_atm):
    
    phase_factor = np.exp(1j*(np.kron(Qx[cond],ux)\
                             +np.kron(Qy[cond],uy)\
                             +np.kron(Qz[cond],uz)))
    
    F = factors*phase_factor
    
    return np.sum(F.reshape(cond.sum(),n_atm),axis=1)

def debye_waller(h_range, 
                 k_range, 
                 l_range, 
                 nh, 
                 nk,
                 nl, 
                 U11, 
                 U22, 
                 U33, 
                 U23, 
                 U13, 
                 U12, 
                 a_, 
                 b_, 
                 c_,
                 T=np.eye(3)):
    
    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                             np.linspace(k_range[0],k_range[1],nk), 
                             np.linspace(l_range[0],l_range[1],nl), 
                             indexing='ij')
    
    h = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
    k = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
    l = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
        
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