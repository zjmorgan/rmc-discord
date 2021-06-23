#!/usr/bin/env python3

import numpy as np
from scipy import signal, spatial

from disorder.correlation import radii
import itertools

#from scipy import constants

# --- constants

def K():
    
    return 1

def gamma(rx,ry,rz):
    
    r_sq = rx**2+ry**2+rz**2
    
    mask = np.isclose(r_sq, 0)
    r_sq[mask] = 1
    
    Gamma_x, Gamma_y, Gamma_z = rx/r_sq**2, ry/r_sq**2, rz/r_sq**2
    
    Gamma_x[mask] = 0
    Gamma_y[mask] = 0
    Gamma_z[mask] = 0
    
    return Gamma_x, Gamma_y, Gamma_z
    
def delta_pdf(Sx,
              Sy,
              Sz,
              rx,
              ry,
              rz,
              nu,
              nv,
              nw,
              n_atm):
    
    Gamma_x, Gamma_y, Gamma_z = gamma(rx,ry,rz)
    
    n_atm = Sx.shape[0] // (nu*nv*nw)

    Sx = Sx.reshape(nu, nv, nw, n_atm)
    Sy = Sy.reshape(nu, nv, nw, n_atm)
    Sz = Sz.reshape(nu, nv, nw, n_atm)
    
    Gamma_x = Gamma_x.reshape(nu, nv, nw, n_atm)
    Gamma_y = Gamma_y.reshape(nu, nv, nw, n_atm)
    Gamma_z = Gamma_z.reshape(nu, nv, nw, n_atm)
    
    S_convolve_Gamma = (signal.fftconvolve(Sx, Gamma_x, mode='same', axes=[0,1,2])+
                        signal.fftconvolve(Sy, Gamma_y, mode='same', axes=[0,1,2])+
                        signal.fftconvolve(Sz, Gamma_z, mode='same', axes=[0,1,2]))/np.pi**2
    
    delta = K()*(signal.fftconvolve(Sx, Sx[::-1,::-1,::-1,:], mode='same', axes=[0,1,2])\
                +signal.fftconvolve(Sy, Sy[::-1,::-1,::-1,:], mode='same', axes=[0,1,2])\
                +signal.fftconvolve(Sz, Sz[::-1,::-1,::-1,:], mode='same', axes=[0,1,2])\
                -0*signal.fftconvolve(S_convolve_Gamma,\
                                    S_convolve_Gamma[::-1,::-1,::-1,:], mode='same', axes=[0,1,2]))/nu/nv/nw
              
    return np.sum(delta, axis=3)/n_atm

def periodic(array, nu, nv, nw, n_atm):
    
    array = array.reshape(nu,nv,nw,n_atm)
    
    array = np.concatenate((array[nu//2:,:,:,:],
                            array,
                            array[:nu//2,:,:,:]),
                            axis=0)
    
    array = np.concatenate((array[:,nv//2:,:,:],
                            array,
                            array[:,:nv//2,:,:]),
                            axis=1)
    
    array = np.concatenate((array[:,:,nw//2:,:],
                            array,
                            array[:,:,:nw//2,:]),
                            axis=2)
    
    return array
    
def radial(Sx, 
           Sy, 
           Sz, 
           rx, 
           ry, 
           rz, 
           ion, 
           fract=0.1, 
           tol=0.0001, 
           maximum=None, 
           mask=None,
           period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        Sx = periodic(Sx, nu, nv, nw, n_atm).flatten()
        Sy = periodic(Sy, nu, nv, nw, n_atm).flatten()
        Sz = periodic(Sz, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
        
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2
    
    if (mask is not None):

        Sx = Sx[mask]
        Sy = Sy[mask]
        Sz = Sz[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([1.]), \
               np.array([1.]), \
               np.array([1.]), \
               np.array([1.]), \
               np.array([0.]), \
               np.array(['0'])
    
    coordinate = np.array(list(pairs))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
    
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)
        
    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
        
    distance = np.sqrt(dx**2+dy**2+dz**2)
        
    metric =  np.stack((np.round(np.round(distance/tol,1)).astype(int), \
                        ion_labels)).T
                            
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]
    
    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
    
    search = np.append(indices,len(distance))
            
    D = unique.shape[0]
    
    S_corr = np.zeros(D)
    S_coll = np.zeros(D)
    
    S_corr_ = np.zeros(D)
    S_coll_ = np.zeros(D)
        
    distance = distance[indices]                   
    ion_pair = ion_ion[indices]
    
    radii.averaging(S_corr, Sx, Sy, Sz, counts, search, coordinate)
    radii.varying(S_coll, Sx, Sy, Sz, counts, search, coordinate)
    radii.scaling(S_corr_, Sx, Sy, Sz, counts, search, coordinate)
    
    S_coll_ = S_corr**2
         
    return np.insert(S_corr, 0, 1), \
           np.insert(S_coll, 0, 1), \
           np.insert(S_corr_, 0, 1), \
           np.insert(S_coll_, 0, 1), \
           np.insert(distance, 0, 0), \
           np.insert(ion_pair, 0, '0')

def radial3d(Sx, 
             Sy, 
             Sz, 
             rx, 
             ry, 
             rz, 
             ion, 
             fract=0.1, 
             tol=0.0001, 
             maximum=None, 
             mask=None,
             period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        Sx = periodic(Sx, nu, nv, nw, n_atm).flatten()
        Sy = periodic(Sy, nu, nv, nw, n_atm).flatten()
        Sz = periodic(Sz, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
                
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2
        
    if (mask is not None):

        Sx = Sx[mask]
        Sy = Sy[mask]
        Sz = Sz[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([1.]), \
               np.array([1.]), \
               np.array([1.]), \
               np.array([1.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array(['0'])
               
    coordinate = np.array(list(pairs))
        
    # inversion
    coordinate = np.concatenate((coordinate,np.fliplr(coordinate)))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
    
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)

    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
    
    distance = np.stack((dx,dy,dz)).T
    
    metric =  np.vstack((np.round(np.round(distance.T/tol,1)).astype(int), \
                         ion_labels)).T
    
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]

    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
            
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    S_corr = np.zeros(D)
    S_coll = np.zeros(D)
    
    S_corr_ = np.zeros(D)
    S_coll_ = np.zeros(D)
    
    dx = dx[sort][indices]
    dy = dy[sort][indices]
    dz = dz[sort][indices]
    ion_pair = ion_ion[indices]
                       
    radii.averaging(S_corr, Sx, Sy, Sz, counts, search, coordinate)
    radii.varying(S_coll, Sx, Sy, Sz, counts, search, coordinate)
    radii.scaling(S_corr_, Sx, Sy, Sz, counts, search, coordinate)

    S_coll_ = S_corr**2
    
    return np.insert(S_corr, 0, 1), \
           np.insert(S_coll, 0, 1), \
           np.insert(S_corr_, 0, 1), \
           np.insert(S_coll_, 0, 1), \
           np.insert(dx, 0, 0), \
           np.insert(dy, 0, 0), \
           np.insert(dz, 0, 0), \
           np.insert(ion_pair, 0, '0')

def parameter(A, 
              rx, 
              ry, 
              rz, 
              ion, 
              fract=0.1, 
              tol=0.0001, 
              maximum=None, 
              mask=None,
              period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        A = periodic(A, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
        
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2
    
    if (mask is not None):

        A = A[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([1.]), \
               np.array([1.]), \
               np.array([0.]), \
               np.array(['0'])
               
    coordinate = np.array(list(pairs))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
        
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)
        
    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
        
    distance = np.sqrt(dx**2+dy**2+dz**2)
        
    metric =  np.stack((np.round(np.round(distance/tol,1)).astype(int), \
                        ion_labels)).T
    
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]

    unique, indices, counts = np.unique(metric, 
                              axis=0,
                              return_index=True, 
                              return_counts=True)
        
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    S_corr = np.zeros(D)
    S_corr_ = np.zeros(D)
        
    distance = distance[indices]
    ion_pair = ion_ion[indices]
                        
    radii.ordering(S_corr, A, counts, search, coordinate)
    radii.fluctuating(S_corr_, A, counts, search, coordinate)
         
    return np.insert(S_corr, 0, 1), \
           np.insert(S_corr_, 0, 1), \
           np.insert(distance, 0, 0), \
           np.insert(ion_pair, 0, '0')
           
def parameter3d(A, 
                rx, 
                ry, 
                rz, 
                ion, 
                fract=0.1, 
                tol=0.0001, 
                maximum=None, 
                mask=None,
                period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        A = periodic(A, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
        
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2

    if (mask is not None):

        A = A[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([1.]), \
               np.array([1.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array(['0'])
               
    coordinate = np.array(list(pairs))
    
    # inversion
    coordinate = np.concatenate((coordinate,np.fliplr(coordinate)))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
    
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)

    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
    
    distance = np.stack((dx,dy,dz)).T
    
    metric =  np.vstack((np.round(np.round(distance.T/tol,1)).astype(int), \
                         ion_labels)).T
    
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]

    unique, indices, counts = np.unique(metric, 
                              axis=0,
                              return_index=True, 
                              return_counts=True)
            
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    S_corr = np.zeros(D)
    S_corr_ = np.zeros(D)
    
    dx = dx[sort][indices]
    dy = dy[sort][indices]
    dz = dz[sort][indices]
    ion_pair = ion_ion[indices]
                   
    radii.ordering(S_corr, A, counts, search, coordinate)
    radii.fluctuating(S_corr_, A, counts, search, coordinate)
    
    return np.insert(S_corr, 0, 1), \
           np.insert(S_corr_, 0, 1), \
           np.insert(dx, 0, 0), \
           np.insert(dy, 0, 0), \
           np.insert(dz, 0, 0), \
           np.insert(ion_pair, 0, '0')
           
def combination(delta, 
                Sx, 
                Sy, 
                Sz, 
                rx, 
                ry, 
                rz, 
                ion, 
                fract=0.1, 
                tol=0.0001, 
                maximum=None, 
                mask=None,
                period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        delta = periodic(delta, nu, nv, nw, n_atm).flatten()
        Sx = periodic(Sx, nu, nv, nw, n_atm).flatten()
        Sy = periodic(Sy, nu, nv, nw, n_atm).flatten()
        Sz = periodic(Sz, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
        
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2

    if (mask is not None):

        delta = delta[mask]
        
        Sx = Sx[mask]
        Sy = Sy[mask]
        Sz = Sz[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([0.]), \
               np.array([0.]), \
               np.array(['0'])
               
    coordinate = np.array(list(pairs))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
    
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)
        
    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
        
    distance = np.sqrt(dx**2+dy**2+dz**2)
        
    metric =  np.stack((np.round(np.round(distance/tol,1)).astype(int), \
                        ion_labels)).T
    
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]

    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
        
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    S_corr = np.zeros(D)
        
    distance = distance[indices]
    ion_pair = ion_ion[indices]
    
    radii.effect(S_corr, 
                 delta, 
                 Sx, 
                 Sy, 
                 Sz, 
                 rx, 
                 ry, 
                 rz,
                 counts, 
                 search, 
                 coordinate)
         
    return np.insert(S_corr, 0, 0), \
           np.insert(distance, 0, 0), \
           np.insert(ion_pair, 0, '0')
           
def combination3d(delta, 
                  Sx, 
                  Sy, 
                  Sz, 
                  rx, 
                  ry, 
                  rz, 
                  ion, 
                  fract=0.1, 
                  tol=0.0001, 
                  maximum=None,
                  mask=None,
                  period=None):
    
    if (period is not None):
        
        A_lat, nu, nv, nw, n_atm = period
        
        delta = periodic(delta, nu, nv, nw, n_atm).flatten()
        Sx = periodic(Sx, nu, nv, nw, n_atm).flatten()
        Sy = periodic(Sy, nu, nv, nw, n_atm).flatten()
        Sz = periodic(Sz, nu, nv, nw, n_atm).flatten()
        
        A_inv = np.linalg.inv(A_lat)
        
        u = A_inv[0,0]*rx+A_inv[0,1]*ry+A_inv[0,2]*rz
        v = A_inv[1,0]*rx+A_inv[1,1]*ry+A_inv[1,2]*rz
        w = A_inv[2,0]*rx+A_inv[2,1]*ry+A_inv[2,2]*rz
        
        u = periodic(u, nu, nv, nw, n_atm)
        v = periodic(v, nu, nv, nw, n_atm)
        w = periodic(w, nu, nv, nw, n_atm)
        
        u[:(nu+1)//2,:,:,:] -= nu
        u[nu+(nu+1)//2:,:,:,:] += nu
        
        v[:,:(nv+1)//2,:,:] -= nv
        v[:,nv+(nv+1)//2:,:,:] += nv
        
        w[:,:,:(nw+1)//2,:,] -= nw
        w[:,:,nw+(nw+1)//2:,:] += nw
        
        rx = (A_lat[0,0]*u+A_lat[0,1]*v+A_lat[0,2]*w).flatten()
        ry = (A_lat[1,0]*u+A_lat[1,1]*v+A_lat[1,2]*w).flatten()
        rz = (A_lat[2,0]*u+A_lat[2,1]*v+A_lat[2,2]*w).flatten()
        
        ion = periodic(ion, nu, nv, nw, n_atm).flatten()
        
        if (mask is not None):
            
            mask = periodic(mask, nu, nv, nw, n_atm).flatten()
            
        fract /= 2

    if (mask is not None):

        delta = delta[mask]
        
        Sx = Sx[mask]
        Sy = Sy[mask]
        Sz = Sz[mask]
        
        rx = rx[mask]
        ry = ry[mask]
        rz = rz[mask]
        
        ion = ion[mask]
        
    r_max = np.sqrt(rx.max()**2+ry.max()**2+rz.max()**2)
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    pairs = tree.query_pairs(fract*r_max)
    
    if (len(pairs) == 0):
        return np.array([0.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array([0.]), \
               np.array(['0'])
               
    coordinate = np.array(list(pairs))
    
    # inversion
    coordinate = np.concatenate((coordinate,np.fliplr(coordinate)))
        
    i = coordinate[:,0]
    j = coordinate[:,1]
    
    dx = rx[j]-rx[i]
    dy = ry[j]-ry[i]
    dz = rz[j]-rz[i]
    
    if (maximum is not None):
        
        mask = (np.abs(dx) <= maximum[0])\
             & (np.abs(dy) <= maximum[1])\
             & (np.abs(dz) <= maximum[2])
    
        coordinate = coordinate[mask]
        
        i = i[mask]
        j = j[mask]
        
        dx = dx[mask]
        dy = dy[mask]
        dz = dz[mask]
        
    atms = np.sort(np.stack((ion[j],ion[i])), axis=0)

    ion_ion = np.core.defchararray.add( \
              np.core.defchararray.add(atms[0,:], '_'), atms[1,:])
        
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)
    
    distance = np.stack((dx,dy,dz)).T
    
    metric =  np.vstack((np.round(np.round(distance.T/tol,1)).astype(int), \
                         ion_labels)).T
    
    sort = np.lexsort(np.fliplr(metric).T)
        
    distance = distance[sort]
    coordinate[:,0] = i[sort]
    coordinate[:,1] = j[sort]
    
    ion_ion = ion_ion[sort]
    
    metric = metric[sort]

    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
            
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    S_corr = np.zeros(D)

    dx = dx[sort][indices]
    dy = dy[sort][indices]
    dz = dz[sort][indices]
    ion_pair = ion_ion[indices]
                   
    radii.effect(S_corr, 
                 delta, 
                 Sx, 
                 Sy, 
                 Sz, 
                 rx, 
                 ry, 
                 rz,
                 counts, 
                 search, 
                 coordinate)
    
    return np.insert(S_corr, 0, 0), \
           np.insert(dx, 0, 0), \
           np.insert(dy, 0, 0), \
           np.insert(dz, 0, 0), \
           np.insert(ion_pair, 0, '0')