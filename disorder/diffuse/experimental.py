#!/usr/bin/env python3

import numpy as np
from scipy import ndimage
from nexusformat.nexus import nxload

from functools import reduce

from disorder.diffuse.scattering import rebin0, rebin1, rebin2

def data(filename):
    
    data = nxload(filename)
        
    signal = np.array(data.MDHistoWorkspace.data.signal.nxdata.T)
    error_sq = np.array(data.MDHistoWorkspace.data.errors_squared.nxdata.T)
            
    if ('Q1' in data.MDHistoWorkspace.data.keys()):
        Qh = data.MDHistoWorkspace.data['Q1']
        Qk = data.MDHistoWorkspace.data['Q2']
        Ql = data.MDHistoWorkspace.data['Q3']
    elif ('[H,0,0]' in data.MDHistoWorkspace.data.keys()):
        Qh = data.MDHistoWorkspace.data['[H,0,0]']
        Qk = data.MDHistoWorkspace.data['[0,K,0]']
        Ql = data.MDHistoWorkspace.data['[0,0,L]']       
        
    Qh_min, Qk_min, Ql_min = Qh.min(), Qk.min(), Ql.min()
    Qh_max, Qk_max, Ql_max = Qh.max(), Qk.max(), Ql.max()
    
    mh, mk, ml = Qh.size, Qk.size, Ql.size
                
    nh = mh-1
    nk = mk-1
    nl = ml-1
    
    step_h = (Qh_max-Qh_min)/nh
    step_k = (Qk_max-Qk_min)/nk
    step_l = (Ql_max-Ql_min)/nl
                
    min_h = np.round(Qh_min+step_h/2, 4)
    min_k = np.round(Qk_min+step_k/2, 4)
    min_l = np.round(Ql_min+step_l/2, 4)
    
    max_h = np.round(Qh_max-step_h/2, 4)
    max_k = np.round(Qk_max-step_k/2, 4)
    max_l = np.round(Ql_max-step_l/2, 4)
    
    h_range, k_range, l_range = [min_h, max_h], [min_k, max_k], [min_l, max_l]
    
    return signal, error_sq, h_range, k_range, l_range, nh, nk, nl

def mask(signal, error_sq):
    
    mask = np.isnan(signal)\
         + np.isinf(signal)\
         + np.less_equal(signal, 0, where=~np.isnan(signal))\
         + np.isnan(error_sq)\
         + np.isinf(error_sq)\
         + np.less_equal(error_sq, 0, where=~np.isnan(error_sq))
         
    return mask

def punch(data,
          radius_h, 
          radius_k, 
          radius_l, 
          step_h, 
          step_k, 
          step_l, 
          h_range, 
          k_range, 
          l_range, 
          centering='P',
          outlier=1.5,
          punch='Box'):
    
    box = [int(round(radius_h)), int(round(radius_k)), int(round(radius_l))]
        
    min_h, max_h = h_range
    min_k, max_k = k_range
    min_l, max_l = l_range
    
    h_range = [int(round(min_h)), int(round(max_h))]
    k_range = [int(round(min_k)), int(round(max_k))]
    l_range = [int(round(min_l)), int(round(max_l))]
        
    for h in range(h_range[0], h_range[1]+1):
        for k in range(k_range[0], k_range[1]+1):
            for l in range(l_range[0], l_range[1]+1):
                
                if reflections(h, k, l, centering=centering):
                    
                    i_hkl = [int(np.round((h-h_range[0])/step_h,4)),\
                             int(np.round((k-k_range[0])/step_k,4)),\
                             int(np.round((l-l_range[0])/step_l,4))]
                            
                    h0, h1 = i_hkl[0]-box[0], i_hkl[0]+box[0]+1  
                    k0, k1 = i_hkl[1]-box[1], i_hkl[1]+box[1]+1      
                    l0, l1 = i_hkl[2]-box[2], i_hkl[2]+box[2]+1
                    
                    h0 = i_hkl[0]-box[0]
                    
                    if (h0 < 0): h0 = 0
                    if (k0 < 0): k0 = 0
                    if (l0 < 0): l0 = 0
                    
                    if (h1 >= data.shape[0]): h1 = data.shape[0]
                    if (k1 >= data.shape[1]): k1 = data.shape[1]
                    if (l1 >= data.shape[2]): l1 = data.shape[2]
                                        
                    values = data[h0:h1,k0:k1,l0:l1].copy()
                    
                    if (punch == 'Ellipsoid'):
                        values_outside = values.copy()
                        x, y, z = np.meshgrid(np.arange(h0,h1)-i_hkl[0], 
                                              np.arange(k0,k1)-i_hkl[1], 
                                              np.arange(l0,l1)-i_hkl[2], 
                                              indexing='ij')
                        mask = (x/box[0])**2+(y/box[1])**2+(z/box[2])**2 > 1
                        values[mask] = np.nan

                    Q3 = np.nanpercentile(values,75)
                    Q1 = np.nanpercentile(values,25)
                
                    interquartile = Q3-Q1                
                
                    reject = (values >= Q3+outlier*interquartile) | \
                             (values <  Q1-outlier*interquartile)
                    
                    values[reject] = np.nan
                    
                    if (punch == 'Ellipsoid'):
                        values[mask] = values_outside[mask].copy()
                       
                    data[h0:h1,k0:k1,l0:l1] = values.copy()
                    
    return data
  
def rebin(a, binsize):
    
    changed = np.array(binsize) != np.array(a.shape)
    
    if (changed[0] and changed[1] and changed[2]):
        comp0 = weights(a.shape[0], binsize[0])
        comp1 = weights(a.shape[1], binsize[1])
        comp2 = weights(a.shape[2], binsize[2])
        b = rebin0(a, comp0)
        c = rebin1(b, comp1)
        d = rebin2(c, comp2)
        return d
    elif (changed[0] and changed[1]):
        comp0 = weights(a.shape[0], binsize[0])
        comp1 = weights(a.shape[1], binsize[1])
        b = rebin0(a, comp0)
        c = rebin1(b, comp1)
        return c
    elif (changed[1] and changed[2]):
        comp1 = weights(a.shape[1], binsize[1])
        comp2 = weights(a.shape[2], binsize[2])
        b = rebin1(a, comp1)
        c = rebin2(b, comp2)
        return c
    elif (changed[2] and changed[0]):
        comp2 = weights(a.shape[2], binsize[2])
        comp0 = weights(a.shape[0], binsize[0])
        b = rebin2(a, comp2)
        c = rebin0(b, comp0)
        return c
    elif (changed[0]):
        comp0 = weights(a.shape[0], binsize[0])
        b = rebin0(a, comp0)
        return b
    elif (changed[1]):
        comp1 = weights(a.shape[1], binsize[1])
        b = rebin1(a, comp1)
        return b  
    elif (changed[2]):
        comp2 = weights(a.shape[2], binsize[2])
        b = rebin2(a, comp2)
        return b      
    else:
        return a
    
def weights(old, new):
    weights = np.zeros((new,old))
    binning = old/new
    interval = binning
    row, col = 0, 0
    while (row < weights.shape[0] and col < weights.shape[1]):
        if (np.round(interval-col, 1) >= 1):
            weights[row,col] = 1
            col += 1
        elif (interval == col):
            row += 1
            interval += binning
        else:
            partial = interval-col
            weights[row,col] = partial
            row += 1
            weights[row,col] = 1-partial
            col += 1
            interval += binning
    weights /= binning
    return weights

def crop(x, h_slice, k_slice, l_slice):
    
    if (h_slice[0] == 0 and h_slice[1] == x.shape[0] and 
        k_slice[0] == 0 and k_slice[1] == x.shape[1] and 
        l_slice[0] == 0 and l_slice[1] == x.shape[2]):
        
        return np.ascontiguousarray(x)
    
    else:
        
        return np.ascontiguousarray(x[h_slice[0]:h_slice[1],\
                                      k_slice[0]:k_slice[1],\
                                      l_slice[0]:l_slice[1]].copy(order='C'))
    
def factors(n): 
    
    return np.unique(reduce(list.__add__, 
      ([i, n/i] for i in range(1, int(n**0.5) + 1) if n % i == 0))).astype(int)
    
def karen(signal, width):
    
    median = ndimage.filters.median_filter(signal, size=width)
        
    mad = ndimage.filters.median_filter(np.abs(signal-median), size=width) 
    
    asigma = np.abs(mad*3*1.4826) 
    
    mask = np.logical_or(signal < (median-asigma), signal > (median+asigma)) 
    signal[mask] = np.nan

    median = ndimage.filters.median_filter(signal, size=width)
        
    mad = ndimage.filters.median_filter(np.abs(signal-median), size=width) 
    
    asigma = np.abs(mad*3*1.4826) 
    
    mask = np.logical_or(signal < (median-asigma), signal > (median+asigma)) 
    signal[mask] = np.nan
    
    return signal

def reflections(h, k, l, centering='P'):
    
    # centering == 'P', 'R (rhombohedral axes, primitive cell')
    allow = 1
            
    if (centering == 'I'):
        if ((h+k+l) % 2 != 0):
            allow = 0
            
    elif (centering == 'F'):
        if ((h+k) % 2 != 0 or (k+l) % 2 != 0 or (l+h) % 2 != 0):
            allow = 0
            
    elif (centering == 'A'):
        if ((k+l) % 2 != 0):
            allow = 0
            
    elif (centering == 'B'):
        if ((l+h) % 2 != 0):
            allow = 0
            
    elif (centering == 'C'):
        if ((h+k) % 2 != 0):
            allow = 0
            
    elif (centering == 'R (hexagonal axes, triple obverse cell)'):
        if ((-h+k+l) % 3 != 0):
            allow = 0
            
    elif (centering == 'R (hexagonal axes, triple reverse cell)'):
        if ((h-k+l) % 3 != 0):
            allow = 0
            
    elif (centering == 'H (hexagonal axes, triple hexagonal cell)'):
        if ((h-k) % 3 != 0):
            allow = 0
            
    elif (centering == 'D (rhombohedral axes, triple rhombohedral cell)'):
        if ((h+k+l) % 3 != 0):
            allow = 0    
            
    return allow