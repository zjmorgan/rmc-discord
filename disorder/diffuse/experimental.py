#!/usr/bin/env python3

import numpy as np
from scipy import ndimage

from functools import reduce

from disorder.diffuse.scattering import rebin0, rebin1, rebin2

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