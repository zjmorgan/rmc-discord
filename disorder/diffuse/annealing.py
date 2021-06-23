#!/usr/bin/env python3

import numpy as np
import sys

from scipy.optimize import fsolve

PERCENTAGE = -1

def f(x,y):
    return (x-np.sin(x))/np.pi-y

def g(x,y):
    return (1-np.cos(x))/np.pi
    
import disorder.diffuse.original as orig
import disorder.diffuse.candidate as cand

def original(A, B=None, C=None):
                            
    if (B is None and C is None):
                
        return orig.scalar(A)
    
    else:

        return orig.vector(A, B, C)
    
def originals(A, B=None, C=None, structure=None):
    
    if (B is None and C is None):
        
        A_orig = np.zeros(structure.shape[1], dtype=np.double)
        
        i = np.zeros(structure.shape[1], dtype=np.int)
        
        k = orig.scalars(A_orig, i, A)
        
        return A_orig, i, k
    
    else:
        
        A_orig = np.zeros(structure.shape[1], dtype=np.double)
        B_orig = np.zeros(structure.shape[1], dtype=np.double)
        C_orig = np.zeros(structure.shape[1], dtype=np.double)
        
        i = np.zeros(structure.shape[1], dtype=np.int)
        
        k = orig.vectors(A_orig, B_orig, C_orig, i, A, B, C)

        return A_orig, B_orig, C_orig, i, k
    
def candidate(A, B=None, C=None, delta=1, value=None, fixed=True, T=None):
            
    if (value is None):
        
        V = 1.
        
    else:
        
        V = value
    
    if (T is None):
        
        T = np.array([1., 1., 1., 0., 0., 0.])
        
    if (B is None and C is None):
        
        return cand.scalar(A, V, fixed)
    
    else:
                                
        return cand.vector(A, B, C, delta, V, fixed, T)
    
def candidates(A, B=None, C=None, delta=1, value=None, fixed=True):
    
    if (value is None):
        
        V = 1.
        
    else:
        
        V = value
        
    if (B is None and C is None):
        
        A_cand = np.zeros(A.shape[0], dtype=np.double)
        
        cand.vectors(A_cand, A, V, fixed)
        
        return A_cand
    
    else:
        
        A_cand = np.zeros(A.shape[0], dtype=np.double)
        B_cand = np.zeros(B.shape[0], dtype=np.double)
        C_cand = np.zeros(C.shape[0], dtype=np.double)
        
        cand.vectors(A_cand, B_cand, C_cand, A, B, C, delta, V, fixed)

        return A_cand, B_cand, C_cand
                            
def test(beta, E):
    
    return np.exp(-beta*E)

def completion(i, N):
    
    global PERCENTAGE
    
    percentage = np.int(100.*i/N)
    
    if (percentage > PERCENTAGE):
    
        sys.stdout.write('\rpercentage: %02d' % percentage)
        sys.stdout.flush()
        
    PERCENTAGE = percentage

def memory(n):
    
    return str(np.round(n*8*(3+2*4)*1e-9,1))+' GB'