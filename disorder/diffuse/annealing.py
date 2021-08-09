#!/usr/bin/env python3

import numpy as np
import sys

from scipy.optimize import fsolve

PERCENTAGE = -1
    
import disorder.diffuse.original as orig
import disorder.diffuse.candidate as cand

def original(A, B=None, C=None):
                            
    if (B is None and C is None):
                
        return orig.scalar(A)
    
    else:

        return orig.vector(A, B, C)
    
def candidate(A, B=None, C=None, value=None, fixed=True):
            
    if (value is None):
        
        V = 1.
        
    else:
        
        V = value
        
    if (B is None and C is None):
        
        return cand.scalar(A, V, fixed)
    
    else:
                                
        return cand.vector(A, B, C, V, fixed)
    
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