#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

cimport cython

import sys
    
PERCENTAGE = -1

def completion(i, N):
    
    global PERCENTAGE
    
    percentage = np.int(100.*i/N)
    
    if (percentage > PERCENTAGE):
    
        sys.stdout.write('\rpercentage: %02d' % percentage)
        sys.stdout.flush()
        
    PERCENTAGE = percentage