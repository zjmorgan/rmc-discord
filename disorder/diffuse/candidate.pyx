#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport M_PI, sin, cos, acos, fabs, sqrt
from libc.stdlib cimport rand, RAND_MAX
    
cpdef double scalar(double A, double value, bint fixed) nogil:
    
    cdef double V
    
    if fixed:
        V = value
    else:
        V = value*rand()/RAND_MAX
                                
    return 1/V-2-A

cpdef (double, double, double) vector(double A, 
                                      double B, 
                                      double C, 
                                      double value, 
                                      bint fixed) nogil:
    
    cdef double V
    
    cdef double theta = 2*M_PI*rand()/RAND_MAX
    cdef double phi = acos(1-2.*rand()/RAND_MAX)
                        
    cdef double u, v, w
        
    if fixed:
        V = value
    else:
        V = value*rand()/RAND_MAX

    u = V*sin(phi)*cos(theta)
    v = V*sin(phi)*sin(theta)
    w = V*cos(phi)
            
    return u, v, w