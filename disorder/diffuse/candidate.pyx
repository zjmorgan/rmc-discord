#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport M_PI, sin, cos, acos, fabs, sqrt, log
from libc.stdlib cimport rand, RAND_MAX

cdef double M_EPS = np.finfo(float).eps

cdef double random_uniform_nonzero() nogil:
    
    cdef double u = 0
    
    while (u == 0):
        u = float(rand())/RAND_MAX
        
    return u
    
cdef double random_uniform() nogil:
        
    return float(rand())/RAND_MAX

cdef double random_gaussian() nogil:
    
    cdef double x0, x1, w

    w = 2.0
    while (w >= 1.0):
        x0 = 2.0*random_uniform()-1.0
        x1 = 2.0*random_uniform()-1.0
        w = x0*x0+x1*x1
        
    w = sqrt(-2.0*log(w)/w)
    
    return x1*w

cdef (double, double, double) random_gaussian_3d() nogil:
    
    cdef double x0, x1, w
    cdef double x2, x3, v

    w = 2.0
    while (w >= 1.0):
        x0 = 2.0*random_uniform()-1.0
        x1 = 2.0*random_uniform()-1.0
        w = x0*x0+x1*x1

    w = sqrt(-2.0*log(w)/w)
    
    return x0*w, x1*w, random_gaussian()

cdef bint iszero(double a) nogil:
        
    return fabs(a) <= M_EPS

cpdef double composition(double A, double value, bint fixed) nogil:
    
    cdef double V = value if fixed else value*random_uniform_nonzero()
                                    
    return 1/V-2-A

cpdef (double, double, double) moment(double A, 
                                      double B, 
                                      double C, 
                                      double value, 
                                      bint fixed,
                                      bint rotate) nogil:
    
    cdef double theta, phi
                        
    cdef double u, v, w, n
        
    cdef double V
        
    if rotate:
        
        V = value if fixed else value*random_uniform_nonzero()
        
        theta = 2.0*M_PI*random_uniform()
        phi = acos(1.0-2.0*random_uniform())
    
        u = V*sin(phi)*cos(theta)
        v = V*sin(phi)*sin(theta)
        w = V*cos(phi)
        
    else:
        
        if fixed:  
            
            u, v, w = -A, -B, -C
            
        else:
            
            n = sqrt(A*A+B*B+C*C)
            
            if iszero(n):
                
                u, v, w = -A, -B, -C
                
            else:
                
                V = value if fixed else value*random_uniform_nonzero()
                
                u, v, w = -V*A/n, -V*B/n, -V*C/n                
                
    return u, v, w

cpdef (double, double, double) displacement(double A, 
                                            double B, 
                                            double C, 
                                            double D, 
                                            double E, 
                                            double F, 
                                            bint fixed,
                                            bint isotropic) nogil:
    
    cdef double theta, phi
                        
    cdef double u, v, w, l, m, n
                
    if fixed:
        
        theta = 2.0*M_PI*random_uniform()
        phi = acos(1.0-2.0*random_uniform())
    
        l = sin(phi)*cos(theta)
        m = sin(phi)*sin(theta)
        n = cos(phi)
        
        if isotropic:
            
            u, v, w = A*l, B*m, C*n
            
        else:
            
            u = sqrt(A*(A*l+F*m+E*n))*l
            v = sqrt(B*(F*l+B*m+D*n))*m
            w = sqrt(C*(E*l+D*m+C*n))*n
        
    else:
        
        l, m, n = random_gaussian_3d()
        
        if isotropic: 
                        
            u, v, w = A*l, B*m, C*n
            
        else:
                        
            u, v, w = A*l, F*l+B*m, E*l+D*m+C*n
                
    return u, v, w