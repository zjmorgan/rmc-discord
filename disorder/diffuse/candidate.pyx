#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport M_PI, sin, cos, acos, fabs, sqrt
from libc.stdlib cimport rand, RAND_MAX
    
#from scipy.optimize.cython_optimize cimport brentq

cdef double MACHINE_EPSILON = np.finfo(float).eps

XLO, XHI = 0.0, M_PI
XTOL, RTOL, MITR = 1e-3, 1e-3, 10

cdef double f(double x, double y) nogil:
    return (x-sin(x))/M_PI-y
    
cpdef double scalar(double A, double value, bint fixed) nogil:
    
    cdef double V
    
    if (fixed):
        V = value
    else:
        V = value*rand()/RAND_MAX
                                
    return 1/V-2-A

cpdef (double, double, double) vector(double A, 
                                      double B, 
                                      double C, 
                                      double delta, 
                                      double value, 
                                      bint fixed,
                                      double [::1] T) nogil:
    
    cdef double V
    
    cdef double theta = 2*M_PI*rand()/RAND_MAX
    cdef double phi = acos(1-2.*rand()/RAND_MAX)
                        
    cdef double u, v, w
    cdef double x, y, z
    
    cdef double dot, alpha, beta, factor, inv_factor
    
    cdef double i, j, k
    cdef double l, m, n
    
    cdef double s = sqrt(A*A+B*B+C*C)
    
    cdef double a, b, c
        
    if (fixed):
        V = value
    else:
        V = value*rand()/RAND_MAX

    u = sin(phi)*cos(theta)
    v = sin(phi)*sin(theta)
    w = cos(phi)
    
    if (fabs(s) <= MACHINE_EPSILON):
        
        return V*(T[0]*u+T[5]*v+T[4]*w), \
               V*(T[5]*u+T[1]*v+T[3]*w), \
               V*(T[4]*u+T[3]*v+T[2]*w)
               
    a = A/s
    b = B/s
    c = C/s
    
    dot = a*u+b*v+c*w
        
    if (fabs(dot) > 1):
        
        return A, B, C
    
    else:
    
        alpha = acos(dot)
        beta = delta*alpha
        
        inv_factor = sin(alpha)

        if (fabs(inv_factor) <= MACHINE_EPSILON):
            
            return V*(T[0]*u+T[5]*v+T[4]*w), \
                   V*(T[5]*u+T[1]*v+T[3]*w), \
                   V*(T[4]*u+T[3]*v+T[2]*w)
            
        else:
            
            factor = 1/inv_factor
                        
            i = (b*w-c*v)*factor
            j = (c*u-a*w)*factor
            k = (a*v-b*u)*factor
                    
            l = (j*c-k*b)
            m = (k*a-i*c)
            n = (i*b-j*a)
            
            x = a*cos(beta)+l*sin(beta)
            y = b*cos(beta)+m*sin(beta)
            z = c*cos(beta)+n*sin(beta)
            
            return V*(T[0]*x+T[5]*y+T[4]*z), \
                   V*(T[5]*x+T[1]*y+T[3]*z), \
                   V*(T[4]*x+T[3]*y+T[2]*z)
    
cpdef void scalars(double [::1] B, 
                   double [::1] A, 
                   double value, 
                   bint fixed) nogil:
    
    cdef double V
    
    cdef int n = A.shape[0]
    
    cdef int i

    if (fixed):
        V = value
    else:
        V = value*rand()/RAND_MAX  

    for i in range(n):
        B[i] = 1/V-2-A[i]                   
                   
cpdef void vectors(double [::1] D, 
                   double [::1] E, 
                   double [::1] F, 
                   double [::1] A, 
                   double [::1] B, 
                   double [::1] C, 
                   double delta, 
                   double value, 
                   bint fixed) nogil:
    
    cdef double V
    
    cdef double theta = 2*M_PI*rand()
    cdef double phi = acos(1-2.*rand()/RAND_MAX)
    
    cdef double psi
    
    cdef double u, v, w
    cdef double a, b, c
            
    cdef double R00, R01, R02
    cdef double R10, R11, R12
    cdef double R20, R21, R22
    
    cdef int n = A.shape[0]
    
    cdef int i
    
    if (fixed):
        V = value
    else:
        V = value*rand()/RAND_MAX  
    
    psi = 0
    
    u = sin(phi)*cos(theta)
    v = sin(phi)*sin(theta)
    w = cos(phi)
    
    a = sin(phi)*cos(theta)
    b = sin(phi)*sin(theta)
    c = cos(phi)
    
    R00 =    cos(psi)+u*u*(1-cos(psi))
    R01 = -w*sin(psi)+u*v*(1-cos(psi))
    R02 =  v*sin(psi)+u*w*(1-cos(psi))
    
    R10 =  w*sin(psi)+u*v*(1-cos(psi))
    R11 =    cos(psi)+v*v*(1-cos(psi))
    R12 = -u*sin(psi)+v*w*(1-cos(psi))
    
    R20 = -v*sin(psi)+u*w*(1-cos(psi))
    R21 =  u*sin(psi)+v*w*(1-cos(psi))
    R22 =    cos(psi)+w*w*(1-cos(psi))
    
    for i in range(n):
        D[i] = R00*A[i]+R01*B[i]+R02*C[i]+V*a
        E[i] = R10*A[i]+R11*B[i]+R12*C[i]+V*b
        F[i] = R20*A[i]+R21*B[i]+R22*C[i]+V*c