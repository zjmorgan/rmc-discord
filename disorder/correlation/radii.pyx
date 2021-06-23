#cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport sqrt, fabs

cdef double iszero(double a) nogil:
    
    cdef double atol = 1e-08
    
    return fabs(a) <= atol

cpdef void averaging(double [::1] S_corr,
                     double [::1] Sx,
                     double [::1] Sy,
                     double [::1] Sz,
                     signed long long [::1] counts,
                     signed long long [::1] search,
                     long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_corr.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double U, V
    
    cdef Py_ssize_t count
    
    for r in range(D):
        
        count = counts[r]

        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            U = sqrt(Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i])
            V = sqrt(Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j])
                        
            if (not (iszero(U) or iszero(V))):
                            
                S_corr[r] += (Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j])/U/V
                          
            else:
                
                count -= 1
                
        if (count > 0):
            
            S_corr[r] /= count
                        
cpdef void scaling(double [::1] S_corr,
                   double [::1] Sx,
                   double [::1] Sy,
                   double [::1] Sz,
                   signed long long [::1] counts,
                   signed long long [::1] search,
                   long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_corr.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double U, V
    
    cdef double Sx_i, Sy_i, Sz_i
    cdef double Sx_j, Sy_j, Sz_j
    
    cdef Py_ssize_t count

    for r in range(D):
        
        count = counts[r]
        
        Sx_i, Sy_i, Sz_i = 0, 0, 0
        Sx_j, Sy_j, Sz_j = 0, 0, 0
        
        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            U = sqrt(Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i])
            V = sqrt(Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j])
            
            if (not (iszero(U) or iszero(V))):
            
                Sx_i += Sx[i]/U
                Sy_i += Sy[i]/U
                Sz_i += Sz[i]/U
                
                Sx_j += Sx[j]/V
                Sy_j += Sy[j]/V
                Sz_j += Sz[j]/V
                
            else:
                
                count -= 1
                
        if (count > 0):
                                        
            S_corr[r] = (Sx_i*Sx_j+Sy_i*Sy_j+Sz_i*Sz_j)/count**2
            
cpdef void varying(double [::1] S_coll,
                   double [::1] Sx,
                   double [::1] Sy,
                   double [::1] Sz,
                   signed long long [::1] counts,
                   signed long long [::1] search,
                   long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_coll.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double U, V
    
    cdef Py_ssize_t count

    for r in range(D):
        
        count = counts[r]

        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            U = Sx[i]*Sx[i]+Sy[i]*Sy[i]+Sz[i]*Sz[i]
            V = Sx[j]*Sx[j]+Sy[j]*Sy[j]+Sz[j]*Sz[j]
            
            if (not (iszero(U) or iszero(V))):
                            
                S_coll[r] += (Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j])**2/U/V
                          
            else:
                
                count -= 1
                
        if (count > 0):
            
            S_coll[r] /= count
                      
cpdef void ordering(double [::1] S_corr,
                    double [::1] A_r,
                    signed long long [::1] counts,
                    signed long long [::1] search,
                    long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_corr.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double U, V
    
    cdef Py_ssize_t count

    for r in range(D):
        
        count = counts[r]

        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            U = fabs(A_r[i])
            V = fabs(A_r[j])
                                        
            if (not (iszero(U) or iszero(V))):
                            
                S_corr[r] += (A_r[i]*A_r[j])/U/V

            else:
                
                count -= 1
                
        if (count > 0):
            
            S_corr[r] /= count
            
cpdef void fluctuating(double [::1] S_corr,
                       double [::1] A_r,
                       signed long long [::1] counts,
                       signed long long [::1] search,
                       long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_corr.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double U, V
    
    cdef double S_i, S_j
    
    cdef Py_ssize_t count

    for r in range(D):
    
        count = counts[r]
        
        S_i, S_j = 0, 0
        
        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            U = fabs(A_r[i])
            V = fabs(A_r[j])
            
            if (not (iszero(U) or iszero(V))):   
                                       
                S_i += A_r[i]/U            
                S_j += A_r[j]/V
                            
            else:
                
                count -= 1
                            
        S_corr[r] = S_i*S_j/count**2
        
cpdef void effect(double [::1] S_corr,
                  double [::1] delta,
                  double [::1] Sx,
                  double [::1] Sy,
                  double [::1] Sz,
                  double [::1] rx,
                  double [::1] ry,
                  double [::1] rz,
                  signed long long [::1] counts,
                  signed long long [::1] search,
                  long [:,::1] coordinate) nogil:
    
    cdef Py_ssize_t D = S_corr.shape[0]

    cdef Py_ssize_t r, s, i, j
    
    cdef double rx_ij, ry_ij, rz_ij, Sx_ij, Sy_ij, Sz_ij
    
    cdef double U, V, metric
    
    cdef Py_ssize_t count

    for r in range(D):
        
        count = counts[r]
        
        for s in range(search[r],search[r+1]):
            
            i, j = coordinate[s,0], coordinate[s,1]
            
            rx_ij = rx[j]-rx[i]
            ry_ij = ry[j]-ry[i]
            rz_ij = rz[j]-rz[i]
            
            Sx_ij = Sx[j]-Sx[i]
            Sy_ij = Sy[j]-Sy[i]
            Sz_ij = Sz[j]-Sz[i]
            
            U = sqrt(rx_ij*rx_ij+ry_ij*ry_ij+rz_ij*rz_ij)
            V = sqrt(Sx_ij*Sx_ij+Sy_ij*Sy_ij+Sz_ij*Sz_ij)
            
            metric = delta[i]*(1-delta[j])+delta[j]*(1-delta[i])
            
            if (not (iszero(U) or iszero(V)) and metric > 0.5):   
                                       
                S_corr[r] += (rx_ij*Sx_ij+ry_ij*Sy_ij+rz_ij*Sz_ij)/U/V
                
            else:
                
                count -= 1
                
        if (count > 0):
            
            S_corr[r] /= count
            
        else:
            
            S_corr[r] = 0
            