#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport rand
    
cpdef (double, Py_ssize_t) scalar(double [::1] A) nogil:
        
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i = rand() % n
                    
    return A[i], i

cpdef (double, double, double, Py_ssize_t) vector(double [::1] A,
                                                  double [::1] B,
                                                  double [::1] C) nogil:
        
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i = rand() % n
                    
    return A[i], B[i], C[i], i
    
cpdef Py_ssize_t scalars(double [::1] B, 
                         long [::1] i,
                         double [::1] A, 
                         long [:,::1] structure) nogil:
        
    cdef Py_ssize_t m = structure.shape[0]
    cdef Py_ssize_t n = structure.shape[1]
    
    cdef Py_ssize_t k = rand() % m
    
    i = structure[k,:]
        
    cdef Py_ssize_t j
    
    for j in range(n):
        B[j] = A[i[j]]
    
    return k                
        
cpdef Py_ssize_t vectors(double [::1] D, 
                         double [::1] E, 
                         double [::1] F,
                         long [::1] i,
                         double [::1] A, 
                         double [::1] B, 
                         double [::1] C, 
                         long [:,::1] structure) nogil:
        
    cdef Py_ssize_t m = structure.shape[0]
    cdef Py_ssize_t n = structure.shape[1]

    cdef Py_ssize_t k = rand() % m
    
    i = structure[k,:]
    
    cdef Py_ssize_t j
    
    for j in range(n):
        D[j] = A[i[j]]
        E[j] = B[i[j]]
        F[j] = C[i[j]]
    
    return k     