#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport M_PI
    
cdef extern from "math.h" nogil:
    double cos(double x)
    
cdef extern from "math.h" nogil:
    double sin(double x) 
    
cdef extern from "math.h" nogil:
    double sqrt(double x) 
        
cpdef void magnetic(double [::1] summation, 
                    double [::1] auto,
                    double [::1] Q,
                    double [::1] r_ij,
                    double [::1] form_factor,
                    double [::1] A_ij,
                    double [::1] B_ij,
                    double [::1] S_i_dot_S_i,
                    long [::1] k,
                    long [::1] l,
                    long [::1] m,
                    Py_ssize_t n_xyz,
                    Py_ssize_t n_atm) nogil:
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double Qr_ij, a_ij, b_ij, value
        
    cdef Py_ssize_t i, j
    
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_pairs):
            
            Qr_ij = Q[i]*r_ij[j]
            
            a_ij = sin(Qr_ij)/Qr_ij
            b_ij = (a_ij-cos(Qr_ij))/(Qr_ij*Qr_ij)
            
            value += form_factor[k[j]+n_atm*i]\
                  *  form_factor[l[j]+n_atm*i]\
                  *  (A_ij[j]*a_ij+B_ij[j]*b_ij)
                         
        summation[i] = value
                         
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_xyz):
            
            value += form_factor[m[j]+n_atm*i]\
                  *  form_factor[m[j]+n_atm*i]\
                  *  S_i_dot_S_i[j]
                    
        auto[i] = value

cpdef void occupational(double [::1] summation, 
                        double [::1] auto,
                        double [::1] Q,
                        double [::1] r_ij,
                        double complex [::1] scattering_length,
                        double [::1] delta_ij,
                        double [::1] delta_i_delta_i,
                        long [::1] k,
                        long [::1] l,
                        long [::1] m,
                        Py_ssize_t n_xyz,
                        Py_ssize_t n_atm) nogil:
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double complex b_k, b_l, b_m
    
    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double Qr_ij, a_ij, value
        
    cdef Py_ssize_t i, j
    
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_pairs):
            
            Qr_ij = Q[i]*r_ij[j]
            
            a_ij = sin(Qr_ij)/Qr_ij
            
            b_k = scattering_length[k[j]+n_atm*i]
            b_l = scattering_length[l[j]+n_atm*i]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
            
            value += (b_k_real*b_l_real+b_k_imag*b_l_imag)\
                  *  delta_ij[j]*a_ij
                         
        summation[i] = value
                         
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_xyz):
            
            b_m = scattering_length[m[j]+n_atm*i]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
            
            value += (b_m_real*b_m_real+b_m_imag*b_m_imag)\
                  *  delta_i_delta_i[j]
                    
        auto[i] = value
        
cpdef void displacive(double [::1] summation, 
                      double [::1] auto,
                      double [::1] Q,
                      double [::1] Ux,
                      double [::1] Uy,
                      double [::1] Uz,
                      double [::1] rx,
                      double [::1] ry,
                      double [::1] rz,
                      double complex [::1] scattering_length,
                      long [::1] p,
                      long [::1] q,
                      long [::1] k,
                      long [::1] l,
                      long [::1] m,
                      Py_ssize_t n_xyz,
                      Py_ssize_t n_atm) nogil:
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_pairs = k.shape[0]
    
    cdef double complex b_k, b_l, b_m
    
    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double r_ij, rx_ij, ry_ij, rz_ij
    
    cdef double Qr_ij, a_ij, value
        
    cdef Py_ssize_t i, j
    
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_pairs):
            
            rx_ij = rx[q[j]]-rx[p[j]]+Ux[q[j]]-Ux[p[j]]
            ry_ij = ry[q[j]]-ry[p[j]]+Uy[q[j]]-Uy[p[j]]
            rz_ij = rz[q[j]]-rz[p[j]]+Uz[q[j]]-Uz[p[j]]
            
            r_ij = sqrt(rx_ij*rx_ij+ry_ij*ry_ij+rz_ij*rz_ij)
            
            Qr_ij = Q[i]*r_ij
            
            a_ij = sin(Qr_ij)/Qr_ij
            
            b_k = scattering_length[k[j]+n_atm*i]
            b_l = scattering_length[l[j]+n_atm*i]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
            
            value += (b_k_real*b_l_real+b_k_imag*b_l_imag)*a_ij
                         
        summation[i] = value
                         
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_xyz):
            
            b_m = scattering_length[m[j]+n_atm*i]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                        
            value += b_m_real*b_m_real+b_m_imag*b_m_imag
                    
        auto[i] = value

cpdef void average(double [::1] summation, 
                   double [::1] auto,
                   double [::1] Q,
                   double [::1] r_ij,
                   double complex [::1] scattering_length,
                   long [::1] k,
                   long [::1] l,
                   long [::1] m,
                   Py_ssize_t n_xyz,
                   Py_ssize_t n_atm) nogil:
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double complex b_k, b_l, b_m
    
    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double Qr_ij, a_ij, value
        
    cdef Py_ssize_t i, j
    
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_pairs):
            
            Qr_ij = Q[i]*r_ij[j]
            
            a_ij = sin(Qr_ij)/Qr_ij
            
            b_k = scattering_length[k[j]+n_atm*i]
            b_l = scattering_length[l[j]+n_atm*i]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
            
            value += (b_k_real*b_l_real+b_k_imag*b_l_imag)*a_ij
                         
        summation[i] = value
                         
    for i in range(n_hkl):
        
        value = 0
        
        for j in prange(n_xyz):
            
            b_m = scattering_length[m[j]+n_atm*i]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
            
            value += b_m_real*b_m_real+b_m_imag*b_m_imag
                    
        auto[i] = value