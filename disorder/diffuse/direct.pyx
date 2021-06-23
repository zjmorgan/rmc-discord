#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython
    
cdef extern from "math.h" nogil:
    double cos(double x)
    
cdef extern from "math.h" nogil:
    double sin(double x) 
    
cdef extern from "math.h" nogil:
    double sqrt(double x) 
    
def magnetic(double [::1] Qx, 
             double [::1] Qy, 
             double [::1] Qz, 
             double [::1] Qx_norm, 
             double [::1] Qy_norm, 
             double [::1] Qz_norm, 
             double [::1] Sx, 
             double [::1] Sy, 
             double [::1] Sz, 
             double [::1] rx, 
             double [::1] ry, 
             double [::1] rz, 
             double [::1] form_factor):
    
    cdef Py_ssize_t n_hkl = Qx.shape[0]
    cdef Py_ssize_t n_xyz = rx.shape[0]
    
    cdef Py_ssize_t n_atm = form_factor.shape[0] // n_hkl
    cdef Py_ssize_t n_uvw = n_xyz // n_atm
    
    i_np, j_np = np.triu_indices(n_xyz,1)
    
    cdef double rx_ij, ry_ij, rz_ij
    
    I_np = np.zeros(n_hkl, dtype=np.double)

    cdef double [::1] I = I_np
    
    cdef long [::] i = i_np
    cdef long [::] j = j_np
    
    k_np, l_np = np.mod(i_np,n_atm), np.mod(j_np,n_atm)

    cdef long [::] k = k_np
    cdef long [::] l = l_np
    
    m_np = np.mod(np.arange(n_xyz),n_atm)

    cdef long [::] m = m_np
        
    cdef double Sx_perp_i, Sy_perp_i, Sz_perp_i, Q_norm_dot_S_i
    cdef double Sx_perp_j, Sy_perp_j, Sz_perp_j, Q_norm_dot_S_j
    
    cdef double summation, auto, phase_factor, S_perp_i_dot_S_perp_j, factors
    
    cdef Py_ssize_t i_hkl, i_xyz
    
    for i_hkl in range(n_hkl): 
        
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
            
            Q_norm_dot_S_i = Qx_norm[i_hkl]*Sx[i[i_xyz]]\
                           + Qy_norm[i_hkl]*Sy[i[i_xyz]]\
                           + Qz_norm[i_hkl]*Sz[i[i_xyz]]
        
            Sx_perp_i = Sx[i[i_xyz]]-Q_norm_dot_S_i*Qx_norm[i_hkl]
            Sy_perp_i = Sy[i[i_xyz]]-Q_norm_dot_S_i*Qy_norm[i_hkl]
            Sz_perp_i = Sz[i[i_xyz]]-Q_norm_dot_S_i*Qz_norm[i_hkl]
        
            Q_norm_dot_S_j = Qx_norm[i_hkl]*Sx[j[i_xyz]]\
                           + Qy_norm[i_hkl]*Sy[j[i_xyz]]\
                           + Qz_norm[i_hkl]*Sz[j[i_xyz]]
                           
            Sx_perp_j = Sx[j[i_xyz]]-Q_norm_dot_S_j*Qx_norm[i_hkl]
            Sy_perp_j = Sy[j[i_xyz]]-Q_norm_dot_S_j*Qy_norm[i_hkl]
            Sz_perp_j = Sz[j[i_xyz]]-Q_norm_dot_S_j*Qz_norm[i_hkl]
                    
            S_perp_i_dot_S_perp_j = Sx_perp_i*Sx_perp_j\
                                  + Sy_perp_i*Sy_perp_j\
                                  + Sz_perp_i*Sz_perp_j\
                                  
            factors = form_factor[k[i_xyz]+n_atm*i_hkl]\
                    * form_factor[l[i_xyz]+n_atm*i_hkl]
            
            summation += 2*factors*S_perp_i_dot_S_perp_j*phase_factor
            
        auto = 0
        for i_xyz in prange(n_xyz, nogil=True):    
            
            Q_norm_dot_S_i = Qx_norm[i_hkl]*Sx[i_xyz]\
                           + Qy_norm[i_hkl]*Sy[i_xyz]\
                           + Qz_norm[i_hkl]*Sz[i_xyz]
        
            Sx_perp_i = Sx[i_xyz]-Q_norm_dot_S_i*Qx_norm[i_hkl]
            Sy_perp_i = Sy[i_xyz]-Q_norm_dot_S_i*Qy_norm[i_hkl]
            Sz_perp_i = Sz[i_xyz]-Q_norm_dot_S_i*Qz_norm[i_hkl]
                    
            S_perp_i_dot_S_perp_j = Sx_perp_i*Sx_perp_i\
                                  + Sy_perp_i*Sy_perp_i\
                                  + Sz_perp_i*Sz_perp_i\
                                  
            factors = form_factor[m[i_xyz]+n_atm*i_hkl]**2
            
            auto += factors*S_perp_i_dot_S_perp_j
            
        I[i_hkl] = (auto+summation) // n_xyz
    
    return I_np   

def occupational(double [::1] Qx, 
                 double [::1] Qy, 
                 double [::1] Qz, 
                 double [::1] A, 
                 double [::1] rx, 
                 double [::1] ry, 
                 double [::1] rz, 
                 double complex [::1] scattering_length):
    
    cdef Py_ssize_t n_hkl = Qx.shape[0]
    cdef Py_ssize_t n_xyz = rx.shape[0]
    
    cdef Py_ssize_t n_atm = scattering_length.shape[0] // n_hkl
    cdef Py_ssize_t n_uvw = n_xyz // n_atm
    
    i_np, j_np = np.triu_indices(n_xyz,1)
    
    cdef double rx_ij, ry_ij, rz_ij
    
    I_np = np.zeros(n_hkl, dtype=np.double)

    cdef double [::1] I = I_np
    
    cdef long [::] i = i_np
    cdef long [::] j = j_np
    
    k_np, l_np = np.mod(i_np,n_atm), np.mod(j_np,n_atm)

    cdef long [::] k = k_np
    cdef long [::] l = l_np
    
    m_np = np.mod(np.arange(n_xyz),n_atm)

    cdef long [::] m = m_np
    
    cdef double complex b_k, b_l, b_m

    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double A_i, A_j, A_i_A_j

    cdef double summation, auto, phase_factor, factors
    
    cdef Py_ssize_t i_hkl, i_xyz
    
    for i_hkl in range(n_hkl): 
        
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
                    
            A_i = A[i[i_xyz]]
            A_j = A[j[i_xyz]]
            
            A_i_A_j = A_i*A_j
            
            b_k = scattering_length[k[i_xyz]+n_atm*i_hkl]
            b_l = scattering_length[l[i_xyz]+n_atm*i_hkl]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
                                  
            factors = sqrt((b_k_real*b_k_real+b_k_imag*b_k_imag)\
                    *      (b_l_real*b_l_real+b_l_imag*b_l_imag))
            
            summation += 2*factors*A_i_A_j*phase_factor
          
        auto = 0
        for i_xyz in prange(n_xyz, nogil=True):
                    
            A_i = A[i_xyz]
            
            A_i_A_j = A_i*A_i
            
            b_m = scattering_length[m[i_xyz]+n_atm*i_hkl]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                                  
            factors = (b_m_real*b_m_real+b_m_imag*b_m_imag)
            
            auto += factors*A_i_A_j
            
        I[i_hkl] = (auto+summation) // n_xyz
    
    return I_np

def displacive(double [::1] Qx, 
               double [::1] Qy, 
               double [::1] Qz, 
               double [::1] Ux, 
               double [::1] Uy, 
               double [::1] Uz, 
               double [::1] rx, 
               double [::1] ry, 
               double [::1] rz, 
               double complex [::1] scattering_length,
               long [::1] bragg):
    
    cdef Py_ssize_t n_hkl = Qx.shape[0]
    cdef Py_ssize_t n_xyz = rx.shape[0]
    
    cdef Py_ssize_t n_atm = scattering_length.shape[0] // n_hkl
    cdef Py_ssize_t n_uvw = n_xyz // n_atm
    
    cdef Py_ssize_t n_nuc = bragg.shape[0]
    
    i_np, j_np = np.triu_indices(n_xyz,1)
    
    cdef double rx_ij, ry_ij, rz_ij
    
    I_np = np.zeros(n_hkl, dtype=np.double)

    cdef double [::1] I = I_np
    
    cdef long [::] i = i_np
    cdef long [::] j = j_np
    
    k_np, l_np = np.mod(i_np,n_atm), np.mod(j_np,n_atm)

    cdef long [::] k = k_np
    cdef long [::] l = l_np
    
    m_np = np.mod(np.arange(n_xyz),n_atm)

    cdef long [::] m = m_np
    
    cdef double complex b_k, b_l, b_m

    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double summation, auto, phase_factor, factors
    
    cdef Py_ssize_t i_hkl, i_xyz, i_nuc
    
    cdef double Q_dot_r_i, Q_dot_r_j
    cdef double Q_dot_U_i, Q_dot_U_j
    
    for i_hkl in range(n_hkl): 
        
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]+Ux[i[i_xyz]]-Ux[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]+Uy[i[i_xyz]]-Uy[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]+Uz[i[i_xyz]]-Uz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
                                
            b_k = scattering_length[k[i_xyz]+n_atm*i_hkl]
            b_l = scattering_length[l[i_xyz]+n_atm*i_hkl]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
                                  
            factors = b_k_real*b_l_real+b_k_imag*b_l_imag
            
            summation += 2*factors*phase_factor
          
        auto = 0   
        for i_xyz in prange(n_xyz, nogil=True):
                                
            b_m = scattering_length[m[i_xyz]+n_atm*i_hkl]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                                  
            factors = b_m_real*b_m_real+b_m_imag*b_m_imag
            
            auto += factors
            
        I[i_hkl] = (auto+summation) // n_xyz
        
    for i_nuc in range(n_nuc): 
        
        i_hkl = bragg[i_nuc]
            
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]+Ux[i[i_xyz]]-Ux[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]+Uy[i[i_xyz]]-Uy[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]+Uz[i[i_xyz]]-Uz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
                                
            b_k = scattering_length[k[i_xyz]+n_atm*i_hkl]
            b_l = scattering_length[l[i_xyz]+n_atm*i_hkl]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
                                  
            factors = b_k_real*b_l_real+b_k_imag*b_l_imag

            summation += 2*factors*phase_factor
          
        auto = 0   
        for i_xyz in prange(n_xyz, nogil=True):
                                
            b_m = scattering_length[m[i_xyz]+n_atm*i_hkl]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                                  
            factors = b_m_real*b_m_real+b_m_imag*b_m_imag
            
            auto += factors
            
        I[i_hkl] -= (auto+summation) // n_xyz
    
    return I_np

def nonmagnetic(double [::1] Qx, 
                double [::1] Qy, 
                double [::1] Qz, 
                double [::1] Ux, 
                double [::1] Uy, 
                double [::1] Uz, 
                double [::1] A,
                double [::1] rx, 
                double [::1] ry, 
                double [::1] rz, 
                double complex [::1] scattering_length,
                long [::1] bragg):
    
    cdef Py_ssize_t n_hkl = Qx.shape[0]
    cdef Py_ssize_t n_xyz = rx.shape[0]
    
    cdef Py_ssize_t n_atm = scattering_length.shape[0] // n_hkl
    cdef Py_ssize_t n_uvw = n_xyz // n_atm
    
    cdef Py_ssize_t n_nuc = bragg.shape[0]
    
    i_np, j_np = np.triu_indices(n_xyz,1)
    
    cdef double rx_ij, ry_ij, rz_ij
    
    I_np = np.zeros(n_hkl, dtype=np.double)

    cdef double [::1] I = I_np
    
    cdef long [::] i = i_np
    cdef long [::] j = j_np
    
    k_np, l_np = np.mod(i_np,n_atm), np.mod(j_np,n_atm)

    cdef long [::] k = k_np
    cdef long [::] l = l_np
    
    m_np = np.mod(np.arange(n_xyz),n_atm)

    cdef long [::] m = m_np
    
    cdef double complex b_k, b_l, b_m

    cdef double b_k_real, b_k_imag, b_l_real, b_l_imag, b_m_real, b_m_imag
    
    cdef double A_i, A_j

    cdef double summation, auto, phase_factor, factors
    
    cdef Py_ssize_t i_hkl, i_xyz, i_nuc

    for i_hkl in range(n_hkl): 
            
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]+Ux[i[i_xyz]]-Ux[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]+Uy[i[i_xyz]]-Uy[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]+Uz[i[i_xyz]]-Uz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
            
            A_i = A[i[i_xyz]]
            A_j = A[j[i_xyz]]
                                
            b_k = scattering_length[k[i_xyz]+n_atm*i_hkl]
            b_l = scattering_length[l[i_xyz]+n_atm*i_hkl]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
                                  
            factors = b_k_real*b_l_real+b_k_imag*b_l_imag
            
            summation += 2*factors*(1+A_i)*(1+A_j)*phase_factor
          
        auto = 0   
        for i_xyz in prange(n_xyz, nogil=True):
            
            A_i = A[i_xyz]
                                
            b_m = scattering_length[m[i_xyz]+n_atm*i_hkl]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                                  
            factors = b_m_real*b_m_real+b_m_imag*b_m_imag
            
            auto += factors*(1+A_i)**2
            
        I[i_hkl] = (auto+summation) // n_xyz
        
    for i_nuc in range(n_nuc): 
        
        i_hkl = bragg[i_nuc]
            
        summation = 0
        for i_xyz in prange(n_xyz*(n_xyz-1) // 2, nogil=True):    
            
            rx_ij = rx[i[i_xyz]]-rx[j[i_xyz]]+Ux[i[i_xyz]]-Ux[j[i_xyz]]
            ry_ij = ry[i[i_xyz]]-ry[j[i_xyz]]+Uy[i[i_xyz]]-Uy[j[i_xyz]]
            rz_ij = rz[i[i_xyz]]-rz[j[i_xyz]]+Uz[i[i_xyz]]-Uz[j[i_xyz]]
            
            phase_factor = cos(Qx[i_hkl]*rx_ij+\
                               Qy[i_hkl]*ry_ij+\
                               Qz[i_hkl]*rz_ij)
            
            A_i = A[i[i_xyz]]
            A_j = A[j[i_xyz]]
                                
            b_k = scattering_length[k[i_xyz]+n_atm*i_hkl]
            b_l = scattering_length[l[i_xyz]+n_atm*i_hkl]
            
            b_k_real = b_k.real
            b_l_real = b_l.real

            b_k_imag = b_k.imag
            b_l_imag = b_l.imag
                                  
            factors = b_k_real*b_l_real+b_k_imag*b_l_imag

            summation += 2*factors*(1+A_i)*(1+A_j)*phase_factor
          
        auto = 0   
        for i_xyz in prange(n_xyz, nogil=True):
            
            A_i = A[i_xyz]
                    
            b_m = scattering_length[m[i_xyz]+n_atm*i_hkl]
            
            b_m_real = b_m.real
            b_m_imag = b_m.imag
                                  
            factors = b_m_real*b_m_real+b_m_imag*b_m_imag
            
            auto += factors*(1+A_i)**2
            
        I[i_hkl] -= (auto+summation) // n_xyz
    
    return I_np