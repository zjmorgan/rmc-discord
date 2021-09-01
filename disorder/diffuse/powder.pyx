#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from scipy.special.cython_special cimport spherical_jn
from scipy.special import factorial

from libc.math cimport M_PI, cos, sin, exp, sqrt

from disorder.material import tables

def magnetic(double [::1] Sx, 
             double [::1] Sy, 
             double [::1] Sz, 
             double [::1] occupancy,
             double [::1] U11,
             double [::1] U22,
             double [::1] U33,
             double [::1] U23,
             double [::1] U13,
             double [::1] U12,
             double [::1] rx, 
             double [::1] ry, 
             double [::1] rz, 
             ions,
             double [::1] Q,
             double [:,:] D,
             double [::1] g,
             technique='Neutron'):
    
    cdef bint neutron = technique == 'Neutron'
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_xyz = Sx.shape[0]
    
    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    i_np, j_np = np.triu_indices(n_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
        
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    cdef double [::1] rx_ij = rx_ij_np
    cdef double [::1] ry_ij = ry_ij_np
    cdef double [::1] rz_ij = rz_ij_np
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    cdef double [::1] r_ij = r_ij_np
    
    Sx_np = np.copy(Sx, order='C')
    Sy_np = np.copy(Sy, order='C')
    Sz_np = np.copy(Sz, order='C')
    
    cdef double [::1] S_i_dot_S_i = Sx_np[m_np]*Sx_np[m_np]\
                                  + Sy_np[m_np]*Sy_np[m_np]\
                                  + Sz_np[m_np]*Sz_np[m_np]

    S_i_dot_S_j_np = Sx_np[i_np]*Sx_np[j_np]\
                   + Sy_np[i_np]*Sy_np[j_np]\
                   + Sz_np[i_np]*Sz_np[j_np]

    S_i_dot_r_ij_np = (Sx_np[i_np]*rx_ij_np[:]+
                       Sy_np[i_np]*ry_ij_np[:]+
                       Sz_np[i_np]*rz_ij_np[:])/r_ij_np[:]
    
    S_j_dot_r_ij_np = (Sx_np[j_np]*rx_ij_np[:]+\
                       Sy_np[j_np]*ry_ij_np[:]+\
                       Sz_np[j_np]*rz_ij_np[:])/r_ij_np[:]

    S_i_dot_r_ij_S_j_dot_r_ij_np = S_i_dot_r_ij_np*S_j_dot_r_ij_np

    cdef double [::1] A_ij = S_i_dot_S_j_np-S_i_dot_r_ij_S_j_dot_r_ij_np
    cdef double [::1] B_ij = 3*S_i_dot_r_ij_S_j_dot_r_ij_np-S_i_dot_S_j_np
    
    cdef double [::1] summation = np.zeros(Q.shape[0])
    
    cdef double [::1] auto = np.zeros(Q.shape[0])
    
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double Qr_ij, a_ij, b_ij
                
    cdef double [::1] Uxx = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uyy = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uzz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uyz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uxz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uxy = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] Uiso = np.zeros(n_atm, dtype=float)
        
    cdef double occ_k, occ_l, occ_n
    
    cdef double ff_k, ff_l, ff_n
    
    cdef double dw_k, dw_l, dw_n
    
    cdef double Q_sq, s_, s_sq
    
    cdef double complex f_k, f_l, f_n
    
    cdef double f_k_real, f_l_real, f_n_real
    cdef double f_k_imag, f_l_imag, f_n_imag
    
    cdef double [::1] K2 = 2/np.copy(g, order='C')-1
    
    cdef double j0_k, j0_l, j0_n
    cdef double j2_k, j2_l, j2_n
    
    cdef double [::1] A0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] B0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] C0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] D0 = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] A2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] B2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] C2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] D2 = np.zeros(n_atm, dtype=float)

    cdef double [::1] a0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c0 = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c2 = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
    
    cdef Py_ssize_t p, q, u, v, w
    
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double value = 0
        
    for p in range(n_atm):
        
        ion = ions[p]
        
        if (tables.j0.get(ion) is None):
            A0[p], a0[p], \
            B0[p], b0[p], \
            C0[p], c0[p], \
            D0[p] = 0, 0, 0, 0, 0, 0, 0
            A2[p], a2[p], \
            B2[p], b2[p], \
            C2[p], c2[p], \
            D2[p] = 0, 0, 0, 0, 0, 0, 0
        else:
            A0[p], a0[p], \
            B0[p], b0[p], \
            C0[p], c0[p], \
            D0[p] = tables.j0.get(ion)
            A2[p], a2[p], \
            B2[p], b2[p], \
            C2[p], c2[p], \
            D2[p] = tables.j2.get(ion)

        Uxx[p] = D[0,0]*D[0,0]*U11[p]+\
                 D[0,1]*D[0,1]*U22[p]+\
                 D[0,2]*D[0,2]*U33[p]+\
                 D[0,1]*D[0,2]*U23[p]*2+\
                 D[0,2]*D[0,0]*U13[p]*2+\
                 D[0,0]*D[0,1]*U12[p]*2

        Uyy[p] = D[1,0]*D[1,0]*U11[p]+\
                 D[1,1]*D[1,1]*U22[p]+\
                 D[1,2]*D[1,2]*U33[p]+\
                 D[1,1]*D[1,2]*U23[p]*2+\
                 D[1,2]*D[1,0]*U13[p]*2+\
                 D[1,0]*D[1,1]*U12[p]*2

        Uzz[p] = D[2,0]*D[2,0]*U11[p]+\
                 D[2,1]*D[2,1]*U22[p]+\
                 D[2,2]*D[2,2]*U33[p]+\
                 D[2,1]*D[2,2]*U23[p]*2+\
                 D[2,2]*D[2,0]*U13[p]*2+\
                 D[2,0]*D[2,1]*U12[p]*2

        Uyz[p] =  D[1,0]*D[2,0]*U11[p]+\
                  D[1,1]*D[2,1]*U22[p]+\
                  D[1,2]*D[2,2]*U33[p]+\
                 (D[1,1]*D[2,2]+D[1,2]*D[2,1])*U23[p]+\
                 (D[1,2]*D[2,0]+D[1,0]*D[2,2])*U13[p]+\
                 (D[1,0]*D[2,1]+D[1,1]*D[2,0])*U12[p]

        Uxz[p] =  D[0,0]*D[2,0]*U11[p]+\
                  D[0,1]*D[2,1]*U22[p]+\
                  D[0,2]*D[2,2]*U33[p]+\
                 (D[0,1]*D[2,2]+D[0,2]*D[2,1])*U23[p]+\
                 (D[0,2]*D[2,0]+D[0,0]*D[2,2])*U13[p]+\
                 (D[0,0]*D[2,1]+D[0,1]*D[2,0])*U12[p]

        Uxy[p] =  D[0,0]*D[1,0]*U11[p]+\
                  D[0,1]*D[1,1]*U22[p]+\
                  D[0,2]*D[1,2]*U33[p]+\
                 (D[0,1]*D[1,2]+D[0,2]*D[1,1])*U23[p]+\
                 (D[0,2]*D[1,0]+D[0,0]*D[1,2])*U13[p]+\
                 (D[0,0]*D[1,1]+D[0,1]*D[1,0])*U12[p]
                 
        Up, _ = np.linalg.eig(np.array([[Uxx[p], Uxy[p], Uxz[p]],
                                        [Uxy[p], Uyy[p], Uyz[p]],
                                        [Uxz[p], Uyz[p], Uzz[p]]]))

        Uiso[p] = np.mean(Up).real
    
    for q in range(n_hkl):
        
        value = 0
                
        for p in prange(n_pairs, nogil=True):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
                            
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
                                                    
            j0_k = A0[u]*exp(-a0[u]*s_sq)\
                 + B0[u]*exp(-b0[u]*s_sq)\
                 + C0[u]*exp(-c0[u]*s_sq)\
                 + D0[u]
                 
            j0_l = A0[v]*exp(-a0[v]*s_sq)\
                 + B0[v]*exp(-b0[v]*s_sq)\
                 + C0[v]*exp(-c0[v]*s_sq)\
                 + D0[v]
            
            j2_k = (A2[u]*exp(-a2[u]*s_sq)\
                 +  B2[u]*exp(-b2[u]*s_sq)\
                 +  C2[u]*exp(-c2[u]*s_sq)\
                 +  D2[u])*s_sq
                
            j2_l = (A2[v]*exp(-a2[v]*s_sq)\
                 +  B2[v]*exp(-b2[v]*s_sq)\
                 +  C2[v]*exp(-c2[v]*s_sq)\
                 +  D2[v])*s_sq
                
            ff_k = j0_k+K2[u]*j2_k
            ff_l = j0_l+K2[v]*j2_l
            
            if (ff_k < 0):
                ff_k = 0
                
            if (ff_l < 0):
                ff_l = 0
                        
            Q_sq = Q[q]*Q[q]
            
            dw_k = exp(-0.5*Q_sq*Uiso[u])
            dw_l = exp(-0.5*Q_sq*Uiso[v])
            
            f_k = occ_k*ff_k*dw_k
            f_l = occ_l*ff_l*dw_l
            
            f_k_real = f_k.real
            f_l_real = f_l.real
            
            f_k_imag = f_k.imag
            f_l_imag = f_l.imag
            
            Qr_ij = Q[q]*r_ij[p]
            
            a_ij = sin(Qr_ij)/Qr_ij
            b_ij = (a_ij-cos(Qr_ij))/(Qr_ij*Qr_ij)
            
            value += (f_k_real*f_l_real+f_k_imag*f_l_imag)\
                  *  (A_ij[p]*a_ij+B_ij[p]*b_ij)
                         
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            j0_n = A0[w]*exp(-a0[w]*s_sq)\
                 + B0[w]*exp(-b0[w]*s_sq)\
                 + C0[w]*exp(-c0[w]*s_sq)\
                 + D0[w]
                 
            j2_n = (A2[w]*exp(-a2[w]*s_sq)\
                 +  B2[w]*exp(-b2[w]*s_sq)\
                 +  C2[w]*exp(-c2[w]*s_sq)\
                 +  D2[w])*s_sq
                
            ff_n = j0_n+K2[w]*j2_n
            
            if (ff_n < 0):
                ff_n = 0
                        
            Q_sq = Q[q]*Q[q]
            
            dw_n = exp(-0.5*Q_sq*Uiso[w])
            
            f_n = occ_n*ff_n*dw_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)*S_i_dot_S_i[p]   
                 
        auto[q] = value
      
    for q in range(n_hkl):
        
        I[q] = (auto[q]+2*summation[q])/n_xyz
        
    return I_np

def occupational(double [::1] A_r, 
                 double [::1] occupancy,
                 double [::1] U11,
                 double [::1] U22,
                 double [::1] U33,
                 double [::1] U23,
                 double [::1] U13,
                 double [::1] U12,
                 double [::1] rx, 
                 double [::1] ry, 
                 double [::1] rz, 
                 atms,
                 double [::1] Q,
                 double [:,:] D,
                 technique='Neutron'):
    
    cdef bint neutron = technique == 'Neutron'
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_xyz = A_r.shape[0]
    
    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    i_np, j_np = np.triu_indices(n_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
        
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    cdef double [::1] rx_ij = rx_ij_np
    cdef double [::1] ry_ij = ry_ij_np
    cdef double [::1] rz_ij = rz_ij_np
    
    cdef double [::1] r_ij = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    A_r_np = np.copy(A_r, order='C')
    
    cdef double [::1] delta_ii = (1+A_r_np[m_np])**2
    cdef double [::1] delta_ij = (1+A_r_np[i_np])*(1+A_r_np[j_np])
    
    cdef double [::1] summation = np.zeros(Q.shape[0])
    
    cdef double [::1] auto = np.zeros(Q.shape[0])
    
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double Qr_ij, a_ij
                
    cdef double [::1] Uxx = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uyy = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uzz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uyz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uxz = np.zeros(n_atm, dtype=float)
    cdef double [::1] Uxy = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] Uiso = np.zeros(n_atm, dtype=float)
        
    cdef double occ_k, occ_l, occ_n
    
    cdef double complex sl_k, sl_l, sl_n
    
    cdef double dw_k, dw_l, dw_n
    
    cdef double Q_sq, s_, s_sq
    
    cdef double complex f_k, f_l, f_n
    
    cdef double f_k_real, f_l_real, f_n_real
    cdef double f_k_imag, f_l_imag, f_n_imag
    
    cdef double complex [::1] b = np.zeros(n_atm, dtype=complex)
    
    cdef double [::1] a1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c  = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
    
    cdef Py_ssize_t p, q, u, v, w
    
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double value = 0
        
    for p in range(n_atm):
        
        atm = atms[p]
        
        if neutron:
            b[p] = tables.bc.get(atm)
        else:
            a1[p], b1[p], \
            a2[p], b2[p], \
            a3[p], b3[p], \
            a4[p], b4[p], \
            c[p] = tables.X.get(atm)

        Uxx[p] = D[0,0]*D[0,0]*U11[p]+\
                 D[0,1]*D[0,1]*U22[p]+\
                 D[0,2]*D[0,2]*U33[p]+\
                 D[0,1]*D[0,2]*U23[p]*2+\
                 D[0,2]*D[0,0]*U13[p]*2+\
                 D[0,0]*D[0,1]*U12[p]*2

        Uyy[p] = D[1,0]*D[1,0]*U11[p]+\
                 D[1,1]*D[1,1]*U22[p]+\
                 D[1,2]*D[1,2]*U33[p]+\
                 D[1,1]*D[1,2]*U23[p]*2+\
                 D[1,2]*D[1,0]*U13[p]*2+\
                 D[1,0]*D[1,1]*U12[p]*2

        Uzz[p] = D[2,0]*D[2,0]*U11[p]+\
                 D[2,1]*D[2,1]*U22[p]+\
                 D[2,2]*D[2,2]*U33[p]+\
                 D[2,1]*D[2,2]*U23[p]*2+\
                 D[2,2]*D[2,0]*U13[p]*2+\
                 D[2,0]*D[2,1]*U12[p]*2

        Uyz[p] =  D[1,0]*D[2,0]*U11[p]+\
                  D[1,1]*D[2,1]*U22[p]+\
                  D[1,2]*D[2,2]*U33[p]+\
                 (D[1,1]*D[2,2]+D[1,2]*D[2,1])*U23[p]+\
                 (D[1,2]*D[2,0]+D[1,0]*D[2,2])*U13[p]+\
                 (D[1,0]*D[2,1]+D[1,1]*D[2,0])*U12[p]

        Uxz[p] =  D[0,0]*D[2,0]*U11[p]+\
                  D[0,1]*D[2,1]*U22[p]+\
                  D[0,2]*D[2,2]*U33[p]+\
                 (D[0,1]*D[2,2]+D[0,2]*D[2,1])*U23[p]+\
                 (D[0,2]*D[2,0]+D[0,0]*D[2,2])*U13[p]+\
                 (D[0,0]*D[2,1]+D[0,1]*D[2,0])*U12[p]

        Uxy[p] =  D[0,0]*D[1,0]*U11[p]+\
                  D[0,1]*D[1,1]*U22[p]+\
                  D[0,2]*D[1,2]*U33[p]+\
                 (D[0,1]*D[1,2]+D[0,2]*D[1,1])*U23[p]+\
                 (D[0,2]*D[1,0]+D[0,0]*D[1,2])*U13[p]+\
                 (D[0,0]*D[1,1]+D[0,1]*D[1,0])*U12[p]
                 
        Up, _ = np.linalg.eig(np.array([[Uxx[p], Uxy[p], Uxz[p]],
                                        [Uxy[p], Uyy[p], Uyz[p]],
                                        [Uxz[p], Uyz[p], Uzz[p]]]))

        Uiso[p] = np.mean(Up).real
    
    for q in range(n_hkl):
        
        value = 0
                
        for p in prange(n_pairs, nogil=True):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
            
            if neutron:
                
                sl_k = b[u]
                sl_l = b[v]
                
            else:
                            
                s_ = Q[q]*inv_M_SP
                s_sq = s_*s_
                                        
                sl_k = a1[u]*exp(-b1[u]*s_sq)\
                     + a2[u]*exp(-b2[u]*s_sq)\
                     + a3[u]*exp(-b3[u]*s_sq)\
                     + a4[u]*exp(-b4[u]*s_sq)\
                     + c[u]
                     
                sl_l = a1[v]*exp(-b1[v]*s_sq)\
                     + a2[v]*exp(-b2[v]*s_sq)\
                     + a3[v]*exp(-b3[v]*s_sq)\
                     + a4[v]*exp(-b4[v]*s_sq)\
                     + c[v]
                        
            Q_sq = Q[q]*Q[q]
            
            dw_k = exp(-0.5*Q_sq*Uiso[u])
            dw_l = exp(-0.5*Q_sq*Uiso[v])
            
            f_k = occ_k*sl_k*dw_k
            f_l = occ_l*sl_l*dw_l
            
            f_k_real = f_k.real
            f_l_real = f_l.real
            
            f_k_imag = f_k.imag
            f_l_imag = f_l.imag
            
            Qr_ij = Q[q]*r_ij[p]
            
            a_ij = sin(Qr_ij)/Qr_ij
            
            value += (f_k_real*f_l_real+f_k_imag*f_l_imag)*delta_ij[p]*a_ij
                         
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            if (technique == 'Neutron'):
                
                sl_n = b[w]
                
            else:
                            
                s_ = Q[q]*inv_M_SP
                s_sq = s_*s_
                                        
                sl_n = a1[w]*exp(-b1[w]*s_sq)\
                     + a2[w]*exp(-b2[w]*s_sq)\
                     + a3[w]*exp(-b3[w]*s_sq)\
                     + a4[w]*exp(-b4[w]*s_sq)\
                     + c[w]
                        
            Q_sq = Q[q]*Q[q]
            
            dw_n = exp(-0.5*Q_sq*Uiso[w])
            
            f_n = occ_n*sl_n*dw_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)*delta_ii[p]
                    
        auto[q] = value
      
    for q in range(n_hkl):
        
        I[q] = (auto[q]+2*summation[q])/n_xyz
        
    return I_np
        
def displacive(double [::1] Ux, 
               double [::1] Uy, 
               double [::1] Uz, 
               double [::1] occupancy,
               double [::1] rx, 
               double [::1] ry, 
               double [::1] rz, 
               atms,
               double [::1] Q,
               double [:,:] D,
               int order,
               technique='Neutron'):

    cdef bint neutron = technique == 'Neutron'
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_xyz = Ux.shape[0]
    
    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    i_np, j_np = np.triu_indices(n_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
        
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    cdef double [::1] rx_ij = rx_ij_np
    cdef double [::1] ry_ij = ry_ij_np
    cdef double [::1] rz_ij = rz_ij_np
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    cdef double [::1] r_ij = r_ij_np
    
    Ux_np = np.copy(Ux, order='C')
    Uy_np = np.copy(Uy, order='C')
    Uz_np = np.copy(Uz, order='C')
    
    cdef double [::1] summation = np.zeros(Q.shape[0])
    
    cdef double [::1] auto = np.zeros(Q.shape[0])
    
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double Qr_ij, a_ij, s_ij, c_ij
    
    cdef double [::1] A_ij = np.zeros(order+1)
    
    cdef double Qu_ij, jn
    
    cdef double Qu_ij_pow = 1
    
    Ux_ij_np = Ux_np[j_np]-Ux_np[i_np]
    Uy_ij_np = Uy_np[j_np]-Uy_np[i_np]
    Uz_ij_np = Uz_np[j_np]-Uz_np[i_np]
    
    u_ij_np = np.sqrt(Ux_ij_np**2+Uy_ij_np**2+Uz_ij_np**2)
    
    cdef double [::1] u_ij = u_ij_np
    
    u_ij_mul_r_ij = u_ij_np*r_ij_np
    
    u_ij_mul_r_ij[np.isclose(u_ij_mul_r_ij, 0)] = 1
    
    u_hat_ij_dot_r_hat_ij_np = (Ux_ij_np*rx_ij_np
                             +  Uy_ij_np*ry_ij_np\
                             +  Uz_ij_np*rz_ij_np)/u_ij_mul_r_ij
                
    cdef double [:,::1] u_hat_ij_dot_r_hat_ij_pow = \
                        u_hat_ij_dot_r_hat_ij_np**np.arange(order+1)[:,None]
        
    cdef double occ_k, occ_l, occ_n
    
    cdef double complex sl_k, sl_l, sl_n
        
    cdef double Q_sq, s_, s_sq
    
    cdef double complex f_k, f_l, f_n
    
    cdef double f_k_real, f_l_real, f_n_real
    cdef double f_k_imag, f_l_imag, f_n_imag
    
    cdef double complex [::1] b = np.zeros(n_atm, dtype=complex)
    
    cdef double [::1] a1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c  = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
    
    cdef Py_ssize_t p, q, u, v, w
    
    cdef Py_ssize_t r, s, t
    
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double value, values
    
    cs = np.ones(order+1)
    seq_id = np.array([])
    
    for t in range(order//2+1):
        seq_id = np.concatenate((seq_id, cs))
        cs = np.cumsum(cs[1:order-t])
        
    num = (seq_id+1)*seq_id/2
    
    den = np.array([])
    
    for t in range(order//2+1):
        den = np.concatenate((den, (-1)**t*factorial(np.arange(2*t, order+1))))
        
    cdef double [::1] coeff = num/den
            
    for p in range(n_atm):
        
        atm = atms[p]
        
        if neutron:
            b[p] = tables.bc.get(atm)
        else:
            a1[p], b1[p], \
            a2[p], b2[p], \
            a3[p], b3[p], \
            a4[p], b4[p], \
            c[p] = tables.X.get(atm)
    
    for q in range(n_hkl):
        
        value = 0
                
        for p in prange(n_pairs, nogil=True):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
            
            if neutron:
                
                sl_k = b[u]
                sl_l = b[v]
                
            else:
                            
                s_ = Q[q]*inv_M_SP
                s_sq = s_*s_
                                        
                sl_k = a1[u]*exp(-b1[u]*s_sq)\
                     + a2[u]*exp(-b2[u]*s_sq)\
                     + a3[u]*exp(-b3[u]*s_sq)\
                     + a4[u]*exp(-b4[u]*s_sq)\
                     + c[u]
                     
                sl_l = a1[v]*exp(-b1[v]*s_sq)\
                     + a2[v]*exp(-b2[v]*s_sq)\
                     + a3[v]*exp(-b3[v]*s_sq)\
                     + a4[v]*exp(-b4[v]*s_sq)\
                     + c[v]
                        
            Q_sq = Q[q]*Q[q]
            
            f_k = occ_k*sl_k
            f_l = occ_l*sl_l
            
            f_k_real = f_k.real
            f_l_real = f_l.real
            
            f_k_imag = f_k.imag
            f_l_imag = f_l.imag
            
            Qr_ij = Q[q]*r_ij[p]
            Qu_ij = Q[q]*u_ij[p]
            
            Qu_ij_pow = 1
                        
            for r in range(order+1):
                A_ij[r] = 0
                t = r
                for s in range(r // 2+1):
                    A_ij[r-s] += coeff[t]*Qu_ij_pow\
                              *  u_hat_ij_dot_r_hat_ij_pow[r-s,p]
                    t += order-2*r
                Qu_ij_pow *= Qu_ij
            
            values = 0
            for r in range(order+1):
                values += A_ij[r]*spherical_jn(r, Qr_ij)
            
            value += (f_k_real*f_l_real+f_k_imag*f_l_imag)*values
                         
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            if (technique == 'Neutron'):
                
                sl_n = b[w]
                
            else:
                            
                s_ = Q[q]*inv_M_SP
                s_sq = s_*s_
                                        
                sl_n = a1[w]*exp(-b1[w]*s_sq)\
                     + a2[w]*exp(-b2[w]*s_sq)\
                     + a3[w]*exp(-b3[w]*s_sq)\
                     + a4[w]*exp(-b4[w]*s_sq)\
                     + c[w]
                                                
            f_n = occ_n*sl_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)
                    
        auto[q] = value
      
    for q in range(n_hkl):
        
        I[q] = (auto[q]+2*summation[q])/n_xyz
        
    return I_np