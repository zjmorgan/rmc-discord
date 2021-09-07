#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from scipy.special import factorial

from libc.math cimport M_PI, cos, sin, exp, sqrt

from disorder.material import tables

import os

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
             double [:,:] A,
             Py_ssize_t nu,
             Py_ssize_t nv,
             Py_ssize_t nw,
             double [::1] g):
        
    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_uvw = nu*nv*nw
    
    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
    
    rx_np = rx_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    ry_np = ry_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    rz_np = rz_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    
    i_np, j_np = np.triu_indices(m_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    Sx_np = np.copy(Sx, order='C')
    Sy_np = np.copy(Sy, order='C')
    Sz_np = np.copy(Sz, order='C')
        
    iu, iv, iw, i_atm = np.unravel_index(i_np, (mu,mv,mw,n_atm))
    ju, jv, jw, j_atm = np.unravel_index(j_np, (mu,mv,mw,n_atm))
        
    cu, cv, cw = np.unravel_index(np.arange(n_uvw), (nu,nv,nw))
    
    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cu[:,None], nw)
    
    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cu[:,None], nw)
    
    i_np = np.ascontiguousarray((i_atm+n_atm*(iw+nw*(iv+nv*iu))).T)
    j_np = np.ascontiguousarray((j_atm+n_atm*(jw+nw*(jv+nv*ju))).T)
    
    cdef double [::1] S_i_dot_S_i = Sx_np[m_np]*Sx_np[m_np]\
                                  + Sy_np[m_np]*Sy_np[m_np]\
                                  + Sz_np[m_np]*Sz_np[m_np]
    
    S_i_dot_S_j_np = Sx_np[i_np]*Sx_np[j_np]\
                   + Sy_np[i_np]*Sy_np[j_np]\
                   + Sz_np[i_np]*Sz_np[j_np]
    
    S_i_dot_r_ij_np = ((Sx_np[i_np].T*rx_ij_np[:]+\
                        Sy_np[i_np].T*ry_ij_np[:]+\
                        Sz_np[i_np].T*rz_ij_np[:])/r_ij_np[:]).T
    
    S_j_dot_r_ij_np = ((Sx_np[j_np].T*rx_ij_np[:]+\
                        Sy_np[j_np].T*ry_ij_np[:]+\
                        Sz_np[j_np].T*rz_ij_np[:])/r_ij_np[:]).T

    S_i_dot_r_ij_S_j_dot_r_ij_np = S_i_dot_r_ij_np*S_j_dot_r_ij_np

    cdef double [:,::1] A_ij = S_i_dot_S_j_np-S_i_dot_r_ij_S_j_dot_r_ij_np
    cdef double [:,::1] B_ij = 3*S_i_dot_r_ij_S_j_dot_r_ij_np-S_i_dot_S_j_np
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    cdef double [::1] r_ij = r_ij_np
    
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
    
    cdef Py_ssize_t p, q, r, u, v, w
    
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double factors, value = 0
        
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

        U = np.array([[U11[p], U12[p], U13[p]],
                      [U12[p], U22[p], U23[p]],
                      [U13[p], U23[p], U33[p]]])
        
        Up = np.dot(np.dot(D, U), D.T)
        
        Uxx[p] = Up[0,0]
        Uyy[p] = Up[1,1]
        Uzz[p] = Up[2,2]
        Uyz[p] = Up[1,2]
        Uxz[p] = Up[0,2]
        Uxy[p] = Up[0,1]
                 
        Up, _ = np.linalg.eig(Up)

        Uiso[p] = np.mean(Up).real
    
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
        
        s_ = Q[q]*inv_M_SP
        s_sq = s_*s_
                
        for p in range(n_pairs):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
                                                                                
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
            b_ij = (a_ij-cos(Qr_ij))/Qr_ij
            
            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw
            
            for r in prange(n_uvw, nogil=True):
                
                value += factors*(A_ij[r,p]*a_ij+B_ij[r,p]*b_ij)
                         
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
        
        s_ = Q[q]*inv_M_SP
        s_sq = s_*s_
        
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
                                    
            dw_n = exp(-0.5*Q_sq*Uiso[w])
            
            f_n = occ_n*ff_n*dw_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)*S_i_dot_S_i[p]   
                 
        auto[q] = value
      
    for q in prange(n_hkl, nogil=True):
        
        I[q] = (auto[q]/n_xyz+2*summation[q]/m_xyz)
        
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
                 double [:,:] A,
                 Py_ssize_t nu,
                 Py_ssize_t nv,
                 Py_ssize_t nw,
                 technique='Neutron'):
    
    cdef bint neutron = technique == 'Neutron'
    
    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_uvw = nu*nv*nw
    
    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
    
    rx_np = rx_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    ry_np = ry_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    rz_np = rz_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    
    i_np, j_np = np.triu_indices(m_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    A_r_np = np.copy(A_r, order='C')
        
    iu, iv, iw, i_atm = np.unravel_index(i_np, (mu,mv,mw,n_atm))
    ju, jv, jw, j_atm = np.unravel_index(j_np, (mu,mv,mw,n_atm))
        
    cu, cv, cw = np.unravel_index(np.arange(n_uvw), (nu,nv,nw))
    
    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cu[:,None], nw)
    
    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cu[:,None], nw)
    
    i_np = np.ascontiguousarray((i_atm+n_atm*(iw+nw*(iv+nv*iu))).T)
    j_np = np.ascontiguousarray((j_atm+n_atm*(jw+nw*(jv+nv*ju))).T)
                
    cdef double [::1] delta_ii = (1+A_r_np[m_np])**2
    cdef double [:,::1] delta_ij = (1+A_r_np[i_np])*(1+A_r_np[j_np])
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    cdef double [::1] r_ij = r_ij_np
        
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
    
    cdef Py_ssize_t p, q, r, u, v, w
    
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double factors, value = 0
        
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

        U = np.array([[U11[p], U12[p], U13[p]],
                      [U12[p], U22[p], U23[p]],
                      [U13[p], U23[p], U33[p]]])

        Up = np.dot(np.dot(D, U), D.T)
        
        Uxx[p] = Up[0,0]
        Uyy[p] = Up[1,1]
        Uzz[p] = Up[2,2]
        Uyz[p] = Up[1,2]
        Uxz[p] = Up[0,2]
        Uxy[p] = Up[0,1]
                 
        Up, _ = np.linalg.eig(Up)

        Uiso[p] = np.mean(Up).real
    
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
        
        if not neutron:
            
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
                
        for p in range(n_pairs):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
            
            if neutron:
                
                sl_k = b[u]
                sl_l = b[v]
                
            else:
                                                                    
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
            
            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw
            
            for r in prange(n_uvw, nogil=True):
                
                value += factors*delta_ij[p,r]*a_ij
                         
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
        
        if not neutron:
            
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            if neutron:
                
                sl_n = b[w]
                
            else:

                sl_n = a1[w]*exp(-b1[w]*s_sq)\
                     + a2[w]*exp(-b2[w]*s_sq)\
                     + a3[w]*exp(-b3[w]*s_sq)\
                     + a4[w]*exp(-b4[w]*s_sq)\
                     + c[w]
                                    
            dw_n = exp(-0.5*Q_sq*Uiso[w])
            
            f_n = occ_n*sl_n*dw_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)*delta_ii[p]
                    
        auto[q] = value
      
    for q in prange(n_hkl, nogil=True):
        
        I[q] = (auto[q]/n_xyz+2*summation[q]/m_xyz)
        
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
               double [:,:] A,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               int order,
               technique='Neutron'):

    cdef bint neutron = technique == 'Neutron'

    cdef Py_ssize_t n_atm = occupancy.shape[0]
    
    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_uvw = nu*nv*nw
    
    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
    
    rx_np = rx_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    ry_np = ry_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    rz_np = rz_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    
    i_np, j_np = np.triu_indices(m_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    Ux_np = np.copy(Ux, order='C')
    Uy_np = np.copy(Uy, order='C')
    Uz_np = np.copy(Uz, order='C')
    
    iu, iv, iw, i_atm = np.unravel_index(i_np, (mu,mv,mw,n_atm))
    ju, jv, jw, j_atm = np.unravel_index(j_np, (mu,mv,mw,n_atm))
        
    cu, cv, cw = np.unravel_index(np.arange(n_uvw), (nu,nv,nw))
    
    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cu[:,None], nw)
    
    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cu[:,None], nw)
    
    i_np = np.ascontiguousarray((i_atm+n_atm*(iw+nw*(iv+nv*iu))).T)
    j_np = np.ascontiguousarray((j_atm+n_atm*(jw+nw*(jv+nv*ju))).T)
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    cdef double [::1] r_ij = r_ij_np
    
    cdef double [::1] summation = np.zeros(Q.shape[0])
    
    cdef double [::1] auto = np.zeros(Q.shape[0])
    
    cdef Py_ssize_t n_pairs = r_ij.shape[0]
    
    cdef double Qr_ij, Qu_ij
    
    coeff_size = int(np.ceil((order+2)/2)*np.floor((order+2)/2))
    
    cdef double [::1] a_ij = np.zeros(order+1)
    cdef double [::1] A_ij = np.zeros(coeff_size)
    
    cdef double Qr_ij_pow = 1
    cdef double Qu_ij_pow = 1
    
    Ux_ij_np = Ux_np[j_np]-Ux_np[i_np]
    Uy_ij_np = Uy_np[j_np]-Uy_np[i_np]
    Uz_ij_np = Uz_np[j_np]-Uz_np[i_np]
    
    U_ij_np = np.sqrt(Ux_ij_np**2+Uy_ij_np**2+Uz_ij_np**2)
        
    cdef double [:,::1] U_ij = U_ij_np
    
    U_ij_mul_r_ij = (U_ij_np.T*r_ij_np).T
    
    U_ij_mul_r_ij[np.isclose(U_ij_mul_r_ij, 0)] = 1
    
    U_hat_ij_dot_r_hat_ij = (Ux_ij_np.T*rx_ij_np
                          +  Uy_ij_np.T*ry_ij_np\
                          +  Uz_ij_np.T*rz_ij_np).T/U_ij_mul_r_ij
                
    cdef double [:,:,::1] U_hat_ij_dot_r_hat_ij_pow = \
                          U_hat_ij_dot_r_hat_ij[:,None]\
                           **np.arange(order+1)[:,None]
                                     
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
    
    cdef Py_ssize_t p, q, r, s, t, u, v, w, x
        
    cdef double inv_M_SP = 1/(4*np.pi)
    
    cdef double factor, value, values = 0
    
    cs = np.ones(order+1)
    seq_id = np.array([])
    
    for t in range(order//2+1):
        seq_id = np.concatenate((seq_id, cs))
        cs = np.cumsum(cs[1:cs.size-1])
        
    num = (seq_id+1)*seq_id/2
    
    den = np.array([])
    
    for t in range(order//2+1):
        den = np.concatenate((den, factorial(np.arange(2*t, order+1))))
        
    sign = np.zeros(coeff_size)
        
    for r in range(order+1):
        t = r
        for s in range(r // 2+1):
            sign[t] = (-1)**r
            t += order-1-2*s

    cdef double [::1] coeff = sign*num/den
            
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
        
        Q_sq = Q[q]*Q[q]
        
        if not neutron:
            
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
                
        for p in range(n_pairs):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
            
            if neutron:
                
                sl_k = b[u]
                sl_l = b[v]
                
            else:
          
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
                                    
            f_k = occ_k*sl_k
            f_l = occ_l*sl_l
            
            f_k_real = f_k.real
            f_l_real = f_l.real
            
            f_k_imag = f_k.imag
            f_l_imag = f_l.imag
            
            Qr_ij = Q[q]*r_ij[p]
            
            Qr_ij_pow = 1
            
            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw
                
            for r in range(order+1):
                if (r == 0):
                    a_ij[0] = sin(Qr_ij)/Qr_ij
                elif (r == 1):
                    a_ij[1] = (a_ij[0]-cos(Qr_ij))/Qr_ij
                else:
                    a_ij[r] = (2*r-1)/Qr_ij*a_ij[r-1]-a_ij[r-2]
                
            for r in range(order+1):
                t = r
                for s in range(r // 2+1):
                    A_ij[t] = coeff[t]/Qr_ij_pow
                    t += order-1-2*s
                    Qr_ij_pow *= Qr_ij
                Qr_ij_pow = 1
                                                 
            for x in prange(n_uvw, nogil=True):
                
                Qu_ij = Q[q]*U_ij[p,x]
                
                Qu_ij_pow = 1

                values = 0
                for r in range(order+1):
                    t = r
                    for s in range(r // 2+1):
                        values = values+A_ij[t]*Qu_ij_pow\
                               * U_hat_ij_dot_r_hat_ij_pow[p,r-2*s,x]*a_ij[r]
                        t = t+order-1-2*s    
                    Qu_ij_pow = Qu_ij_pow*Qu_ij
                    
                value += factors*values
                    
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            if neutron:
                
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
      
    for q in prange(n_hkl, nogil=True):
        
        I[q] = (auto[q]/n_xyz+2*summation[q]/m_xyz)
        
    return I_np

def structural(double [::1] occupancy,
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
               double [:,:] A,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               technique='Neutron'):
    
    cdef bint neutron = technique == 'Neutron'
    
    cdef Py_ssize_t n_atm = occupancy.shape[0]
                
    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    cdef Py_ssize_t n_uvw = nu*nv*nw
    
    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm
    
    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')
    
    rx_np = rx_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    ry_np = ry_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    rz_np = rz_np.reshape((nu,nv,nw,n_atm))[:mu,:mv,:mw,:].flatten()
    
    i_np, j_np = np.triu_indices(m_xyz, k=1)
    
    k_np = np.mod(i_np, n_atm)
    l_np = np.mod(j_np, n_atm)
    
    m_np = np.arange(n_xyz, dtype=int)
    n_np = np.mod(m_np, n_atm)
    
    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)
    
    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)
    cdef long [::1] m = m_np.astype(int)
    cdef long [::1] n = n_np.astype(int)
    
    cdef double [::1] r_ij = r_ij_np
    
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
            
        U = np.array([[U11[p], U12[p], U13[p]],
                      [U12[p], U22[p], U23[p]],
                      [U13[p], U23[p], U33[p]]])

        Up = np.dot(np.dot(D, U), D.T)
        
        Uxx[p] = Up[0,0]
        Uyy[p] = Up[1,1]
        Uzz[p] = Up[2,2]
        Uyz[p] = Up[1,2]
        Uxz[p] = Up[0,2]
        Uxy[p] = Up[0,1]
                 
        Up, _ = np.linalg.eig(Up)

        Uiso[p] = np.mean(Up).real
    
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
                
        if not neutron:
            
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
                
        for p in prange(n_pairs, nogil=True):
            
            u, v = k[p], l[p]
            
            occ_k = occupancy[u]
            occ_l = occupancy[v]
            
            if neutron:
                
                sl_k = b[u]
                sl_l = b[v]
                
            else:
                            
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
                        
            value += (f_k_real*f_l_real+f_k_imag*f_l_imag)*a_ij
                                     
        summation[q] = value
                         
    for q in range(n_hkl):
        
        value = 0
        
        Q_sq = Q[q]*Q[q]
        
        if not neutron:
        
            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_
        
        for p in prange(n_xyz, nogil=True):
            
            w = n[p]
            
            occ_n = occupancy[w]
            
            if neutron:
                
                sl_n = b[w]
                
            else:
                                                                    
                sl_n = a1[w]*exp(-b1[w]*s_sq)\
                     + a2[w]*exp(-b2[w]*s_sq)\
                     + a3[w]*exp(-b3[w]*s_sq)\
                     + a4[w]*exp(-b4[w]*s_sq)\
                     + c[w]
                                    
            dw_n = exp(-0.5*Q_sq*Uiso[w])
            
            f_n = occ_n*sl_n*dw_n
            
            f_n_real = f_n.real
            f_n_imag = f_n.imag
            
            value += (f_n_real*f_n_real+f_n_imag*f_n_imag)
                    
        auto[q] = value
      
    for q in prange(n_hkl, nogil=True):
        
        I[q] = (auto[q]/n_xyz+2*summation[q]/m_xyz)
        
    return I_np

    # scale = n_xyz/((sqrt(8*n_pairs+1)+1)/2)