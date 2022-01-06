#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython
cimport openmp

from scipy.special import factorial

from libc.math cimport M_PI, cos, sin, exp, sqrt

from disorder.material import crystal, tables

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
             double [:,:] A,
             double [:,:] D,
             Py_ssize_t nu,
             Py_ssize_t nv,
             Py_ssize_t nw,
             double [::1] g):

    cdef Py_ssize_t n_atm = occupancy.shape[0]

    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2

    cdef Py_ssize_t n_hkl = Q.shape[0]

    cdef Py_ssize_t m_uvw = mu*mv*mw
    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm

    m_np = np.arange(n_atm, dtype=int)
    n_np = np.mod(m_np, n_atm)

    Sx_np = np.copy(Sx, order='C')
    Sy_np = np.copy(Sy, order='C')
    Sz_np = np.copy(Sz, order='C')

    c_uvw = np.arange(n_uvw)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    is_np = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm))
    js_np = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm))

    i_np = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm))
    j_np = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm))

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    diff_u = ju-iu
    diff_v = jv-iv
    diff_w = jw-iw

    diff_u[diff_u >= mu] -= nu
    diff_v[diff_v >= mv] -= nv
    diff_w[diff_w >= mw] -= nw

    diff_u[diff_u <= -mu] += nu
    diff_v[diff_v <= -mv] += nv
    diff_w[diff_w <= -mw] += nw

    mult_s_np = (mu-np.abs(diff_u))*(mv-np.abs(diff_v))*(mw-np.abs(diff_w))*1.

    cdef double [::1] mult_s = mult_s_np
    cdef double [::1] mult = np.full(n_uvw, m_uvw, dtype=float)
    
    A_inv = np.linalg.inv(A)

    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')

    rx_s_ij_np = rx_np[js_np]-rx_np[is_np]
    ry_s_ij_np = ry_np[js_np]-ry_np[is_np]
    rz_s_ij_np = rz_np[js_np]-rz_np[is_np]

    u_s_ij, v_s_ij, w_s_ij = crystal.transform(rx_s_ij_np, 
                                               ry_s_ij_np, 
                                               rz_s_ij_np, A_inv)

    u_s_ij[u_s_ij <= -mu] += nu
    v_s_ij[v_s_ij <= -mv] += nv
    w_s_ij[w_s_ij <= -mw] += nw

    u_s_ij[u_s_ij >= mu] -= nu
    v_s_ij[v_s_ij >= mv] -= nv
    w_s_ij[w_s_ij >= mw] -= nw

    rx_s_ij_np, ry_s_ij_np, rz_s_ij_np = crystal.transform(u_s_ij,
                                                           v_s_ij,
                                                           w_s_ij, A)

    rs_ij_np = np.sqrt(rx_s_ij_np**2+ry_s_ij_np**2+rz_s_ij_np**2)

    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    u_ij, v_ij, w_ij = crystal.transform(rx_ij_np, ry_ij_np, rz_ij_np, A_inv)

    u_ij[u_ij <= -mu] += nu
    v_ij[v_ij <= -mv] += nv
    w_ij[w_ij <= -mw] += nw

    u_ij[u_ij >= mu] -= nu
    v_ij[v_ij >= mv] -= nv
    w_ij[w_ij >= mw] -= nw
    
    rx_ij_np, ry_ij_np, rz_ij_np = crystal.transform(u_ij, v_ij, w_ij, A)
    
    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)

    ks_np = np.mod(is_np[:,0], n_atm)
    ls_np = np.mod(js_np[:,0], n_atm)

    k_np = np.mod(i_np[:,0], n_atm)
    l_np = np.mod(j_np[:,0], n_atm)

    Ss_i_dot_Ss_j_np = Sx_np[is_np]*Sx_np[js_np]\
                     + Sy_np[is_np]*Sy_np[js_np]\
                     + Sz_np[is_np]*Sz_np[js_np]

    S_i_dot_S_j_np = Sx_np[i_np]*Sx_np[j_np]\
                   + Sy_np[i_np]*Sy_np[j_np]\
                   + Sz_np[i_np]*Sz_np[j_np]

    Ss_i_dot_rs_ij_np = ((Sx_np[is_np]*rx_s_ij_np+\
                          Sy_np[is_np]*ry_s_ij_np+\
                          Sz_np[is_np]*rz_s_ij_np)/rs_ij_np)

    S_i_dot_r_ij_np = ((Sx_np[i_np]*rx_ij_np+\
                        Sy_np[i_np]*ry_ij_np+\
                        Sz_np[i_np]*rz_ij_np)/r_ij_np)

    Ss_j_dot_rs_ij_np = ((Sx_np[js_np]*rx_s_ij_np+\
                          Sy_np[js_np]*ry_s_ij_np+\
                          Sz_np[js_np]*rz_s_ij_np)/rs_ij_np)

    S_j_dot_r_ij_np = ((Sx_np[j_np]*rx_ij_np+\
                        Sy_np[j_np]*ry_ij_np+\
                        Sz_np[j_np]*rz_ij_np)/r_ij_np)

    Ss_i_dot_rs_ij_Ss_j_dot_rs_ij_np = Ss_i_dot_rs_ij_np*Ss_j_dot_rs_ij_np

    S_i_dot_r_ij_S_j_dot_r_ij_np = S_i_dot_r_ij_np*S_j_dot_r_ij_np

    cdef double [:,::1] As_ij = Ss_i_dot_Ss_j_np\
                              - Ss_i_dot_rs_ij_Ss_j_dot_rs_ij_np

    cdef double [:,::1] A_ij = S_i_dot_S_j_np-S_i_dot_r_ij_S_j_dot_r_ij_np

    cdef double [:,::1] Bs_ij = 3*Ss_i_dot_rs_ij_Ss_j_dot_rs_ij_np\
                                - Ss_i_dot_Ss_j_np

    cdef double [:,::1] B_ij = 3*S_i_dot_r_ij_S_j_dot_r_ij_np-S_i_dot_S_j_np

    cdef long [::1] ks = ks_np.astype(int)
    cdef long [::1] ls = ls_np.astype(int)

    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)

    cdef long [::1] n = n_np.astype(int)

    cdef double [:,::1] rs_ij = rs_ij_np
    cdef double [:,::1] r_ij = r_ij_np

    cdef double [::1] summation = np.zeros(Q.shape[0])

    cdef double [::1] auto = np.zeros(Q.shape[0])

    cdef Py_ssize_t n_pairs = rs_ij.shape[1]
    cdef Py_ssize_t n_pair = r_ij.shape[1]

    cdef Py_ssize_t n_types = n_atm*n_atm
    cdef Py_ssize_t n_type = n_atm*(n_atm-1)//2

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

    cdef Py_ssize_t a, p, q, r, u, v, w

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

        for a in range(n_types):

            u, v = ks[a], ls[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pairs, nogil=True):

                Qr_ij = Q[q]*rs_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij
                b_ij = (a_ij-cos(Qr_ij))/Qr_ij

                value += factors*(As_ij[a,p]*a_ij+Bs_ij[a,p]*b_ij)*mult_s[p]

        for a in range(n_type):

            u, v = k[a], l[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pair, nogil=True):

                Qr_ij = Q[q]*r_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij
                b_ij = (a_ij-cos(Qr_ij))/Qr_ij

                value += factors*(A_ij[a,p]*a_ij+B_ij[a,p]*b_ij)*mult[p]

        summation[q] = value

    for q in range(n_hkl):

        value = 0

        Q_sq = Q[q]*Q[q]

        s_ = Q[q]*inv_M_SP
        s_sq = s_*s_

        for p in prange(n_atm, nogil=True):

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

            value += 2.0*(f_n_real*f_n_real+f_n_imag*f_n_imag)/3.0

        auto[q] = value

    for q in prange(n_hkl, nogil=True):

        I[q] = (auto[q]/n_atm+2*summation[q]/m_xyz)

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
                 double [:,:] A,
                 double [:,:] D,
                 Py_ssize_t nu,
                 Py_ssize_t nv,
                 Py_ssize_t nw,
                 source='neutron'):

    cdef bint neutron = source == 'neutron'

    cdef Py_ssize_t n_atm = occupancy.shape[0]

    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2

    cdef Py_ssize_t n_hkl = Q.shape[0]

    cdef Py_ssize_t m_uvw = mu*mv*mw
    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm

    m_np = np.arange(n_atm, dtype=int)
    n_np = np.mod(m_np, n_atm)

    A_r_np = np.copy(A_r, order='C')

    c_uvw = np.arange(n_uvw)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    is_np = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm))
    js_np = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm))

    i_np = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm))
    j_np = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm))

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    diff_u = ju-iu
    diff_v = jv-iv
    diff_w = jw-iw

    diff_u[diff_u >= mu] -= nu
    diff_v[diff_v >= mv] -= nv
    diff_w[diff_w >= mw] -= nw

    diff_u[diff_u <= -mu] += nu
    diff_v[diff_v <= -mv] += nv
    diff_w[diff_w <= -mw] += nw

    mult_s_np = (mu-np.abs(diff_u))*(mv-np.abs(diff_v))*(mw-np.abs(diff_w))*1.

    cdef double [::1] mult_s = mult_s_np
    cdef double [::1] mult = np.full(n_uvw, m_uvw, dtype=float)
    
    A_inv = np.linalg.inv(A)

    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')

    rx_s_ij_np = rx_np[js_np]-rx_np[is_np]
    ry_s_ij_np = ry_np[js_np]-ry_np[is_np]
    rz_s_ij_np = rz_np[js_np]-rz_np[is_np]

    u_s_ij, v_s_ij, w_s_ij = crystal.transform(rx_s_ij_np, 
                                               ry_s_ij_np, 
                                               rz_s_ij_np, A_inv)

    u_s_ij[u_s_ij <= -mu] += nu
    v_s_ij[v_s_ij <= -mv] += nv
    w_s_ij[w_s_ij <= -mw] += nw

    u_s_ij[u_s_ij >= mu] -= nu
    v_s_ij[v_s_ij >= mv] -= nv
    w_s_ij[w_s_ij >= mw] -= nw

    rx_s_ij_np, ry_s_ij_np, rz_s_ij_np = crystal.transform(u_s_ij,
                                                           v_s_ij,
                                                           w_s_ij, A)

    rs_ij_np = np.sqrt(rx_s_ij_np**2+ry_s_ij_np**2+rz_s_ij_np**2)

    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    u_ij, v_ij, w_ij = crystal.transform(rx_ij_np, ry_ij_np, rz_ij_np, A_inv)

    u_ij[u_ij <= -mu] += nu
    v_ij[v_ij <= -mv] += nv
    w_ij[w_ij <= -mw] += nw

    u_ij[u_ij >= mu] -= nu
    v_ij[v_ij >= mv] -= nv
    w_ij[w_ij >= mw] -= nw
    
    rx_ij_np, ry_ij_np, rz_ij_np = crystal.transform(u_ij, v_ij, w_ij, A)

    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)

    ks_np = np.mod(is_np[:,0], n_atm)
    ls_np = np.mod(js_np[:,0], n_atm)

    k_np = np.mod(i_np[:,0], n_atm)
    l_np = np.mod(j_np[:,0], n_atm)

    cdef double [::1] delta_ii = (1+A_r_np[m_np])**2

    cdef double [:,::1] delta_s_ij = (1+A_r_np[is_np])*(1+A_r_np[js_np])
    cdef double [:,::1] delta_ij = (1+A_r_np[i_np])*(1+A_r_np[j_np])

    cdef long [::1] ks = ks_np.astype(int)
    cdef long [::1] ls = ls_np.astype(int)

    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)

    cdef long [::1] n = n_np.astype(int)

    cdef double [:,::1] rs_ij = rs_ij_np
    cdef double [:,::1] r_ij = r_ij_np

    cdef double [::1] summation = np.zeros(Q.shape[0])

    cdef double [::1] auto = np.zeros(Q.shape[0])

    cdef Py_ssize_t n_pairs = rs_ij.shape[1]
    cdef Py_ssize_t n_pair = r_ij.shape[1]

    cdef Py_ssize_t n_types = n_atm*n_atm
    cdef Py_ssize_t n_type = n_atm*(n_atm-1)//2

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

    cdef Py_ssize_t a, p, q, r, u, v, w

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

        for a in range(n_types):

            u, v = ks[a], ls[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pairs, nogil=True):

                Qr_ij = Q[q]*rs_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij

                value += factors*delta_s_ij[a,p]*a_ij*mult_s[p]

        for a in range(n_type):

            u, v = k[a], l[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pair, nogil=True):

                Qr_ij = Q[q]*r_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij

                value += factors*delta_ij[a,p]*a_ij*mult[p]

        summation[q] = value

    for q in range(n_hkl):

        value = 0

        Q_sq = Q[q]*Q[q]

        if not neutron:

            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_

        for p in prange(n_atm, nogil=True):

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

        I[q] = (auto[q]/n_atm+2*summation[q]/m_xyz)

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
               double [:,:] A,
               double [:,:] D,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               int order,
               source='neutron'):

    cdef bint neutron = source == 'neutron'

    cdef Py_ssize_t thread_id, num_threads = openmp.omp_get_max_threads()

    cdef Py_ssize_t n_atm = occupancy.shape[0]

    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2

    cdef Py_ssize_t n_hkl = Q.shape[0]

    cdef Py_ssize_t m_uvw = mu*mv*mw
    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm

    m_np = np.arange(n_atm, dtype=int)
    n_np = np.mod(m_np, n_atm)

    Ux_np = np.copy(Ux, order='C')
    Uy_np = np.copy(Uy, order='C')
    Uz_np = np.copy(Uz, order='C')

    c_uvw = np.arange(n_uvw)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    is_np = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm))
    js_np = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm))

    i_np = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm))
    j_np = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm))

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    diff_u = ju-iu
    diff_v = jv-iv
    diff_w = jw-iw

    diff_u[diff_u >= mu] -= nu
    diff_v[diff_v >= mv] -= nv
    diff_w[diff_w >= mw] -= nw

    diff_u[diff_u <= -mu] += nu
    diff_v[diff_v <= -mv] += nv
    diff_w[diff_w <= -mw] += nw

    mult_s_np = (mu-np.abs(diff_u))*(mv-np.abs(diff_v))*(mw-np.abs(diff_w))*1.

    cdef double [::1] mult_s = mult_s_np
    cdef double [::1] mult = np.full(n_uvw, m_uvw, dtype=float)

    A_inv = np.linalg.inv(A)

    ks_np = np.mod(is_np[:,0], n_atm)
    ls_np = np.mod(js_np[:,0], n_atm)

    k_np = np.mod(i_np[:,0], n_atm)
    l_np = np.mod(j_np[:,0], n_atm)

    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')

    rx_s_ij_np = rx_np[js_np]-rx_np[is_np]
    ry_s_ij_np = ry_np[js_np]-ry_np[is_np]
    rz_s_ij_np = rz_np[js_np]-rz_np[is_np]

    u_s_ij, v_s_ij, w_s_ij = crystal.transform(rx_s_ij_np, 
                                               ry_s_ij_np, 
                                               rz_s_ij_np, A_inv)

    u_s_ij[u_s_ij <= -mu] += nu
    v_s_ij[v_s_ij <= -mv] += nv
    w_s_ij[w_s_ij <= -mw] += nw

    u_s_ij[u_s_ij >= mu] -= nu
    v_s_ij[v_s_ij >= mv] -= nv
    w_s_ij[w_s_ij >= mw] -= nw

    rx_s_ij_np, ry_s_ij_np, rz_s_ij_np = crystal.transform(u_s_ij,
                                                           v_s_ij,
                                                           w_s_ij, A)

    rs_ij_np = np.sqrt(rx_s_ij_np**2+ry_s_ij_np**2+rz_s_ij_np**2)

    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    u_ij, v_ij, w_ij = crystal.transform(rx_ij_np, ry_ij_np, rz_ij_np, A_inv)

    u_ij[u_ij <= -mu] += nu
    v_ij[v_ij <= -mv] += nv
    w_ij[w_ij <= -mw] += nw

    u_ij[u_ij >= mu] -= nu
    v_ij[v_ij >= mv] -= nv
    w_ij[w_ij >= mw] -= nw
    
    rx_ij_np, ry_ij_np, rz_ij_np = crystal.transform(u_ij, v_ij, w_ij, A)

    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)

    Ux_s_ij_np = Ux_np[js_np]-Ux_np[is_np]
    Uy_s_ij_np = Uy_np[js_np]-Uy_np[is_np]
    Uz_s_ij_np = Uz_np[js_np]-Uz_np[is_np]

    Us_ij_np = np.sqrt(Ux_s_ij_np**2+Uy_s_ij_np**2+Uz_s_ij_np**2)

    Ux_ij_np = Ux_np[j_np]-Ux_np[i_np]
    Uy_ij_np = Uy_np[j_np]-Uy_np[i_np]
    Uz_ij_np = Uz_np[j_np]-Uz_np[i_np]

    U_ij_np = np.sqrt(Ux_ij_np**2+Uy_ij_np**2+Uz_ij_np**2)

    cdef double [:,::1] Us_ij = Us_ij_np
    cdef double [:,::1] U_ij = U_ij_np

    Us_ij_mul_rs_ij = Us_ij_np*rs_ij_np

    U_ij_mul_r_ij = U_ij_np*r_ij_np

    Us_ij_mul_rs_ij[np.isclose(Us_ij_mul_rs_ij, 0)] = 1

    U_ij_mul_r_ij[np.isclose(U_ij_mul_r_ij, 0)] = 1

    Us_hat_ij_dot_rs_hat_ij = (Ux_s_ij_np*rx_s_ij_np
                            +  Uy_s_ij_np*ry_s_ij_np\
                            +  Uz_s_ij_np*rz_s_ij_np)/Us_ij_mul_rs_ij

    U_hat_ij_dot_r_hat_ij = (Ux_ij_np*rx_ij_np
                          +  Uy_ij_np*ry_ij_np\
                          +  Uz_ij_np*rz_ij_np)/U_ij_mul_r_ij

    cdef double [:,:,::1] Us_hat_ij_dot_rs_hat_ij_pow = \
        Us_hat_ij_dot_rs_hat_ij[:,:,None]**np.arange(order+1)

    cdef double [:,:,::1] U_hat_ij_dot_r_hat_ij_pow = \
        U_hat_ij_dot_r_hat_ij[:,:,None]**np.arange(order+1)

    cdef long [::1] ks = ks_np.astype(int)
    cdef long [::1] ls = ls_np.astype(int)

    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)

    cdef long [::1] n = n_np.astype(int)

    cdef double [:,::1] rs_ij = rs_ij_np
    cdef double [:,::1] r_ij = r_ij_np

    cdef double [::1] summation = np.zeros(Q.shape[0])

    cdef double [::1] auto = np.zeros(Q.shape[0])

    cdef Py_ssize_t n_pairs = rs_ij.shape[1]
    cdef Py_ssize_t n_pair = r_ij.shape[1]

    cdef Py_ssize_t n_types = n_atm*n_atm
    cdef Py_ssize_t n_type = n_atm*(n_atm-1)//2

    cdef double Qr_ij, Qu_ij

    coeff_size = int(np.ceil(float(order+2)/2)*np.floor(float(order+2)/2))

    cdef double [:,::1] a_ij = np.zeros((num_threads,order+1))
    cdef double [:,::1] A_ij = np.zeros((num_threads,coeff_size))

    cdef double Qr_ij_pow = 1
    cdef double Qu_ij_pow = 1

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

    cdef Py_ssize_t a, p, q, r, s, t, u, v, w

    cdef double inv_M_SP = 1/(4*np.pi)

    cdef double factor, factors, value, values = 0

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
            sign[t] = (-1)**(r-s)
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

        for a in range(n_types):

            u, v = ks[a], ls[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pairs, nogil=True):

                Qr_ij = Q[q]*rs_ij[a,p]
                Qu_ij = Q[q]*Us_ij[a,p]

                Qr_ij_pow = 1
                Qu_ij_pow = 1

                thread_id = openmp.omp_get_thread_num()

                for r in range(order+1):
                    if (r == 0):
                        a_ij[thread_id,0] = sin(Qr_ij)/Qr_ij
                    elif (r == 1):
                        a_ij[thread_id,1] = (a_ij[thread_id,0]\
                                             -cos(Qr_ij))/Qr_ij
                    else:
                        a_ij[thread_id,r] = (2*r-1)/Qr_ij*a_ij[thread_id,r-1]\
                                                         -a_ij[thread_id,r-2]

                for r in range(order+1):
                    t = r
                    for s in range(r // 2+1):
                        A_ij[thread_id,t] = coeff[t]*Qu_ij_pow/Qr_ij_pow\
                            * Us_hat_ij_dot_rs_hat_ij_pow[a,p,r-2*s]
                        t = t+order-1-2*s
                        Qr_ij_pow = Qr_ij_pow*Qr_ij
                    Qr_ij_pow = 1
                    Qu_ij_pow = Qu_ij_pow*Qu_ij

                values = 0
                for r in range(order+1):
                    t = r
                    for s in range(r // 2+1):
                        values = values+A_ij[thread_id,t]*a_ij[thread_id,r-s]
                        t = t+order-1-2*s

                value += factors*values*mult_s[p]

        for a in range(n_type):

            u, v = k[a], l[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pair, nogil=True):

                Qr_ij = Q[q]*r_ij[a,p]
                Qu_ij = Q[q]*U_ij[a,p]

                Qr_ij_pow = 1
                Qu_ij_pow = 1

                thread_id = openmp.omp_get_thread_num()

                for r in range(order+1):
                    if (r == 0):
                        a_ij[thread_id,0] = sin(Qr_ij)/Qr_ij
                    elif (r == 1):
                        a_ij[thread_id,1] = (a_ij[thread_id,0]\
                                              -cos(Qr_ij))/Qr_ij
                    else:
                        a_ij[thread_id,r] = (2*r-1)/Qr_ij*a_ij[thread_id,r-1]\
                                                          -a_ij[thread_id,r-2]

                for r in range(order+1):
                    t = r
                    for s in range(r // 2+1):
                        A_ij[thread_id,t] = coeff[t]*Qu_ij_pow/Qr_ij_pow\
                            * U_hat_ij_dot_r_hat_ij_pow[a,p,r-2*s]
                        t = t+order-1-2*s
                        Qr_ij_pow = Qr_ij_pow*Qr_ij
                    Qr_ij_pow = 1
                    Qu_ij_pow = Qu_ij_pow*Qu_ij

                values = 0
                for r in range(order+1):
                    t = r
                    for s in range(r // 2+1):
                        values = values+A_ij[thread_id,t]*a_ij[thread_id,r-s]
                        t = t+order-1-2*s

                value += factors*values*mult[p]

        summation[q] = value

    for q in range(n_hkl):

        value = 0

        for p in prange(n_atm, nogil=True):

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

        I[q] = (auto[q]/n_atm+2*summation[q]/m_xyz)

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
               double [:,:] A,
               double [:,:] D,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               source='neutron'):

    cdef bint neutron = source == 'neutron'

    cdef Py_ssize_t n_atm = occupancy.shape[0]

    cdef Py_ssize_t mu = (nu+1) // 2
    cdef Py_ssize_t mv = (nv+1) // 2
    cdef Py_ssize_t mw = (nw+1) // 2

    cdef Py_ssize_t n_hkl = Q.shape[0]

    cdef Py_ssize_t m_uvw = mu*mv*mw
    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t m_xyz = mu*mv*mw*n_atm
    cdef Py_ssize_t n_xyz = nu*nv*nw*n_atm

    m_np = np.arange(n_atm, dtype=int)
    n_np = np.mod(m_np, n_atm)

    c_uvw = np.arange(n_uvw)

    cu, cv, cw = np.unravel_index(c_uvw, (nu,nv,nw))

    i_lat, j_lat = np.triu_indices(m_uvw, k=1)

    iu, iv, iw = np.unravel_index(i_lat, (mu,mv,mw))
    ju, jv, jw = np.unravel_index(j_lat, (mu,mv,mw))

    iu = np.mod(iu+cu[:,None], nu)
    iv = np.mod(iv+cv[:,None], nv)
    iw = np.mod(iw+cw[:,None], nw)

    ju = np.mod(ju+cu[:,None], nu)
    jv = np.mod(jv+cv[:,None], nv)
    jw = np.mod(jw+cw[:,None], nw)

    i_lat = np.ravel_multi_index((iu,iv,iw), (nu,nv,nw))
    j_lat = np.ravel_multi_index((ju,jv,jw), (nu,nv,nw))

    pairs = np.stack((i_lat,j_lat)).reshape(2,n_uvw*m_uvw*(m_uvw-1)//2)

    i_lat, j_lat = np.unique(np.sort(pairs, axis=0), axis=1)

    i_atm, j_atm = np.triu_indices(n_atm, k=1)

    i_atms = np.concatenate((i_atm,j_atm))
    j_atms = np.concatenate((j_atm,i_atm))

    i_atms = np.concatenate((i_atms,np.arange(n_atm)))
    j_atms = np.concatenate((j_atms,np.arange(n_atm)))

    is_np = np.ravel_multi_index((i_lat,i_atms[:,None]), (n_uvw,n_atm))
    js_np = np.ravel_multi_index((j_lat,j_atms[:,None]), (n_uvw,n_atm))

    i_np = np.ravel_multi_index((c_uvw,i_atm[:,None]), (n_uvw,n_atm))
    j_np = np.ravel_multi_index((c_uvw,j_atm[:,None]), (n_uvw,n_atm))

    iu, iv, iw = np.unravel_index(i_lat, (nu,nv,nw))
    ju, jv, jw = np.unravel_index(j_lat, (nu,nv,nw))

    diff_u = ju-iu
    diff_v = jv-iv
    diff_w = jw-iw

    diff_u[diff_u >= mu] -= nu
    diff_v[diff_v >= mv] -= nv
    diff_w[diff_w >= mw] -= nw

    diff_u[diff_u <= -mu] += nu
    diff_v[diff_v <= -mv] += nv
    diff_w[diff_w <= -mw] += nw

    mult_s_np = (mu-np.abs(diff_u))*(mv-np.abs(diff_v))*(mw-np.abs(diff_w))*1.

    cdef double [::1] mult_s = mult_s_np
    cdef double [::1] mult = np.full(n_uvw, m_uvw, dtype=float)
    
    A_inv = np.linalg.inv(A)

    rx_np = np.copy(rx, order='C')
    ry_np = np.copy(ry, order='C')
    rz_np = np.copy(rz, order='C')

    rx_s_ij_np = rx_np[js_np]-rx_np[is_np]
    ry_s_ij_np = ry_np[js_np]-ry_np[is_np]
    rz_s_ij_np = rz_np[js_np]-rz_np[is_np]
    
    u_s_ij, v_s_ij, w_s_ij = crystal.transform(rx_s_ij_np, 
                                               ry_s_ij_np, 
                                               rz_s_ij_np, A_inv)

    u_s_ij[u_s_ij <= -mu] += nu
    v_s_ij[v_s_ij <= -mv] += nv
    w_s_ij[w_s_ij <= -mw] += nw

    u_s_ij[u_s_ij >= mu] -= nu
    v_s_ij[v_s_ij >= mv] -= nv
    w_s_ij[w_s_ij >= mw] -= nw

    rx_s_ij_np, ry_s_ij_np, rz_s_ij_np = crystal.transform(u_s_ij,
                                                           v_s_ij,
                                                           w_s_ij, A)

    rs_ij_np = np.sqrt(rx_s_ij_np**2+ry_s_ij_np**2+rz_s_ij_np**2)

    rx_ij_np = rx_np[j_np]-rx_np[i_np]
    ry_ij_np = ry_np[j_np]-ry_np[i_np]
    rz_ij_np = rz_np[j_np]-rz_np[i_np]
    
    u_ij, v_ij, w_ij = crystal.transform(rx_ij_np, ry_ij_np, rz_ij_np, A_inv)

    u_ij[u_ij <= -mu] += nu
    v_ij[v_ij <= -mv] += nv
    w_ij[w_ij <= -mw] += nw

    u_ij[u_ij >= mu] -= nu
    v_ij[v_ij >= mv] -= nv
    w_ij[w_ij >= mw] -= nw
    
    rx_ij_np, ry_ij_np, rz_ij_np = crystal.transform(u_ij, v_ij, w_ij, A)

    r_ij_np = np.sqrt(rx_ij_np**2+ry_ij_np**2+rz_ij_np**2)

    ks_np = np.mod(is_np[:,0], n_atm)
    ls_np = np.mod(js_np[:,0], n_atm)

    k_np = np.mod(i_np[:,0], n_atm)
    l_np = np.mod(j_np[:,0], n_atm)

    cdef long [::1] ks = ks_np.astype(int)
    cdef long [::1] ls = ls_np.astype(int)

    cdef long [::1] k = k_np.astype(int)
    cdef long [::1] l = l_np.astype(int)

    cdef long [::1] n = n_np.astype(int)

    cdef double [:,::1] rs_ij = rs_ij_np
    cdef double [:,::1] r_ij = r_ij_np

    cdef double [::1] summation = np.zeros(Q.shape[0])

    cdef double [::1] auto = np.zeros(Q.shape[0])

    cdef Py_ssize_t n_pairs = rs_ij.shape[1]
    cdef Py_ssize_t n_pair = r_ij.shape[1]

    cdef Py_ssize_t n_types = n_atm*n_atm
    cdef Py_ssize_t n_type = n_atm*(n_atm-1)//2

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

    cdef Py_ssize_t a, p, q, r, u, v, w

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

        for a in range(n_types):

            u, v = ks[a], ls[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pairs, nogil=True):

                Qr_ij = Q[q]*rs_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij

                value += factors*a_ij*mult_s[p]

        for a in range(n_type):

            u, v = k[a], l[a]

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

            factors = (f_k_real*f_l_real+f_k_imag*f_l_imag)/n_uvw

            for p in prange(n_pair, nogil=True):

                Qr_ij = Q[q]*r_ij[a,p]

                a_ij = sin(Qr_ij)/Qr_ij

                value += factors*a_ij*mult[p]

        summation[q] = value

    for q in range(n_hkl):

        value = 0

        Q_sq = Q[q]*Q[q]

        if not neutron:

            s_ = Q[q]*inv_M_SP
            s_sq = s_*s_

        for p in prange(n_atm, nogil=True):

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

        I[q] = (auto[q]/n_atm+2*summation[q]/m_xyz)

    return I_np