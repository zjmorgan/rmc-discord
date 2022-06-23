#cython: boundscheck=False, wraparound=True, nonecheck=False, cdivision=True
#cython: language_level=3

import os, sys

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel

cimport cython
cimport openmp

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport M_PI, cos, sin, exp, sqrt, acos, fabs, log

from disorder.diffuse cimport filters

def parallelism(app=True):

    threads = os.environ.get('OMP_NUM_THREADS')

    cdef Py_ssize_t i_thread, num_threads

    for i_thread in prange(1, nogil=True):
        num_threads = openmp.omp_get_max_threads()

    if (len(sys.argv) > 1 and app):
        openmp.omp_set_num_threads(int(sys.argv[1]))
    elif (threads is None):
        if (num_threads > 8):
            openmp.omp_set_num_threads(8)
        else:
            openmp.omp_set_num_threads(num_threads)

    for i_thread in prange(1, nogil=True):
        num_threads = openmp.omp_get_max_threads()

    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    print('threads:', num_threads)

def threads():

    cdef Py_ssize_t i_thread, thread_id, num_threads

    for i_thread in prange(1, nogil=True):
        num_threads = openmp.omp_get_max_threads()

    buf_np = np.zeros(num_threads, dtype=int)

    cdef long [:] buf = buf_np

    with nogil, parallel(num_threads=num_threads):
        thread_id = openmp.omp_get_thread_num()
        buf[thread_id] = thread_id

    for i_thread in range(num_threads):
        print('id:', buf[i_thread])

cpdef (double, Py_ssize_t) original_scalar(double [::1] A) nogil:

    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i = rand() % n

    return A[i], i

cpdef (double, \
       double, \
       double, \
       Py_ssize_t) original_vector(double [::1] A,
                                   double [::1] B,
                                   double [::1] C) nogil:

    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i = rand() % n

    return A[i], B[i], C[i], i

cpdef Py_ssize_t original_scalars(double [::1] B,
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

cpdef Py_ssize_t original_vectors(double [::1] D,
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

cpdef double candidate_composition(double A, double value) nogil:

    return 1/value-2-A

cpdef (double, double, double) candidate_moment(double A,
                                                double B,
                                                double C,
                                                double value) nogil:

    cdef double theta, phi

    cdef double u, v, w, n

    theta = 2.0*M_PI*random_uniform()
    phi = acos(1.0-2.0*random_uniform())

    u = value*sin(phi)*cos(theta)
    v = value*sin(phi)*sin(theta)
    w = value*cos(phi)

    return u, v, w

cpdef (double, double, double) candidate_displacement(double A,
                                                      double B,
                                                      double C,
                                                      double D,
                                                      double E,
                                                      double F) nogil:

    cdef double theta, phi

    cdef double u, v, w, l, m, n

    l, m, n = random_gaussian_3d()

    u, v, w = A*l, F*l+B*m, E*l+D*m+C*n

    return u, v, w

cdef double complex cexp(double complex z) nogil:

    cdef double x = z.real
    cdef double y = z.imag

    return exp(x)*(cos(y)+1j*sin(y))

cdef double complex iexp(double y) nogil:

    return cos(y)+1j*sin(y)

cpdef void extract_complex(double complex [::1] B,
                           double complex [::1] A,
                           Py_ssize_t j,
                           Py_ssize_t n) nogil:

    cdef Py_ssize_t m = B.shape[0]

    cdef Py_ssize_t i

    for i in prange(m):
        B[i] = A[j+n*i]

cpdef void insert_complex(double complex [::1] A,
                          double complex [::1] B,
                          Py_ssize_t j,
                          Py_ssize_t n) nogil:

    cdef Py_ssize_t m = B.shape[0]

    cdef Py_ssize_t i

    for i in prange(m):
        A[j+n*i] = B[i]

cpdef void extract_many_complex(double complex [::1] B,
                                double complex [::1] A,
                                long [::1] indices,
                                Py_ssize_t n) nogil:

    cdef Py_ssize_t m = indices.shape[0]
    cdef Py_ssize_t l = B.shape[0] // m

    cdef Py_ssize_t i, j, k

    for i in prange(l):

        for j in range(m):

            k = indices[j]

            B[j+m*i] = A[k+n*i]

cpdef void insert_many_complex(double complex [::1] A,
                               double complex [::1] B,
                               long [::1] indices,
                               Py_ssize_t n) nogil:

    cdef Py_ssize_t m = indices.shape[0]
    cdef Py_ssize_t l = B.shape[0] // m

    cdef Py_ssize_t i, j, k

    for i in prange(l):

        for j in range(m):

            k = indices[j]

            A[k+n*i] = B[j+m*i]

cpdef void extract_real(double [::1] B,
                        double [::1] A,
                        Py_ssize_t j,
                        Py_ssize_t n) nogil:

    cdef Py_ssize_t m = B.shape[0]

    cdef Py_ssize_t i

    for i in prange(m):
        B[i] = A[j+n*i]

cpdef void insert_real(double [::1] A,
                       double [::1] B,
                       Py_ssize_t j,
                       Py_ssize_t n) nogil:

    cdef Py_ssize_t m = B.shape[0]

    cdef Py_ssize_t i

    for i in prange(m):
        A[j+n*i] = B[i]

cpdef void extract_many_real(double [::1] B,
                             double [::1] A,
                             long [::1] indices,
                             Py_ssize_t n) nogil:

    cdef Py_ssize_t m = indices.shape[0]
    cdef Py_ssize_t l = B.shape[0] // m

    cdef Py_ssize_t i, j, k

    for i in prange(l):

        for j in range(m):

            k = indices[j]

            B[j+m*i] = A[k+n*i]

cpdef void insert_many_real(double [::1] A,
                            double [::1] B,
                            long [::1] indices,
                            Py_ssize_t n) nogil:

    cdef Py_ssize_t m = indices.shape[0]
    cdef Py_ssize_t l = B.shape[0] // m

    cdef Py_ssize_t i, j, k

    for i in prange(l):

        for j in range(m):

            k = indices[j]

            A[k+n*i] = B[j+m*i]

cpdef void copy_complex(double complex [::1] B, double complex [::1] A) nogil:

    cdef Py_ssize_t m = A.shape[0]

    cdef Py_ssize_t i

    for i in prange(m):
        B[i] = A[i]

cpdef void scattering_intensity(double [::1] I,
                                double [::1] I_calc,
                                long [::1] inverses,
                                long [::1] i_mask) nogil:

    cdef Py_ssize_t n_hkl = inverses.shape[0]
    cdef Py_ssize_t n_veil = i_mask.shape[0]

    cdef Py_ssize_t i_hkl, i_veil

    for i_hkl in prange(n_hkl):

        I[i_hkl] = I_calc[inverses[i_hkl]]

    for i_veil in prange(n_veil):

        i_hkl = i_mask[i_veil]

        I[i_hkl] = 0

cpdef void unmask_intensity(double [::1] I_calc,
                            double [::1] I,
                            long [::1] i_unmask) nogil:

    cdef Py_ssize_t n_hkl = I_calc.shape[0]

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        I_calc[i_hkl] = I[i_unmask[i_hkl]]

cpdef (double, double) reduced_chi_square(double [::1] calc,
                                          double [::1] exp,
                                          double [::1] inv_error_sq) nogil:

    cdef Py_ssize_t n_hkl = calc.shape[0]

    cdef double chi_sq = 0

    cdef double sum_calc = 0, sum_exp = 0

    cdef double scale, inter_calc, diff

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        inter_calc = calc[i_hkl]*inv_error_sq[i_hkl]

        sum_exp += exp[i_hkl]*inter_calc
        sum_calc += calc[i_hkl]*inter_calc

    scale = sum_exp/sum_calc

    for i_hkl in prange(n_hkl):

        diff = scale*calc[i_hkl]-exp[i_hkl]

        chi_sq += inv_error_sq[i_hkl]*diff*diff

    return chi_sq, scale

cpdef void products(double [::1] V,
                    double Vx,
                    double Vy,
                    double Vz,
                    Py_ssize_t p) nogil:

    cdef Py_ssize_t i, j, u, v, w

    j = 0
    for i in range(p+1):
        for w in range(i+1):
            for v in range(i+1):
                for u in range(i+1):
                    if (u+v+w == i):
                        V[j] = Vx**u*Vy**v*Vz**w
                        j += 1

cpdef void products_mol(double [:,::1] V,
                        double [::1] Vx,
                        double [::1] Vy,
                        double [::1] Vz,
                        Py_ssize_t p) nogil:

    cdef Py_ssize_t n = V.shape[1]

    cdef Py_ssize_t i, j, u, v, w

    j = 0
    for i in range(p+1):
        for w in range(i+1):
            for v in range(i+1):
                for u in range(i+1):
                    if (u+v+w == i):
                        for k in range(n):
                            V[j,k] = Vx[k]**u*Vy[k]**v*Vz[k]**w
                        j += 1

cpdef void magnetic_intensity(double [::1] I,
                              double [::1] Qx_norm,
                              double [::1] Qy_norm,
                              double [::1] Qz_norm,
                              double complex [::1] Fx,
                              double complex [::1] Fy,
                              double complex [::1] Fz,
                              Py_ssize_t n_xyz) nogil:

    cdef Py_ssize_t n_hkl = I.shape[0]

    cdef double complex Q_norm_dot_F, Fx_perp, Fy_perp, Fz_perp

    cdef double Fx_perp_real, Fy_perp_real, Fz_perp_real
    cdef double Fx_perp_imag, Fy_perp_imag, Fz_perp_imag

    cdef double factor = 1./n_xyz

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        Q_norm_dot_F = Qx_norm[i_hkl]*Fx[i_hkl]\
                     + Qy_norm[i_hkl]*Fy[i_hkl]\
                     + Qz_norm[i_hkl]*Fz[i_hkl]

        Fx_perp = Fx[i_hkl]-Q_norm_dot_F*Qx_norm[i_hkl]
        Fy_perp = Fy[i_hkl]-Q_norm_dot_F*Qy_norm[i_hkl]
        Fz_perp = Fz[i_hkl]-Q_norm_dot_F*Qz_norm[i_hkl]

        Fx_perp_real = Fx_perp.real
        Fy_perp_real = Fy_perp.real
        Fz_perp_real = Fz_perp.real

        Fx_perp_imag = Fx_perp.imag
        Fy_perp_imag = Fy_perp.imag
        Fz_perp_imag = Fz_perp.imag

        I[i_hkl] = (Fx_perp_real*Fx_perp_real+Fx_perp_imag*Fx_perp_imag\
                 +  Fy_perp_real*Fy_perp_real+Fy_perp_imag*Fy_perp_imag\
                 +  Fz_perp_real*Fz_perp_real+Fz_perp_imag*Fz_perp_imag)*factor

cpdef void occupational_intensity(double [::1] I,
                                  double complex [::1] F,
                                  Py_ssize_t n_xyz) nogil:

    cdef Py_ssize_t n_hkl = F.shape[0]

    cdef double factor = 1./n_xyz

    cdef double F_perp_real, F_perp_imag

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        F_perp_real = F[i_hkl].real
        F_perp_imag = F[i_hkl].imag

        I[i_hkl] = (F_perp_real*F_perp_real+F_perp_imag*F_perp_imag)*factor

cpdef void displacive_intensity(double [::1] I,
                                double complex [::1] F,
                                double complex [::1] F_nuc,
                                long [::1] bragg,
                                Py_ssize_t n_xyz) nogil:

    cdef Py_ssize_t n_hkl = F.shape[0]

    cdef Py_ssize_t n_nuc = F_nuc.shape[0]

    cdef double factor = 1./n_xyz

    cdef double F_perp_real, F_perp_imag

    cdef Py_ssize_t i_nuc, i_hkl

    for i_nuc in prange(n_nuc):

        F[bragg[i_nuc]] = F_nuc[i_nuc]-F[bragg[i_nuc]]

    for i_hkl in prange(n_hkl):

        F_perp_real = F[i_hkl].real
        F_perp_imag = F[i_hkl].imag

        I[i_hkl] = (F_perp_real*F_perp_real+F_perp_imag*F_perp_imag)*factor

cpdef void update_spin(double complex [::1] Sx_k_cand,
                       double complex [::1] Sy_k_cand,
                       double complex [::1] Sz_k_cand,
                       double Sx_cand,
                       double Sy_cand,
                       double Sz_cand,
                       double complex [::1] Sx_k_orig,
                       double complex [::1] Sy_k_orig,
                       double complex [::1] Sz_k_orig,
                       double Sx_orig,
                       double Sy_orig,
                       double Sz_orig,
                       double complex [::1] space_factor,
                       Py_ssize_t i,
                       Py_ssize_t nu,
                       Py_ssize_t nv,
                       Py_ssize_t nw,
                       Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t w = i // n_atm % nw
    cdef Py_ssize_t v = i // n_atm // nw % nv
    cdef Py_ssize_t u = i // n_atm // nw // nv % nu

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef double dSx = Sx_cand-Sx_orig
    cdef double dSy = Sy_cand-Sy_orig
    cdef double dSz = Sz_cand-Sz_orig

    cdef Py_ssize_t i_uvw

    for iu in prange(nu):

        i_u = nv*iu
        i_f = mv*(u+iu*nu)

        for iv in range(nv):

            i_uv = nw*(iv+i_u)
            i_fa = mw*(v+iv*nv+i_f)

            for iw in range(nw):

                i_uvw = iw+i_uv
                i_fac = w+iw*nw+i_fa

                Sx_k_cand[i_uvw] = Sx_k_orig[i_uvw]+dSx*space_factor[i_fac]
                Sy_k_cand[i_uvw] = Sy_k_orig[i_uvw]+dSy*space_factor[i_fac]
                Sz_k_cand[i_uvw] = Sz_k_orig[i_uvw]+dSz*space_factor[i_fac]

cpdef void update_composition(double complex [::1] A_k_cand,
                              double A_cand,
                              double complex [::1] A_k_orig,
                              double A_orig,
                              double complex [::1] space_factor,
                              Py_ssize_t i,
                              Py_ssize_t nu,
                              Py_ssize_t nv,
                              Py_ssize_t nw,
                              Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t w = i // n_atm % nw
    cdef Py_ssize_t v = i // n_atm // nw % nv
    cdef Py_ssize_t u = i // n_atm // nw // nv % nu

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef double dA = A_cand-A_orig

    cdef Py_ssize_t i_uvw

    for iu in prange(nu):

        i_u = nv*iu
        i_f = mv*(u+iu*nu)

        for iv in range(nv):

            i_uv = nw*(iv+i_u)
            i_fa = mw*(v+iv*nv+i_f)

            for iw in range(nw):

                i_uvw = iw+i_uv
                i_fac = w+iw*nw+i_fa

                A_k_cand[i_uvw] = A_k_orig[i_uvw]+dA*space_factor[i_fac]

cpdef void update_composition_molecule(double complex [::1] A_k_cand,
                                       double [::1] A_cand,
                                       double complex [::1] A_k_orig,
                                       double [::1] A_orig,
                                       double complex [::1] space_factor,
                                       long [::1] i_atm,
                                       Py_ssize_t nu,
                                       Py_ssize_t nv,
                                       Py_ssize_t nw,
                                       Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_ind = i_atm.shape[0]

    cdef Py_ssize_t u, v, w

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef double dA

    cdef Py_ssize_t i_uvw, i_ind, i

    for i_ind in range(n_ind):

        i = i_atm[i_ind]

        dA = A_cand[i_ind]-A_orig[i_ind]

        w = i // n_atm % nw
        v = i // n_atm // nw % nv
        u = i // n_atm // nw // nv % nu

        for iu in prange(nu):

            i_u = nv*iu
            i_f = mv*(u+iu*nu)

            for iv in range(nv):

                i_uv = nw*(iv+i_u)
                i_fa = mw*(v+iv*nv+i_f)

                for iw in range(nw):

                    i_uvw = iw+i_uv
                    i_fac = w+iw*nw+i_fa

                    A_k_cand[i_ind+n_ind*i_uvw] = A_k_orig[i_ind+n_ind*i_uvw]\
                                                + dA*space_factor[i_fac]

cpdef void update_expansion(double complex [::1] U_k_cand,
                            double [::1] U_cand,
                            double complex [::1] U_k_orig,
                            double [::1] U_orig,
                            double complex [::1] space_factor,
                            Py_ssize_t i,
                            Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_prod = U_cand.shape[0]

    cdef Py_ssize_t w = i // n_atm % nw
    cdef Py_ssize_t v = i // n_atm // nw % nv
    cdef Py_ssize_t u = i // n_atm // nw // nv % nu

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU

    cdef Py_ssize_t i_prod, i_uvw

    for i_prod in range(n_prod):

        dU = U_cand[i_prod]-U_orig[i_prod]

        for iu in prange(nu):

            i_u = nv*iu
            i_f = mv*(u+iu*nu)

            for iv in range(nv):

                i_uv = nw*(iv+i_u)
                i_fa = mw*(v+iv*nv+i_f)

                for iw in range(nw):

                    i_uvw = iw+i_uv
                    i_fac = w+iw*nw+i_fa

                    iU = i_uvw+n_uvw*i_prod

                    U_k_cand[iU] = U_k_orig[iU]+dU*space_factor[i_fac]

cpdef void update_expansion_molecule(double complex [::1] U_k_cand,
                                     double [::1] U_cand,
                                     double complex [::1] U_k_orig,
                                     double [::1] U_orig,
                                     double complex [::1] space_factor,
                                     long [::1] i_atm,
                                     Py_ssize_t nu,
                                     Py_ssize_t nv,
                                     Py_ssize_t nw,
                                     Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_ind = i_atm.shape[0]

    cdef Py_ssize_t n_prod = U_cand.shape[0] // n_ind

    cdef Py_ssize_t u, v, w

    cdef Py_ssize_t i_fac, i_ind, i

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU

    cdef Py_ssize_t i_prod, i_uvw

    for i_ind in range(n_ind):

        i = i_atm[i_ind]

        w = i // n_atm % nw
        v = i // n_atm // nw % nv
        u = i // n_atm // nw // nv % nu

        for i_prod in range(n_prod):

            dU = U_cand[i_ind+n_ind*i_prod]-U_orig[i_ind+n_ind*i_prod]

            for iu in prange(nu):

                i_u = nv*iu
                i_f = mv*(u+iu*nu)

                for iv in range(nv):

                    i_uv = nw*(iv+i_u)
                    i_fa = mw*(v+iv*nv+i_f)

                    for iw in range(nw):

                        i_uvw = iw+i_uv
                        i_fac = w+iw*nw+i_fa

                        iU = i_uvw+n_uvw*i_prod

                        U_k_cand[i_ind+n_ind*iU] = U_k_orig[i_ind+n_ind*iU]\
                                                 + dU*space_factor[i_fac]

cpdef void update_relaxation(double complex [::1] A_k_cand,
                             double A_cand,
                             double complex [::1] A_k_orig,
                             double A_orig,
                             double [::1] U,
                             double complex [::1] space_factor,
                             Py_ssize_t i,
                             Py_ssize_t nu,
                             Py_ssize_t nv,
                             Py_ssize_t nw,
                             Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_prod = U.shape[0]

    cdef Py_ssize_t w = i // n_atm % nw
    cdef Py_ssize_t v = i // n_atm // nw % nv
    cdef Py_ssize_t u = i // n_atm // nw // nv % nu

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU

    cdef double dA = A_cand-A_orig

    cdef Py_ssize_t i_prod, i_uvw

    for i_prod in range(n_prod):

        dU = dA*U[i_prod]

        for iu in prange(nu):

            i_u = nv*iu
            i_f = mv*(u+iu*nu)

            for iv in range(nv):

                i_uv = nw*(iv+i_u)
                i_fa = mw*(v+iv*nv+i_f)

                for iw in range(nw):

                    i_uvw = iw+i_uv
                    i_fac = w+iw*nw+i_fa

                    iU = i_uvw+n_uvw*i_prod

                    A_k_cand[iU] = A_k_orig[iU]+dU*space_factor[i_fac]

cpdef void update_relaxation_molecule(double complex [::1] A_k_cand,
                                      double [::1] A_cand,
                                      double complex [::1] A_k_orig,
                                      double [::1] A_orig,
                                      double [::1] U,
                                      double complex [::1] space_factor,
                                      long [::1] i_atm,
                                      Py_ssize_t nu,
                                      Py_ssize_t nv,
                                      Py_ssize_t nw,
                                      Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_ind = i_atm.shape[0]

    cdef Py_ssize_t n_prod = U.shape[0] // n_ind

    cdef Py_ssize_t u, v, w

    cdef Py_ssize_t i_fac, i_ind, i

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU, dA

    cdef Py_ssize_t i_prod, i_uvw

    for i_ind in range(n_ind):

        i = i_atm[i_ind]

        w = i // n_atm % nw
        v = i // n_atm // nw % nv
        u = i // n_atm // nw // nv % nu

        dA = A_cand[i_ind]-A_orig[i_ind]

        for i_prod in range(n_prod):

            dU = dA*U[i_ind+n_ind*i_prod]

            for iu in prange(nu):

                i_u = nv*iu
                i_f = mv*(u+iu*nu)

                for iv in range(nv):

                    i_uv = nw*(iv+i_u)
                    i_fa = mw*(v+iv*nv+i_f)

                    for iw in range(nw):

                        i_uvw = iw+i_uv
                        i_fac = w+iw*nw+i_fa

                        iU = i_uvw+n_uvw*i_prod

                        A_k_cand[i_ind+n_ind*iU] = A_k_orig[i_ind+n_ind*iU]\
                                                 + dU*space_factor[i_fac]

cpdef void update_extension(double complex [::1] U_k_cand,
                            double complex [::1] A_k_cand,
                            double [::1] U_cand,
                            double complex [::1] U_k_orig,
                            double complex [::1] A_k_orig,
                            double [::1] U_orig,
                            double A,
                            double complex [::1] space_factor,
                            Py_ssize_t i,
                            Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_prod = U_cand.shape[0]

    cdef Py_ssize_t w = i // n_atm % nw
    cdef Py_ssize_t v = i // n_atm // nw % nv
    cdef Py_ssize_t u = i // n_atm // nw // nv % nu

    cdef Py_ssize_t i_fac

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU, dA

    cdef Py_ssize_t i_prod, i_uvw

    for i_prod in range(n_prod):

        dU = U_cand[i_prod]-U_orig[i_prod]
        dA = A*dU

        for iu in prange(nu):

            i_u = nv*iu
            i_f = mv*(u+iu*nu)

            for iv in range(nv):

                i_uv = nw*(iv+i_u)
                i_fa = mw*(v+iv*nv+i_f)

                for iw in range(nw):

                    i_uvw = iw+i_uv
                    i_fac = w+iw*nw+i_fa

                    iU = i_uvw+n_uvw*i_prod

                    U_k_cand[iU] = U_k_orig[iU]+dU*space_factor[i_fac]
                    A_k_cand[iU] = A_k_orig[iU]+dA*space_factor[i_fac]

cpdef void update_extension_mol(double complex [::1] U_k_cand,
                                double complex [::1] A_k_cand,
                                double [::1] U_cand,
                                double complex [::1] U_k_orig,
                                double complex [::1] A_k_orig,
                                double [::1] U_orig,
                                double [::1] A_orig,
                                double complex [::1] space_factor,
                                long [::1] i_atm,
                                Py_ssize_t nu,
                                Py_ssize_t nv,
                                Py_ssize_t nw,
                                Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_ind = i_atm.shape[0]

    cdef Py_ssize_t n_prod = U_cand.shape[0] // n_ind

    cdef Py_ssize_t u, v, w

    cdef Py_ssize_t i_fac, i_ind, i

    cdef Py_ssize_t n_uvw = nu*nv*nw

    cdef Py_ssize_t mv = nv**2
    cdef Py_ssize_t mw = nw**2

    cdef Py_ssize_t iu, iv, iw

    cdef Py_ssize_t i_u, i_uv
    cdef Py_ssize_t i_f, i_fa

    cdef Py_ssize_t iU

    cdef double dU, A

    cdef Py_ssize_t i_prod, i_uvw

    for i_ind in range(n_ind):

        i = i_atm[i_ind]

        w = i // n_atm % nw
        v = i // n_atm // nw % nv
        u = i // n_atm // nw // nv % nu

        A = A_orig[i_ind]

        for i_prod in range(n_prod):

            dU = U_cand[i_ind+n_ind*i_prod]-U_orig[i_ind+n_ind*i_prod]
            dA = A*dU

            for iu in prange(nu):

                i_u = nv*iu
                i_f = mv*(u+iu*nu)

                for iv in range(nv):

                    i_uv = nw*(iv+i_u)
                    i_fa = mw*(v+iv*nv+i_f)

                    for iw in range(nw):

                        i_uvw = iw+i_uv
                        i_fac = w+iw*nw+i_fa

                        iU = i_uvw+n_uvw*i_prod

                        U_k_cand[i_ind+n_ind*iU] = U_k_orig[i_ind+n_ind*iU]\
                                                 + dU*space_factor[i_fac]
                        A_k_cand[i_ind+n_ind*iU] = A_k_orig[i_ind+n_ind*iU]\
                                                 + dA*space_factor[i_fac]

cpdef void magnetic_structure_factor(double complex [::1] Fx_cand,
                                     double complex [::1] Fy_cand,
                                     double complex [::1] Fz_cand,
                                     double complex [::1] prod_x_cand,
                                     double complex [::1] prod_y_cand,
                                     double complex [::1] prod_z_cand,
                                     double complex [::1] Sx_k_cand,
                                     double complex [::1] Sy_k_cand,
                                     double complex [::1] Sz_k_cand,
                                     double complex [::1] Fx_orig,
                                     double complex [::1] Fy_orig,
                                     double complex [::1] Fz_orig,
                                     double complex [::1] prod_x_orig,
                                     double complex [::1] prod_y_orig,
                                     double complex [::1] prod_z_orig,
                                     double complex [::1] factors,
                                     Py_ssize_t j,
                                     long [::1] i_dft,
                                     Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = Fx_cand.shape[0]

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        prod_x_cand[i_hkl] = factors[j+n_atm*i_hkl]*Sx_k_cand[i_dft[i_hkl]]
        prod_y_cand[i_hkl] = factors[j+n_atm*i_hkl]*Sy_k_cand[i_dft[i_hkl]]
        prod_z_cand[i_hkl] = factors[j+n_atm*i_hkl]*Sz_k_cand[i_dft[i_hkl]]

        Fx_cand[i_hkl] = Fx_orig[i_hkl]+prod_x_cand[i_hkl]-prod_x_orig[i_hkl]
        Fy_cand[i_hkl] = Fy_orig[i_hkl]+prod_y_cand[i_hkl]-prod_y_orig[i_hkl]
        Fz_cand[i_hkl] = Fz_orig[i_hkl]+prod_z_cand[i_hkl]-prod_z_orig[i_hkl]

cpdef void occupational_structure_factor(double complex [::1] F_cand,
                                         double complex [::1] prod_cand,
                                         double complex [::1] A_k_cand,
                                         double complex [::1] F_orig,
                                         double complex [::1] prod_orig,
                                         double complex [::1] factors,
                                         Py_ssize_t j,
                                         long [::1] i_dft,
                                         Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = F_cand.shape[0]

    cdef Py_ssize_t i_hkl

    for i_hkl in prange(n_hkl):

        prod_cand[i_hkl] = factors[j+n_atm*i_hkl]*A_k_cand[i_dft[i_hkl]]

        F_cand[i_hkl] = F_orig[i_hkl]+prod_cand[i_hkl]-prod_orig[i_hkl]

cpdef void displacive_structure_factor(double complex [::1] F_cand,
                                       double complex [::1] F_nuc_cand,
                                       double complex [::1] prod_cand,
                                       double complex [::1] prod_nuc_cand,
                                       double complex [::1] V_k_cand,
                                       double complex [::1] V_k_nuc_cand,
                                       double complex [::1] U_k_cand,
                                       double complex [::1] F_orig,
                                       double complex [::1] F_nuc_orig,
                                       double complex [::1] prod_orig,
                                       double complex [::1] prod_nuc_orig,
                                       double complex [::1] V_k_orig,
                                       double complex [::1] V_k_nuc_orig,
                                       double complex [::1] U_k_orig,
                                       double [::1] Q_k,
                                       double complex [::1] factors,
                                       double complex [::1] coeffs,
                                       long [::1] even,
                                       long [::1] bragg,
                                       long [::1] i_dft,
                                       Py_ssize_t p,
                                       Py_ssize_t j,
                                       Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = F_cand.shape[0]

    cdef Py_ssize_t n_prod = coeffs.shape[0]
    cdef Py_ssize_t n_even = even.shape[0]
    cdef Py_ssize_t n_nuc = bragg.shape[0]

    cdef Py_ssize_t n_uvw = U_k_cand.shape[0] // n_prod

    cdef Py_ssize_t i_prod, i_even, i_nuc, i_hkl

    for i_hkl in prange(n_hkl):

        V_k_cand[i_hkl] = V_k_orig[i_hkl]+0

        for i_prod in range(n_prod):

            V_k_cand[i_hkl] = coeffs[i_prod]\
                            * (U_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                              -U_k_orig[i_dft[i_hkl]+n_uvw*i_prod])\
                            *  Q_k[i_hkl+n_hkl*i_prod]+V_k_cand[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        V_k_nuc_cand[i_nuc] = V_k_nuc_orig[i_nuc]+0

        for i_even in range(n_even):

            i_prod = even[i_even]

            V_k_nuc_cand[i_nuc] = coeffs[i_prod]\
                                * (U_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                                  -U_k_orig[i_dft[i_hkl]+n_uvw*i_prod])\
                             *  Q_k[i_hkl+n_hkl*i_prod]+V_k_nuc_cand[i_nuc]

    for i_hkl in prange(n_hkl):

        prod_cand[i_hkl] = factors[j+n_atm*i_hkl]*V_k_cand[i_hkl]

        F_cand[i_hkl] = F_orig[i_hkl]+prod_cand[i_hkl]-prod_orig[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        prod_nuc_cand[i_nuc] = factors[j+n_atm*i_hkl]*V_k_nuc_cand[i_nuc]

        F_nuc_cand[i_nuc] = F_nuc_orig[i_nuc]\
                          + prod_nuc_cand[i_nuc]-prod_nuc_orig[i_nuc]

cpdef void displacive_structure_factor_mol(double complex [::1] F_cand,
                                           double complex [::1] F_nuc_cand,
                                           double complex [::1] prod_cand,
                                           double complex [::1] prod_nuc_cand,
                                           double complex [::1] V_k_cand,
                                           double complex [::1] V_k_nuc_cand,
                                           double complex [::1] U_k_cand,
                                           double complex [::1] F_orig,
                                           double complex [::1] F_nuc_orig,
                                           double complex [::1] prod_orig,
                                           double complex [::1] prod_nuc_orig,
                                           double complex [::1] V_k_orig,
                                           double complex [::1] V_k_nuc_orig,
                                           double complex [::1] U_k_orig,
                                           double [::1] Q_k,
                                           double complex [::1] factors,
                                           double complex [::1] coeffs,
                                           long [::1] even,
                                           long [::1] bragg,
                                           long [::1] i_dft,
                                           Py_ssize_t p,
                                           long [::1] j_atm,
                                           Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = F_cand.shape[0]

    cdef Py_ssize_t n_ind = j_atm.shape[0]

    cdef Py_ssize_t i_ind, j

    cdef Py_ssize_t n_prod = coeffs.shape[0]
    cdef Py_ssize_t n_even = even.shape[0]
    cdef Py_ssize_t n_nuc = bragg.shape[0]

    cdef Py_ssize_t n_uvw = U_k_cand.shape[0] // n_prod // n_ind

    cdef Py_ssize_t i_prod, i_even, i_nuc, i_hkl

    for i_hkl in prange(n_hkl):

        for i_ind in range(n_ind):

            V_k_cand[i_ind+n_ind*i_hkl] = V_k_orig[i_ind+n_ind*i_hkl]+0

            for i_prod in range(n_prod):

                V_k_cand[i_ind+n_ind*i_hkl] = coeffs[i_prod]\
                                            * (U_k_cand[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)]\
                                            -  U_k_orig[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)])\
                                            *  Q_k[i_hkl+n_hkl*i_prod]\
                                            +  V_k_cand[i_ind+n_ind*i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        for i_ind in range(n_ind):

            V_k_nuc_cand[i_ind+n_ind*i_nuc] = V_k_nuc_orig[i_ind+n_ind*i_nuc]+0

            for i_even in range(n_even):

                i_prod = even[i_even]

                V_k_nuc_cand[i_ind+n_ind*i_nuc] = coeffs[i_prod]\
                                                * (U_k_cand[i_ind\
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)]\
                                                -  U_k_orig[i_ind
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)])\
                                                *  Q_k[i_hkl+n_hkl*i_prod]\
                                                +  V_k_nuc_cand[i_ind+n_ind*i_nuc]

    for i_hkl in prange(n_hkl):

        F_cand[i_hkl] = F_orig[i_hkl]+0

        for i_ind in range(n_ind):

            j = j_atm[i_ind]

            prod_cand[i_ind+n_ind*i_hkl] = factors[j+n_atm*i_hkl]\
                                         * V_k_cand[i_ind+n_ind*i_hkl]

            F_cand[i_hkl] = prod_cand[i_ind+n_ind*i_hkl]\
                          - prod_orig[i_ind+n_ind*i_hkl]+F_cand[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        F_nuc_cand[i_nuc] = F_nuc_orig[i_nuc]+0

        for i_ind in range(n_ind):

            j = j_atm[i_ind]

            prod_nuc_cand[i_ind+n_ind*i_nuc] = factors[j+n_atm*i_hkl]\
                                             * V_k_nuc_cand[i_ind+n_ind*i_nuc]

            F_nuc_cand[i_nuc] = prod_nuc_cand[i_ind+n_ind*i_nuc]\
                              - prod_nuc_orig[i_ind+n_ind*i_nuc]\
                              + F_nuc_cand[i_nuc]

cpdef void nonmagnetic_structure_factor(double complex [::1] F_cand,
                                        double complex [::1] F_nuc_cand,
                                        double complex [::1] prod_cand,
                                        double complex [::1] prod_nuc_cand,
                                        double complex [::1] V_k_cand,
                                        double complex [::1] V_k_nuc_cand,
                                        double complex [::1] U_k_cand,
                                        double complex [::1] A_k_cand,
                                        double complex [::1] F_orig,
                                        double complex [::1] F_nuc_orig,
                                        double complex [::1] prod_orig,
                                        double complex [::1] prod_nuc_orig,
                                        double complex [::1] V_k_orig,
                                        double complex [::1] V_k_nuc_orig,
                                        double complex [::1] U_k_orig,
                                        double complex [::1] A_k_orig,
                                        double [::1] Q_k,
                                        double complex [::1] factors,
                                        double complex [::1] coeffs,
                                        long [::1] even,
                                        long [::1] bragg,
                                        long [::1] i_dft,
                                        Py_ssize_t p,
                                        Py_ssize_t j,
                                        Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = F_cand.shape[0]

    cdef Py_ssize_t n_prod = coeffs.shape[0]
    cdef Py_ssize_t n_even = even.shape[0]
    cdef Py_ssize_t n_nuc = bragg.shape[0]

    cdef Py_ssize_t n_uvw = U_k_cand.shape[0] // n_prod

    cdef Py_ssize_t i_prod, i_even, i_nuc, i_hkl

    for i_hkl in prange(n_hkl):

        V_k_cand[i_hkl] = V_k_orig[i_hkl]+0

        for i_prod in range(n_prod):

            V_k_cand[i_hkl] = coeffs[i_prod]\
                            * (U_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                              -U_k_orig[i_dft[i_hkl]+n_uvw*i_prod]\
                              +A_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                              -A_k_orig[i_dft[i_hkl]+n_uvw*i_prod])\
                            *  Q_k[i_hkl+n_hkl*i_prod]+V_k_cand[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        V_k_nuc_cand[i_nuc] = V_k_nuc_orig[i_nuc]+0

        for i_even in range(n_even):

            i_prod = even[i_even]

            V_k_nuc_cand[i_nuc] = coeffs[i_prod]\
                                * (U_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                                  -U_k_orig[i_dft[i_hkl]+n_uvw*i_prod]\
                                  +A_k_cand[i_dft[i_hkl]+n_uvw*i_prod]\
                                  -A_k_orig[i_dft[i_hkl]+n_uvw*i_prod])\
                                *  Q_k[i_hkl+n_hkl*i_prod]+V_k_nuc_cand[i_nuc]

    for i_hkl in prange(n_hkl):

        prod_cand[i_hkl] = factors[j+n_atm*i_hkl]*V_k_cand[i_hkl]

        F_cand[i_hkl] = F_orig[i_hkl]+prod_cand[i_hkl]-prod_orig[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        prod_nuc_cand[i_nuc] = factors[j+n_atm*i_hkl]*V_k_nuc_cand[i_nuc]

        F_nuc_cand[i_nuc] = F_nuc_orig[i_nuc]\
                          + prod_nuc_cand[i_nuc]-prod_nuc_orig[i_nuc]

cpdef void nonmagnetic_structure_factor_mol(double complex [::1] F_cand,
                                            double complex [::1] F_nuc_cand,
                                            double complex [::1] prod_cand,
                                            double complex [::1] prod_nuc_cand,
                                            double complex [::1] V_k_cand,
                                            double complex [::1] V_k_nuc_cand,
                                            double complex [::1] U_k_cand,
                                            double complex [::1] A_k_cand,
                                            double complex [::1] F_orig,
                                            double complex [::1] F_nuc_orig,
                                            double complex [::1] prod_orig,
                                            double complex [::1] prod_nuc_orig,
                                            double complex [::1] V_k_orig,
                                            double complex [::1] V_k_nuc_orig,
                                            double complex [::1] U_k_orig,
                                            double complex [::1] A_k_orig,
                                            double [::1] Q_k,
                                            double complex [::1] factors,
                                            double complex [::1] coeffs,
                                            long [::1] even,
                                            long [::1] bragg,
                                            long [::1] i_dft,
                                            Py_ssize_t p,
                                            long [::1] j_atm,
                                            Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t n_hkl = F_cand.shape[0]

    cdef Py_ssize_t n_ind = j_atm.shape[0]

    cdef Py_ssize_t i_ind, j

    cdef Py_ssize_t n_prod = coeffs.shape[0]
    cdef Py_ssize_t n_even = even.shape[0]
    cdef Py_ssize_t n_nuc = bragg.shape[0]

    cdef Py_ssize_t n_uvw = U_k_cand.shape[0] // n_prod // n_ind

    cdef Py_ssize_t i_prod, i_even, i_nuc, i_hkl

    for i_hkl in prange(n_hkl):

        for i_ind in range(n_ind):

            V_k_cand[i_ind+n_ind*i_hkl] = V_k_orig[i_ind+n_ind*i_hkl]+0

            for i_prod in range(n_prod):

                V_k_cand[i_ind+n_ind*i_hkl] = coeffs[i_prod]\
                                            * (U_k_cand[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)]\
                                            -  U_k_orig[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)]
                                            +  A_k_cand[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)]\
                                            -  A_k_orig[i_ind\
                                                        +n_ind*(i_dft[i_hkl]\
                                                        +n_uvw*i_prod)])\
                                            *  Q_k[i_hkl+n_hkl*i_prod]\
                                            +  V_k_cand[i_ind+n_ind*i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        for i_ind in range(n_ind):

            V_k_nuc_cand[i_ind+n_ind*i_nuc] = V_k_nuc_orig[i_ind+n_ind*i_nuc]+0

            for i_even in range(n_even):

                i_prod = even[i_even]

                V_k_nuc_cand[i_ind+n_ind*i_nuc] = coeffs[i_prod]\
                                                * (U_k_cand[i_ind\
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)]\
                                                -  U_k_orig[i_ind
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)]
                                                +  A_k_cand[i_ind\
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)]\
                                                -  A_k_orig[i_ind
                                                            +n_ind*(
                                                             i_dft[i_hkl]\
                                                            +n_uvw*i_prod)])\
                                                *  Q_k[i_hkl+n_hkl*i_prod]\
                                             +  V_k_nuc_cand[i_ind+n_ind*i_nuc]

    for i_hkl in prange(n_hkl):

        F_cand[i_hkl] = F_orig[i_hkl]+0

        for i_ind in range(n_ind):

            j = j_atm[i_ind]

            prod_cand[i_ind+n_ind*i_hkl] = factors[j+n_atm*i_hkl]\
                                         * V_k_cand[i_ind+n_ind*i_hkl]

            F_cand[i_hkl] = prod_cand[i_ind+n_ind*i_hkl]\
                          - prod_orig[i_ind+n_ind*i_hkl]+F_cand[i_hkl]

    for i_nuc in prange(n_nuc):

        i_hkl = bragg[i_nuc]

        F_nuc_cand[i_nuc] = F_nuc_orig[i_nuc]+0

        for i_ind in range(n_ind):

            j = j_atm[i_ind]

            prod_nuc_cand[i_ind+n_ind*i_nuc] = factors[j+n_atm*i_hkl]\
                                             * V_k_nuc_cand[i_ind+n_ind*i_nuc]

            F_nuc_cand[i_nuc] = prod_nuc_cand[i_ind+n_ind*i_nuc]\
                              - prod_nuc_orig[i_ind+n_ind*i_nuc]\
                              + F_nuc_cand[i_nuc]

cpdef void magnetic(double [::1] Sx,
                    double [::1] Sy,
                    double [::1] Sz,
                    double [::1] Qx_norm,
                    double [::1] Qy_norm,
                    double [::1] Qz_norm,
                    double complex [::1] Sx_k,
                    double complex [::1] Sy_k,
                    double complex [::1] Sz_k,
                    double complex [::1] Sx_k_orig,
                    double complex [::1] Sy_k_orig,
                    double complex [::1] Sz_k_orig,
                    double complex [::1] Sx_k_cand,
                    double complex [::1] Sy_k_cand,
                    double complex [::1] Sz_k_cand,
                    double complex [::1] Fx,
                    double complex [::1] Fy,
                    double complex [::1] Fz,
                    double complex [::1] Fx_orig,
                    double complex [::1] Fy_orig,
                    double complex [::1] Fz_orig,
                    double complex [::1] Fx_cand,
                    double complex [::1] Fy_cand,
                    double complex [::1] Fz_cand,
                    double complex [::1] prod_x,
                    double complex [::1] prod_y,
                    double complex [::1] prod_z,
                    double complex [::1] prod_x_orig,
                    double complex [::1] prod_y_orig,
                    double complex [::1] prod_z_orig,
                    double complex [::1] prod_x_cand,
                    double complex [::1] prod_y_cand,
                    double complex [::1] prod_z_cand,
                    double complex [::1] space_factor,
                    double complex [::1] factors,
                    double [::1] moment,
                    double [::1] I_calc,
                    double [::1] I_exp,
                    double [::1] inv_sigma_sq,
                    double [::1] I_raw,
                    double [::1] I_flat,
                    double [::1] I_ref,
                    double [::1] v_inv,
                    double [::1] a_filt,
                    double [::1] b_filt,
                    double [::1] c_filt,
                    double [::1] d_filt,
                    double [::1] e_filt,
                    double [::1] f_filt,
                    double [::1] g_filt,
                    double [::1] h_filt,
                    double [::1] i_filt,
                    long [::1] boxes,
                    long [::1] i_dft,
                    long [::1] inverses,
                    long [::1] i_mask,
                    long [::1] i_unmask,
                    list acc_moves,
                    list acc_temps,
                    list rej_moves,
                    list rej_temps,
                    list chi_sq,
                    list energy,
                    list temperature,
                    list scale,
                    double constant,
                    bint fixed,
                    bint heisenberg,
                    Py_ssize_t nh,
                    Py_ssize_t nk,
                    Py_ssize_t nl,
                    Py_ssize_t nu,
                    Py_ssize_t nv,
                    Py_ssize_t nw,
                    Py_ssize_t n_atm,
                    Py_ssize_t n,
                    Py_ssize_t N):

    cdef double temp, inv_temp

    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor

    cdef double Sx_orig, Sy_orig, Sz_orig, Sx_cand, Sy_cand, Sz_cand, mu

    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)

    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef Py_ssize_t i, j, s

    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]

    with nogil:

        for s in range(N):

            temp = exp(log(temp)-constant)

            inv_temp = 1/temp

            Sx_orig, Sy_orig, Sz_orig, i = original_vector(Sx, Sy, Sz)

            j = i % n_atm

            mu = moment[j]

            Sx_cand, Sy_cand, Sz_cand = candidate_moment(Sx_orig,
                                                         Sy_orig,
                                                         Sz_orig,
                                                         mu)

            copy_complex(Fx_orig, Fx)
            copy_complex(Fy_orig, Fy)
            copy_complex(Fz_orig, Fz)

            extract_complex(prod_x_orig, prod_x, j, n_atm)
            extract_complex(prod_y_orig, prod_y, j, n_atm)
            extract_complex(prod_z_orig, prod_z, j, n_atm)

            extract_complex(Sx_k_orig, Sx_k, j, n_atm)
            extract_complex(Sy_k_orig, Sy_k, j, n_atm)
            extract_complex(Sz_k_orig, Sz_k, j, n_atm)

            Sx[i], Sy[i], Sz[i] = Sx_cand, Sy_cand, Sz_cand

            update_spin(Sx_k_cand,
                        Sy_k_cand,
                        Sz_k_cand,
                        Sx_cand,
                        Sy_cand,
                        Sz_cand,
                        Sx_k_orig,
                        Sy_k_orig,
                        Sz_k_orig,
                        Sx_orig,
                        Sy_orig,
                        Sz_orig,
                        space_factor,
                        i,
                        nu,
                        nv,
                        nw,
                        n_atm)

            insert_complex(Sx_k, Sx_k_cand, j, n_atm)
            insert_complex(Sy_k, Sy_k_cand, j, n_atm)
            insert_complex(Sz_k, Sz_k_cand, j, n_atm)

            magnetic_structure_factor(Fx_cand,
                                      Fy_cand,
                                      Fz_cand,
                                      prod_x_cand,
                                      prod_y_cand,
                                      prod_z_cand,
                                      Sx_k_cand,
                                      Sy_k_cand,
                                      Sz_k_cand,
                                      Fx_orig,
                                      Fy_orig,
                                      Fz_orig,
                                      prod_x_orig,
                                      prod_y_orig,
                                      prod_z_orig,
                                      factors,
                                      j,
                                      i_dft,
                                      n_atm)

            insert_complex(prod_x, prod_x_cand, j, n_atm)
            insert_complex(prod_y, prod_y_cand, j, n_atm)
            insert_complex(prod_z, prod_z_cand, j, n_atm)

            copy_complex(Fx, Fx_cand)
            copy_complex(Fy, Fy_cand)
            copy_complex(Fz, Fz_cand)

            magnetic_intensity(I_calc,
                               Qx_norm,
                               Qy_norm,
                               Qz_norm,
                               Fx_cand,
                               Fy_cand,
                               Fz_cand,
                               n)

            scattering_intensity(I_raw, I_calc, inverses, i_mask)

            filters.filtering(I_flat,
                              I_raw,
                              v_inv,
                              boxes,
                              a_filt,
                              b_filt,
                              c_filt,
                              d_filt,
                              e_filt,
                              f_filt,
                              g_filt,
                              h_filt,
                              i_filt,
                              nh,
                              nk,
                              nl)

            unmask_intensity(I_ref, I_flat, i_unmask)

            chi_sq_cand, scale_factor = reduced_chi_square(I_ref,
                                                           I_exp,
                                                           inv_sigma_sq)

            delta_chi_sq = chi_sq_cand-chi_sq_orig

            if (delta_chi_sq > 0):
                if (random_uniform() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand

                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    Sx[i], Sy[i], Sz[i] = Sx_orig, Sy_orig, Sz_orig

                    insert_complex(Sx_k, Sx_k_orig, j, n_atm)
                    insert_complex(Sy_k, Sy_k_orig, j, n_atm)
                    insert_complex(Sz_k, Sz_k_orig, j, n_atm)

                    insert_complex(prod_x, prod_x_orig, j, n_atm)
                    insert_complex(prod_y, prod_y_orig, j, n_atm)
                    insert_complex(prod_z, prod_z_orig, j, n_atm)

                    copy_complex(Fx, Fx_orig)
                    copy_complex(Fy, Fy_orig)
                    copy_complex(Fz, Fz_orig)

                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand

                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp

            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor

    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)

cpdef void occupational(double [::1] A_r,
                        double complex [::1] A_k,
                        double complex [::1] A_k_orig,
                        double complex [::1] A_k_cand,
                        double complex [::1] F,
                        double complex [::1] F_orig,
                        double complex [::1] F_cand,
                        double complex [::1] prod,
                        double complex [::1] prod_orig,
                        double complex [::1] prod_cand,
                        double complex [::1] space_factor,
                        double complex [::1] factors,
                        double [::1] occupancy,
                        double [::1] I_calc,
                        double [::1] I_exp,
                        double [::1] inv_sigma_sq,
                        double [::1] I_raw,
                        double [::1] I_flat,
                        double [::1] I_ref,
                        double [::1] v_inv,
                        double [::1] a_filt,
                        double [::1] b_filt,
                        double [::1] c_filt,
                        double [::1] d_filt,
                        double [::1] e_filt,
                        double [::1] f_filt,
                        double [::1] g_filt,
                        double [::1] h_filt,
                        double [::1] i_filt,
                        long [::1] boxes,
                        long [::1] i_dft,
                        long [::1] inverses,
                        long [::1] i_mask,
                        long [::1] i_unmask,
                        list acc_moves,
                        list acc_temps,
                        list rej_moves,
                        list rej_temps,
                        list chi_sq,
                        list energy,
                        list temperature,
                        list scale,
                        double constant,
                        bint fixed,
                        Py_ssize_t nh,
                        Py_ssize_t nk,
                        Py_ssize_t nl,
                        Py_ssize_t nu,
                        Py_ssize_t nv,
                        Py_ssize_t nw,
                        Py_ssize_t n_atm,
                        Py_ssize_t n,
                        Py_ssize_t N):

    cdef double temp, inv_temp

    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor

    cdef double A_r_orig, A_r_cand, occ

    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)

    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef Py_ssize_t i, j, s

    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]

    with nogil:

        for s in range(N):

            temp = exp(log(temp)-constant)

            inv_temp = 1/temp

            A_r_orig, i = original_scalar(A_r)

            j = i % n_atm

            occ = occupancy[j]

            A_r_cand = candidate_composition(A_r_orig, occ)

            copy_complex(F_orig, F)

            extract_complex(prod_orig, prod, j, n_atm)

            extract_complex(A_k_orig, A_k, j, n_atm)

            A_r[i] = A_r_cand

            update_composition(A_k_cand,
                               A_r_cand,
                               A_k_orig,
                               A_r_orig,
                               space_factor,
                               i,
                               nu,
                               nv,
                               nw,
                               n_atm)

            insert_complex(A_k, A_k_cand, j, n_atm)

            occupational_structure_factor(F_cand,
                                          prod_cand,
                                          A_k_cand,
                                          F_orig,
                                          prod_orig,
                                          factors,
                                          j,
                                          i_dft,
                                          n_atm)

            insert_complex(prod, prod_cand, j, n_atm)

            copy_complex(F, F_cand)

            occupational_intensity(I_calc, F_cand, n)

            scattering_intensity(I_raw, I_calc, inverses, i_mask)

            filters.filtering(I_flat,
                              I_raw,
                              v_inv,
                              boxes,
                              a_filt,
                              b_filt,
                              c_filt,
                              d_filt,
                              e_filt,
                              f_filt,
                              g_filt,
                              h_filt,
                              i_filt,
                              nh,
                              nk,
                              nl)

            unmask_intensity(I_ref, I_flat, i_unmask)

            chi_sq_cand, scale_factor = reduced_chi_square(I_ref,
                                                           I_exp,
                                                           inv_sigma_sq)

            delta_chi_sq = chi_sq_cand-chi_sq_orig

            if (delta_chi_sq > 0):
                if (random_uniform() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand

                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    A_r[i] = A_r_orig

                    insert_complex(A_k, A_k_orig, j, n_atm)

                    insert_complex(prod, prod_orig, j, n_atm)

                    copy_complex(F, F_orig)

                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand

                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp

            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor

    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)

cpdef void displacive(double [::1] Ux,
                      double [::1] Uy,
                      double [::1] Uz,
                      double [::1] U_r,
                      double [::1] U_r_orig,
                      double [::1] U_r_cand,
                      double complex [::1] U_k,
                      double complex [::1] U_k_orig,
                      double complex [::1] U_k_cand,
                      double complex [::1] V_k,
                      double complex [::1] V_k_nuc,
                      double complex [::1] V_k_orig,
                      double complex [::1] V_k_nuc_orig,
                      double complex [::1] V_k_cand,
                      double complex [::1] V_k_nuc_cand,
                      double complex [::1] F,
                      double complex [::1] F_nuc,
                      double complex [::1] F_orig,
                      double complex [::1] F_nuc_orig,
                      double complex [::1] F_cand,
                      double complex [::1] F_nuc_cand,
                      double complex [::1] prod,
                      double complex [::1] prod_nuc,
                      double complex [::1] prod_orig,
                      double complex [::1] prod_nuc_orig,
                      double complex [::1] prod_cand,
                      double complex [::1] prod_nuc_cand,
                      double complex [::1] space_factor,
                      double complex [::1] factors,
                      double complex [::1] coeffs,
                      double [::1] Q_k,
                      double [::1] Lxx,
                      double [::1] Lyy,
                      double [::1] Lzz,
                      double [::1] Lyz,
                      double [::1] Lxz,
                      double [::1] Lxy,
                      double [::1] I_calc,
                      double [::1] I_exp,
                      double [::1] inv_sigma_sq,
                      double [::1] I_raw,
                      double [::1] I_flat,
                      double [::1] I_ref,
                      double [::1] v_inv,
                      double [::1] a_filt,
                      double [::1] b_filt,
                      double [::1] c_filt,
                      double [::1] d_filt,
                      double [::1] e_filt,
                      double [::1] f_filt,
                      double [::1] g_filt,
                      double [::1] h_filt,
                      double [::1] i_filt,
                      long [::1] bragg,
                      long [::1] even,
                      long [::1] boxes,
                      long [::1] i_dft,
                      long [::1] inverses,
                      long [::1] i_mask,
                      long [::1] i_unmask,
                      list acc_moves,
                      list acc_temps,
                      list rej_moves,
                      list rej_temps,
                      list chi_sq,
                      list energy,
                      list temperature,
                      list scale,
                      double constant,
                      bint fixed,
                      bint isotropic,
                      Py_ssize_t p,
                      Py_ssize_t nh,
                      Py_ssize_t nk,
                      Py_ssize_t nl,
                      Py_ssize_t nu,
                      Py_ssize_t nv,
                      Py_ssize_t nw,
                      Py_ssize_t n_atm,
                      Py_ssize_t n,
                      Py_ssize_t N):

    cdef double temp, inv_temp

    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor

    cdef double Ux_orig, Uy_orig, Uz_orig, Ux_cand, Uy_cand, Uz_cand

    cdef double lxx, lyy, lzz, lyz, lxz, lxy

    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)

    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef double [::1] result = np.zeros(3, dtype=np.double)

    cdef Py_ssize_t i, j, s

    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]

    with nogil:

        for s in range(N):

            temp = exp(log(temp)-constant)

            inv_temp = 1/temp

            Ux_orig, Uy_orig, Uz_orig, i = original_vector(Ux, Uy, Uz)

            j = i % n_atm

            lxx, lyy, lzz = Lxx[j], Lyy[j], Lzz[j]
            lyz, lxz, lxy = Lyz[j], Lxz[j], Lxy[j]

            Ux_cand, Uy_cand, Uz_cand = candidate_displacement(lxx,
                                                               lyy,
                                                               lzz,
                                                               lyz,
                                                               lxz,
                                                               lxy)

            copy_complex(F_orig, F)
            copy_complex(F_nuc_orig, F_nuc)

            extract_complex(prod_orig, prod, j, n_atm)
            extract_complex(prod_nuc_orig, prod_nuc, j, n_atm)

            extract_complex(V_k_orig, V_k, j, n_atm)
            extract_complex(V_k_nuc_orig, V_k_nuc, j, n_atm)

            Ux[i], Uy[i], Uz[i] = Ux_cand, Uy_cand, Uz_cand

            extract_real(U_r_orig, U_r, i, n)

            extract_complex(U_k_orig, U_k, j, n_atm)

            products(U_r_cand, Ux_cand, Uy_cand, Uz_cand, p)

            insert_real(U_r, U_r_cand, i, n)

            update_expansion(U_k_cand,
                             U_r_cand,
                             U_k_orig,
                             U_r_orig,
                             space_factor,
                             i,
                             nu,
                             nv,
                             nw,
                             n_atm)

            insert_complex(U_k, U_k_cand, j, n_atm)

            displacive_structure_factor(F_cand,
                                        F_nuc_cand,
                                        prod_cand,
                                        prod_nuc_cand,
                                        V_k_cand,
                                        V_k_nuc_cand,
                                        U_k_cand,
                                        F_orig,
                                        F_nuc_orig,
                                        prod_orig,
                                        prod_nuc_orig,
                                        V_k_orig,
                                        V_k_nuc_orig,
                                        U_k_orig,
                                        Q_k,
                                        factors,
                                        coeffs,
                                        even,
                                        bragg,
                                        i_dft,
                                        p,
                                        j,
                                        n_atm)

            insert_complex(V_k, V_k_cand, j, n_atm)
            insert_complex(V_k_nuc, V_k_nuc_cand, j, n_atm)

            insert_complex(prod, prod_cand, j, n_atm)
            insert_complex(prod_nuc, prod_nuc_cand, j, n_atm)

            copy_complex(F, F_cand)
            copy_complex(F_nuc, F_nuc_cand)

            displacive_intensity(I_calc, F_cand, F_nuc_cand, bragg, n)

            scattering_intensity(I_raw, I_calc, inverses, i_mask)

            filters.filtering(I_flat,
                              I_raw,
                              v_inv,
                              boxes,
                              a_filt,
                              b_filt,
                              c_filt,
                              d_filt,
                              e_filt,
                              f_filt,
                              g_filt,
                              h_filt,
                              i_filt,
                              nh,
                              nk,
                              nl)

            unmask_intensity(I_ref, I_flat, i_unmask)

            chi_sq_cand, scale_factor = reduced_chi_square(I_ref,
                                                           I_exp,
                                                           inv_sigma_sq)

            delta_chi_sq = chi_sq_cand-chi_sq_orig

            if (delta_chi_sq > 0):
                if (random_uniform() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand

                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    Ux[i], Uy[i], Uz[i] = Ux_orig, Uy_orig, Uz_orig

                    insert_real(U_r, U_r_orig, i, n)

                    insert_complex(U_k, U_k_orig, j, n_atm)

                    insert_complex(V_k, V_k_orig, j, n_atm)
                    insert_complex(V_k_nuc, V_k_nuc_orig, j, n_atm)

                    insert_complex(prod, prod_orig, j, n_atm)
                    insert_complex(prod_nuc, prod_nuc_orig, j, n_atm)

                    copy_complex(F, F_orig)
                    copy_complex(F_nuc, F_nuc_orig)

                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand

                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp

            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor

    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)