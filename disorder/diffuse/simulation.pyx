#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport M_PI, fabs, log, exp, sqrt
from libc.math cimport sin, cos, tan
from libc.math cimport acos, atan, atan2

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937 gen)

cdef mt19937 gen

cdef uniform_real_distribution[double] dist
cdef uniform_int_distribution[Py_ssize_t] dist_u
cdef uniform_int_distribution[Py_ssize_t] dist_v
cdef uniform_int_distribution[Py_ssize_t] dist_w
cdef uniform_int_distribution[Py_ssize_t] dist_atm
cdef uniform_int_distribution[Py_ssize_t] dist_temp

cdef double MACHINE_EPSILON = np.finfo(float).eps

cdef void initialize_random(Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm,
                            Py_ssize_t n_temp):

    global gen, dist, dist_u, dist_v, dist_w, dist_atm, dist_temp

    gen = mt19937(20)

    dist = uniform_real_distribution[double](0.0,1.0)

    dist_u = uniform_int_distribution[Py_ssize_t](0,nu-1)
    dist_v = uniform_int_distribution[Py_ssize_t](0,nv-1)
    dist_w = uniform_int_distribution[Py_ssize_t](0,nw-1)
    dist_atm = uniform_int_distribution[Py_ssize_t](0,n_atm-1)
    dist_temp = uniform_int_distribution[Py_ssize_t](0,n_temp-1)

cdef bint iszero(double a) nogil:

    cdef double atol = 1e-08

    return fabs(a) <= atol

cdef double random_uniform():

    return dist(gen)

cdef double alpha(double E, double beta):

    return exp(-beta*E)

cdef (Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t) random_original(Py_ssize_t nu,
                                  Py_ssize_t nv,
                                  Py_ssize_t nw,
                                  Py_ssize_t n_atm):

    cdef Py_ssize_t i = dist_u(gen)
    cdef Py_ssize_t j = dist_v(gen)
    cdef Py_ssize_t k = dist_w(gen)
    cdef Py_ssize_t a = dist_atm(gen)

    return i, j, k, a

cdef (double, double, double) random_vector_candidate():

    cdef double theta = 2*M_PI*random_uniform()
    cdef double phi = acos(1-2*random_uniform())

    cdef double sx = sin(phi)*cos(theta)
    cdef double sy = sin(phi)*sin(theta)
    cdef double sz = cos(phi)

    return sx, sy, sz

cdef (double, double, double) random_vector_length_candidate():

    cdef double theta = 2*M_PI*random_uniform()
    cdef double phi = acos(1-2*random_uniform())

    cdef double s = random_uniform()

    cdef double sx = s*sin(phi)*cos(theta)
    cdef double sy = s*sin(phi)*sin(theta)
    cdef double sz = s*cos(phi)

    return sx, sy, sz

cdef (double, double, double) ising_vector_candidate(double ux,
                                                     double uy,
                                                     double uz):

    return -ux, -uy, -uz

cdef double random_gaussian():

    cdef double x0, x1, w

    w = 2.0
    while (w >= 1.0):
        x0 = 2.0*random_uniform()-1.0
        x1 = 2.0*random_uniform()-1.0
        w = x0*x0+x1*x1

    w = sqrt(-2.0*log(w)/w)

    return x1*w

cdef (double, double, double) random_gaussian_3d():

    cdef double x0, x1, w

    w = 2.0
    while (w >= 1.0):
        x0 = 2.0*random_uniform()-1.0
        x1 = 2.0*random_uniform()-1.0
        w = x0*x0+x1*x1

    w = sqrt(-2.0*log(w)/w)

    return x0*w, x1*w, random_gaussian()

cdef (double, double, double) gaussian_vector_candidate(double ux,
                                                        double uy,
                                                        double uz,
                                                        double sigma):

    cdef double vx, vy, vz
    cdef double wx, wy, wz, w, inv_w

    vx, vy, vz = random_gaussian_3d()

    wx = ux+sigma*vx
    wy = uy+sigma*vy
    wz = uz+sigma*vz

    w = sqrt(wx*wx+wy*wy+wz*wz)
    inv_w = 1.0/w

    return wx*inv_w, wy*inv_w, wz*inv_w

cdef (double, double, double) interpolated_vector_candidate(double ux,
                                                            double uy,
                                                            double uz,
                                                            double sigma):

    cdef double vx, vy, vz
    cdef double wx, wy, wz
        
    cdef double u_dot_v, theta, cos_theta, sin_theta
    
    vx, vy, vz = random_vector_candidate()

    u_dot_v = ux*vx+uy*vy+uz*vz
    
    if iszero(u_dot_v-1):
        
        return ux, uy, uz
    
    elif iszero(u_dot_v+1):
        
        return -ux, -uy, -uz
    
    else:
        
        theta = acos(u_dot_v)*sigma
        
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        
        wx = ux*cos_theta+(vx-u_dot_v*ux)*sin_theta
        wy = uy*cos_theta+(vy-u_dot_v*uy)*sin_theta
        wz = uz*cos_theta+(vz-u_dot_v*uz)*sin_theta
        
        return wx, wy, wz
        
cdef void replica_exchange(double [::1] H,
                           double [::1] beta,
                           double [::1] sigma):

    cdef Py_ssize_t n_temp = H.shape[0]

    cdef Py_ssize_t i, j

    i, j = dist_temp(gen), dist_temp(gen)

    if (i != j):
        if (random_uniform() < alpha(H[j]-H[i], beta[i]-beta[j])):
            beta[i], beta[j] = beta[j], beta[i]
            sigma[i], sigma[j] = sigma[j], sigma[i]

def energy(double [:,:,:,:,::1] Sx,
           double [:,:,:,:,::1] Sy,
           double [:,:,:,:,::1] Sz,
           double [:,:,::1] J,
           double [:,:,::1] A,
           double [:,:,::1] g,
           double [::1] B,
           long [:,::1] atm_ind,
           long [:,::1] img_ind_i,
           long [:,::1] img_ind_j,
           long [:,::1] img_ind_k,
           long [:,::1] pair_ind,
           bint [:,::1] pair_ij):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    cdef Py_ssize_t n_temp = Sx.shape[4]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef Py_ssize_t i, j, k, a, t, p, q

    cdef Py_ssize_t i_, j_, k_, a_

    cdef bint f

    e_np = np.zeros((nu,nv,nw,n_atm,n_pairs+2,n_temp))

    cdef double [:,:,:,:,:,::1] e = e_np

    cdef double ux, uy, uz, vx, vy, vz

    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]

    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    for t in range(n_temp):

                        ux = Sx[i,j,k,a,t]
                        uy = Sy[i,j,k,a,t]
                        uz = Sz[i,j,k,a,t]

                        for p in range(n_pairs):

                            i_ = (i+img_ind_i[a,p]+nu)%nu
                            j_ = (j+img_ind_j[a,p]+nv)%nv
                            k_ = (k+img_ind_k[a,p]+nw)%nw
                            a_ = atm_ind[a,p]

                            vx = Sx[i_,j_,k_,a_,t]
                            vy = Sy[i_,j_,k_,a_,t]
                            vz = Sz[i_,j_,k_,a_,t]

                            q = pair_ind[a,p]
                            f = pair_ij[a,p]

                            if (f == 1):
                                e[i,j,k,a,p,t] = -0.5\
                                    *(ux*(J[q,0,0]*vx+J[q,1,0]*vy+J[q,2,0]*vz)\
                                    + uy*(J[q,0,1]*vx+J[q,1,1]*vy+J[q,2,1]*vz)\
                                    + uz*(J[q,0,2]*vx+J[q,1,2]*vy+J[q,2,2]*vz))
                            else:
                                e[i,j,k,a,p,t] = -0.5\
                                    *(ux*(J[q,0,0]*vx+J[q,0,1]*vy+J[q,0,2]*vz)\
                                    + uy*(J[q,1,0]*vx+J[q,1,1]*vy+J[q,1,2]*vz)\
                                    + uz*(J[q,2,0]*vx+J[q,2,1]*vy+J[q,2,2]*vz))

                        e[i,j,k,a,n_pairs,t] = \
                            -(ux*(A[a,0,0]*ux+A[a,0,1]*uy+A[a,0,2]*uz)\
                            + uy*(A[a,1,0]*ux+A[a,1,1]*uy+A[a,1,2]*uz)\
                            + uz*(A[a,2,0]*ux+A[a,2,1]*uy+A[a,2,2]*uz))

                        e[i,j,k,a,n_pairs+1,t] = \
                            -(Bx*(g[a,0,0]*ux+g[a,0,1]*uy+g[a,0,2]*uz)\
                            + By*(g[a,1,0]*ux+g[a,1,1]*uy+g[a,1,2]*uz)\
                            + Bz*(g[a,2,0]*ux+g[a,2,1]*uy+g[a,2,2]*uz))

    return e_np

cdef double annealing_vector(double [:,:,:,:,::1] Sx,
                             double [:,:,:,:,::1] Sy,
                             double [:,:,:,:,::1] Sz,
                             double sx,
                             double sy,
                             double sz,
                             double [::1] H,
                             double [::1] E,
                             double [::1] beta,
                             double [::1] count,
                             double [::1] total,
                             Py_ssize_t i,
                             Py_ssize_t j,
                             Py_ssize_t k,
                             Py_ssize_t a,
                             Py_ssize_t t):

    if (random_uniform() < alpha(E[t], beta[t])):

        Sx[i,j,k,a,t] = sx
        Sy[i,j,k,a,t] = sy
        Sz[i,j,k,a,t] = sz

        H[t] += E[t]
        count[t] += 1

    total[t] += 1

    return count[t]/total[t]

cdef double magnetic(double [:,:,:,:,::1] Sx,
                     double [:,:,:,:,::1] Sy,
                     double [:,:,:,:,::1] Sz,
                     double vx,
                     double vy,
                     double vz,
                     double ux,
                     double uy,
                     double uz,
                     double [:,:,::1] J,
                     double [:,:,::1] A,
                     double [:,:,::1] g,
                     double [::1] B,
                     long [:,::1] atm_ind,
                     long [:,::1] img_ind_i,
                     long [:,::1] img_ind_j,
                     long [:,::1] img_ind_k,
                     long [:,::1] pair_ind,
                     long [:,::1] pair_ij,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     Py_ssize_t k,
                     Py_ssize_t a,
                     Py_ssize_t t):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef Py_ssize_t i_, j_, k_, a_, p, q

    cdef bint f

    cdef double E = 0

    cdef double wx, wy, wz

    cdef double dx, dy, dz

    dx, dy, dz = vx-ux, vy-uy, vz-uz

    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]

    for p in range(n_pairs):

        i_ = (i+img_ind_i[a,p]+nu)%nu
        j_ = (j+img_ind_j[a,p]+nv)%nv
        k_ = (k+img_ind_k[a,p]+nw)%nw
        a_ = atm_ind[a,p]

        q = pair_ind[a,p]
        f = pair_ij[a,p]

        wx, wy, wz = Sx[i_,j_,k_,a_,t], Sy[i_,j_,k_,a_,t], Sz[i_,j_,k_,a_,t]

        if (f == 1):
            E -= dx*(J[q,0,0]*wx+J[q,1,0]*wy+J[q,2,0]*wz)\
              +  dy*(J[q,0,1]*wx+J[q,1,1]*wy+J[q,2,1]*wz)\
              +  dz*(J[q,0,2]*wx+J[q,1,2]*wy+J[q,2,2]*wz)
        else:
            E -= dx*(J[q,0,0]*wx+J[q,0,1]*wy+J[q,0,2]*wz)\
              +  dy*(J[q,1,0]*wx+J[q,1,1]*wy+J[q,1,2]*wz)\
              +  dz*(J[q,2,0]*wx+J[q,2,1]*wy+J[q,2,2]*wz)

    E -= vx*(A[a,0,0]*vx+A[a,0,1]*vy+A[a,0,2]*vz)\
      +  vy*(A[a,1,0]*vx+A[a,1,1]*vy+A[a,1,2]*vz)\
      +  vz*(A[a,2,0]*vx+A[a,2,1]*vy+A[a,2,2]*vz)

    E += ux*(A[a,0,0]*ux+A[a,0,1]*uy+A[a,0,2]*uz)\
      +  uy*(A[a,1,0]*ux+A[a,1,1]*uy+A[a,1,2]*uz)\
      +  uz*(A[a,2,0]*ux+A[a,2,1]*uy+A[a,2,2]*uz)

    E -= Bx*(g[a,0,0]*dx+g[a,0,1]*dy+g[a,0,2]*dz)\
      +  By*(g[a,1,0]*dx+g[a,1,1]*dy+g[a,1,2]*dz)\
      +  Bz*(g[a,2,0]*dx+g[a,2,1]*dy+g[a,2,2]*dz)

    return E

def heisenberg(double [:,:,:,:,::1] Sx,
               double [:,:,:,:,::1] Sy,
               double [:,:,:,:,::1] Sz,
               double [:,:,::1] J,
               double [:,:,::1] A,
               double [:,:,::1] g,
               double [::1] B,
               long [:,::1] atm_ind,
               long [:,::1] img_ind_i,
               long [:,::1] img_ind_j,
               long [:,::1] img_ind_k,
               long [:,::1] pair_ind,
               long [:,::1] pair_ij,
               double [::1] T_range,
               double kB,
               Py_ssize_t N):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]

    cdef Py_ssize_t n_temp = T_range.shape[0]

    initialize_random(nu, nv, nw, n_atm, n_temp)

    cdef Py_ssize_t i, j, k, a, t, p, n

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double [::1] E = np.zeros(n_temp)
    cdef double [::1] H = np.zeros(n_temp)

    cdef double [:,:,:,:,:,::1] e = energy(Sx, Sy, Sz, J, A, g, B, atm_ind,
                                           img_ind_i, img_ind_j, img_ind_k,
                                           pair_ind, pair_ij)

    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    for p in range(n_pairs+2):
                        for t in range(n_temp):
                            H[t] += e[i,j,k,a,p,t]

    n = nu*nv*nw*n_atm

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double [::1] sigma = np.full(n_temp, 1.)

    cdef double [::1] count = np.zeros(n_temp)
    cdef double [::1] total = np.zeros(n_temp)

    cdef double rate, factor

    cdef double [::1] beta = 1/(kB*np.copy(T_range))

    for _ in range(N):

        for _ in range(1):

            i, j, k, a = random_original(nu, nv, nw, n_atm)

            for t in range(n_temp):

                ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

                vx, vy, vz = gaussian_vector_candidate(ux, uy, uz, sigma[t])

                E[t] = magnetic(Sx, Sy, Sz, vx, vy, vz, ux, uy, uz, J, A, g, B,
                                atm_ind, img_ind_i, img_ind_j, img_ind_k,
                                pair_ind, pair_ij, i, j, k, a, t)

                rate = annealing_vector(Sx, Sy, Sz, vx, vy, vz, H, E, beta,
                                        count, total, i, j, k, a, t)

                if (rate > 0.0 and rate < 1.0):
                    factor = rate/(1.0-rate)
                    sigma[t] *= factor

                    if (sigma[t] < 0.01): sigma[t] = 0.01
                    if (sigma[t] > 10): sigma[t] = 10

                print(sigma[t],rate,factor) 

            replica_exchange(H, beta, sigma)

    return 1/(kB*np.copy(beta))

# ---

cdef double annealing_cluster(double [:,:,:,:,::1] Sx,
                              double [:,:,:,:,::1] Sy,
                              double [:,:,:,:,::1] Sz,
                              double nx,
                              double ny,
                              double nz,
                              double [:,:,::1] J,
                              double [:,::,::1] A,
                              double [:,::,::1] g,
                              double [::1] B,
                              Py_ssize_t [:,::1] clust_i,
                              Py_ssize_t [:,::1] clust_j,
                              Py_ssize_t [:,::1] clust_k,
                              Py_ssize_t [:,::1] clust_a,
                              double [:,:,:,:,::1] h_eff,
                              bint [:,:,:,:,::1] b,
                              bint [:,:,:,:,::1] c,
                              Py_ssize_t [::1] n_c,
                              long [:,::1] atm_ind,
                              long [:,::1] img_ind_i,
                              long [:,::1] img_ind_j,
                              long [:,::1] img_ind_k,
                              long [:,::1] pair_ind,
                              long [:,::1] pair_ij,
                              double [::1] H,
                              double [::1] Eij,
                              double [::1] beta,
                              double [::1] count,
                              double [::1] total,
                              Py_ssize_t t):

    cdef Py_ssize_t n_temp = n_c.shape[0]

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double Ec = 0

    cdef Py_ssize_t p, i_c

    cdef bint f

    cdef Py_ssize_t i, j, k, a
    cdef Py_ssize_t i_, j_, k_, a_

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double dx, dy, dz

    cdef double n_dot_u

    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]

    for i_c in range(n_c[t]):

        i = clust_i[i_c,t]
        j = clust_j[i_c,t]
        k = clust_k[i_c,t]
        a = clust_a[i_c,t]

        ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

        n_dot_u = nx*ux+ny*uy+nz*uz

        vx = ux-2*nx*n_dot_u
        vy = uy-2*ny*n_dot_u
        vz = uz-2*nz*n_dot_u

        Ec -= vx*(A[a,0,0]*vx+A[a,0,1]*vy+A[a,0,2]*vz)\
           +  vy*(A[a,1,0]*vx+A[a,1,1]*vy+A[a,1,2]*vz)\
           +  vz*(A[a,2,0]*vx+A[a,2,1]*vy+A[a,2,2]*vz)

        Ec += ux*(A[a,0,0]*ux+A[a,0,1]*uy+A[a,0,2]*uz)\
           +  uy*(A[a,1,0]*ux+A[a,1,1]*uy+A[a,1,2]*uz)\
           +  uz*(A[a,2,0]*ux+A[a,2,1]*uy+A[a,2,2]*uz)

        dx = vx-ux
        dy = vy-uy
        dz = vz-uz

        Ec -= Bx*(g[a,0,0]*dx+g[a,0,1]*dy+g[a,0,2]*dz)\
           +  By*(g[a,1,0]*dx+g[a,1,1]*dy+g[a,1,2]*dz)\
           +  Bz*(g[a,2,0]*dx+g[a,2,1]*dy+g[a,2,2]*dz)

        Ec += h_eff[i,j,k,a,t]

        b[i,j,k,a,t] = 0
        c[i,j,k,a,t] = 0

        h_eff[i,j,k,a,t] = 0

    if (random_uniform() < alpha(Ec, beta[t])):

        for i_c in range(n_c[t]):

            i = clust_i[i_c,t]
            j = clust_j[i_c,t]
            k = clust_k[i_c,t]
            a = clust_a[i_c,t]

            ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

            n_dot_u = nx*ux+ny*uy+nz*uz

            vx = ux-2*nx*n_dot_u
            vy = uy-2*ny*n_dot_u
            vz = uz-2*nz*n_dot_u

            Sx[i,j,k,a,t] = vx
            Sy[i,j,k,a,t] = vy
            Sz[i,j,k,a,t] = vz

        H[t] += Ec+Eij[t]
        count[t] += 1

    total[t] += 1

    return count[t]/total[t]

cdef Py_ssize_t magnetic_cluster(double [:,:,:,:,::1] Sx,
                                 double [:,:,:,:,::1] Sy,
                                 double [:,:,:,:,::1] Sz,
                                 double nx,
                                 double ny,
                                 double nz,
                                 double ux_perp,
                                 double uy_perp,
                                 double uz_perp,
                                 double n_dot_u,
                                 double [:,:,::1] J,
                                 Py_ssize_t [:,::1] clust_i,
                                 Py_ssize_t [:,::1] clust_j,
                                 Py_ssize_t [:,::1] clust_k,
                                 Py_ssize_t [:,::1] clust_a,
                                 double [:,:,:,:,::1] h_eff,
                                 double [:,:,:,:,:,::1] h_eff_ij,
                                 bint [:,:,:,:,::1] b,
                                 bint [:,:,:,:,::1] c,
                                 Py_ssize_t [::1] n_c,
                                 long [:,::1] atm_ind,
                                 long [:,::1] img_ind_i,
                                 long [:,::1] img_ind_j,
                                 long [:,::1] img_ind_k,
                                 long [:,::1] pair_ind,
                                 long [:,::1] pair_inv,
                                 long [:,::1] pair_ij,
                                 double [::1] beta,
                                 Py_ssize_t i,
                                 Py_ssize_t j,
                                 Py_ssize_t k,
                                 Py_ssize_t a,
                                 Py_ssize_t t):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef Py_ssize_t m_c = n_c[t]

    cdef double E

    cdef Py_ssize_t i_, j_, k_, a_, p_, p, q

    cdef bint f

    cdef double vx, vy, vz
    cdef double vx_perp, vy_perp, vz_perp

    cdef double J_eff, J_eff_ij, J_eff_ji

    cdef double Jx_eff_ij, Jy_eff_ij, Jz_eff_ij
    cdef double Jx_eff_ji, Jy_eff_ji, Jz_eff_ji

    cdef double n_dot_v

    for p in range(n_pairs):

        i_ = (i+img_ind_i[a,p]+nu)%nu
        j_ = (j+img_ind_j[a,p]+nv)%nv
        k_ = (k+img_ind_k[a,p]+nw)%nw
        a_ = atm_ind[a,p]
        p_ = pair_inv[a,p]

        if (b[i_,j_,k_,a_,t] == 0):

            vx = Sx[i_,j_,k_,a_,t]
            vy = Sy[i_,j_,k_,a_,t]
            vz = Sz[i_,j_,k_,a_,t]

            n_dot_v = vx*nx+vy*ny+vz*nz

            vx_perp = vx-nx*n_dot_v
            vy_perp = vy-ny*n_dot_v
            vz_perp = vz-nz*n_dot_v

            q = pair_ind[a,p]
            f = pair_ij[a,p]

            if (f == 1):
                Jx_eff_ij = J[q,0,0]*nx+J[q,1,0]*ny+J[q,2,0]*nz
                Jy_eff_ij = J[q,0,1]*nx+J[q,1,1]*ny+J[q,2,1]*nz
                Jz_eff_ij = J[q,0,2]*nx+J[q,1,2]*ny+J[q,2,2]*nz

                Jx_eff_ji = J[q,0,0]*nx+J[q,0,1]*ny+J[q,0,2]*nz
                Jy_eff_ji = J[q,1,0]*nx+J[q,1,1]*ny+J[q,1,2]*nz
                Jz_eff_ji = J[q,2,0]*nx+J[q,2,1]*ny+J[q,2,2]*nz
            else:
                Jx_eff_ij = J[q,0,0]*nx+J[q,0,1]*ny+J[q,0,2]*nz
                Jy_eff_ij = J[q,1,0]*nx+J[q,1,1]*ny+J[q,1,2]*nz
                Jz_eff_ij = J[q,2,0]*nx+J[q,2,1]*ny+J[q,2,2]*nz

                Jx_eff_ji = J[q,0,0]*nx+J[q,1,0]*ny+J[q,2,0]*nz
                Jy_eff_ji = J[q,0,1]*nx+J[q,1,1]*ny+J[q,2,1]*nz
                Jz_eff_ji = J[q,0,2]*nx+J[q,1,2]*ny+J[q,2,2]*nz

            J_eff_ij = ux_perp*Jx_eff_ij+uy_perp*Jy_eff_ij+uz_perp*Jz_eff_ij

            h_eff_ij[i_,j_,k_,a_,p_,t] = 2*J_eff_ij*n_dot_v

            J_eff_ji = vx_perp*Jx_eff_ji+vy_perp*Jy_eff_ji+vz_perp*Jz_eff_ji

            h_eff_ij[i,j,k,a,p,t] = 2*J_eff_ji*n_dot_u

            if (c[i_,j_,k_,a_,t] == 0):

                J_eff = nx*Jx_eff_ij+ny*Jy_eff_ij+nz*Jz_eff_ij

                E = 2*J_eff*n_dot_u*n_dot_v

                if (random_uniform() < 1-alpha(E, beta[t])):

                    clust_i[m_c,t] = i_
                    clust_j[m_c,t] = j_
                    clust_k[m_c,t] = k_
                    clust_a[m_c,t] = a_

                    c[i_,j_,k_,a_,t] = 1

                    m_c += 1

            h_eff[i,j,k,a,t] += h_eff_ij[i,j,k,a,p,t]

        elif (b[i_,j_,k_,a_,t] == 1):

            h_eff[i,j,k,a,t] += h_eff_ij[i,j,k,a,p,t]

    return m_c

cdef double boundary_energy(double [:,:,:,:,::1] Sx,
                            double [:,:,:,:,::1] Sy,
                            double [:,:,:,:,::1] Sz,
                            double nx,
                            double ny,
                            double nz,
                            double [:,:,::1] J,
                            Py_ssize_t [:,::1] clust_i,
                            Py_ssize_t [:,::1] clust_j,
                            Py_ssize_t [:,::1] clust_k,
                            Py_ssize_t [:,::1] clust_a,
                            bint [:,:,:,:,::1] c,
                            Py_ssize_t [::1] n_c,
                            long [:,::1] atm_ind,
                            long [:,::1] img_ind_i,
                            long [:,::1] img_ind_j,
                            long [:,::1] img_ind_k,
                            long [:,::1] pair_ind,
                            long [:,::1] pair_inv,
                            long [:,::1] pair_ij,
                            Py_ssize_t t):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef Py_ssize_t i_c

    cdef double E = 0

    cdef Py_ssize_t i, j, k, a, p, q, i_, j_, k_, a_

    cdef bint f

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double J_eff

    cdef double Jx_eff_ij, Jy_eff_ij, Jz_eff_ij

    cdef double n_dot_u, n_dot_v

    for i_c in range(n_c[t]):

        i = clust_i[i_c,t]
        j = clust_j[i_c,t]
        k = clust_k[i_c,t]
        a = clust_a[i_c,t]

        ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

        n_dot_u = ux*nx+uy*ny+uz*nz

        for p in range(n_pairs):

            i_ = (i+img_ind_i[a,p]+nu)%nu
            j_ = (j+img_ind_j[a,p]+nv)%nv
            k_ = (k+img_ind_k[a,p]+nw)%nw
            a_ = atm_ind[a,p]

            if (c[i_,j_,k_,a_,t] == 0):

                vx = Sx[i_,j_,k_,a_,t]
                vy = Sy[i_,j_,k_,a_,t]
                vz = Sz[i_,j_,k_,a_,t]

                n_dot_v = vx*nx+vy*ny+vz*nz

                q = pair_ind[a,p]
                f = pair_ij[a,p]

                if (f == 1):
                    Jx_eff_ij = J[q,0,0]*nx+J[q,1,0]*ny+J[q,2,0]*nz
                    Jy_eff_ij = J[q,0,1]*nx+J[q,1,1]*ny+J[q,2,1]*nz
                    Jz_eff_ij = J[q,0,2]*nx+J[q,1,2]*ny+J[q,2,2]*nz
                else:
                    Jx_eff_ij = J[q,0,0]*nx+J[q,0,1]*ny+J[q,0,2]*nz
                    Jy_eff_ij = J[q,1,0]*nx+J[q,1,1]*ny+J[q,1,2]*nz
                    Jz_eff_ij = J[q,2,0]*nx+J[q,2,1]*ny+J[q,2,2]*nz

                J_eff = nx*Jx_eff_ij+ny*Jy_eff_ij+nz*Jz_eff_ij

                E += 2*J_eff*n_dot_u*n_dot_v

    return E

def heisenberg_cluster(double [:,:,:,:,::1] Sx,
                       double [:,:,:,:,::1] Sy,
                       double [:,:,:,:,::1] Sz,
                       double [:,:,::1] J,
                       double [:,::,::1] A,
                       double [:,::,::1] g,
                       double [::1] B,
                       long [:,::1] atm_ind,
                       long [:,::1] img_ind_i,
                       long [:,::1] img_ind_j,
                       long [:,::1] img_ind_k,
                       long [:,::1] pair_ind,
                       long [:,::1] pair_inv,
                       long [:,::1] pair_ij,
                       double [::1] T_range,
                       double kB,
                       Py_ssize_t N):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]

    cdef Py_ssize_t n_temp = T_range.shape[0]

    initialize_random(nu, nv, nw, n_atm, n_temp)

    cdef Py_ssize_t t, n
    cdef Py_ssize_t i, j, k, a,
    cdef Py_ssize_t i_, j_, k_, a_

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double [::1] E = np.zeros(n_temp)
    cdef double [::1] H = np.zeros(n_temp)

    cdef double [:,:,:,:,:,::1] e = energy(Sx, Sy, Sz, J, A, g, B, atm_ind,
                                           img_ind_i, img_ind_j, img_ind_k,
                                           pair_ind, pair_ij)

    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    for p in range(n_pairs+2):
                        for t in range(n_temp):
                            H[t] += e[i,j,k,a,p,t]

    n = nu*nv*nw*n_atm

    cdef Py_ssize_t i_c, m_c

    cdef Py_ssize_t [::1] n_c = np.zeros(n_temp, dtype=np.intp)

    spin_shape = (nu,nv,nw,n_atm,n_temp)

    cdef bint [:,:,:,:,::1] b = np.zeros(spin_shape, dtype=np.intc)
    cdef bint [:,:,:,:,::1] c = np.zeros(spin_shape, dtype=np.intc)

    pairs_shape = (nu,nv,nw,n_atm,n_pairs,n_temp)

    cdef double [:,:,:,:,::1] h_eff = np.zeros(spin_shape, dtype=float)
    cdef double [:,:,:,:,:,::1] h_eff_ij = np.zeros(pairs_shape, dtype=float)

    cdef Py_ssize_t [:,::1] clust_i = np.zeros((n,n_temp), dtype=np.intp)
    cdef Py_ssize_t [:,::1] clust_j = np.zeros((n,n_temp), dtype=np.intp)
    cdef Py_ssize_t [:,::1] clust_k = np.zeros((n,n_temp), dtype=np.intp)
    cdef Py_ssize_t [:,::1] clust_a = np.zeros((n,n_temp), dtype=np.intp)

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double mx, my, mz
    cdef double nx, ny, nz

    cdef double ux_perp, uy_perp, uz_perp
    cdef double vx_perp, vy_perp, vz_perp

    cdef double u, n_dot_u, n_dot_v

    cdef double [::1] sigma = np.full(n_temp, 0.)

    cdef double [::1] count = np.zeros(n_temp)
    cdef double [::1] total = np.zeros(n_temp)

    cdef double rate, factor

    cdef double [::1] beta = 1/(kB*np.copy(T_range))

    for _ in range(N):

        for _ in range(n):

            i, j, k, a = random_original(nu, nv, nw, n_atm)

            for t in range(n_temp):

                ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

                #nx, ny, nz = gaussian_vector_candidate(ux, uy, uz, sigma[t])
                nx, ny, nz = ux, uy, uz

                n_dot_u = ux*nx+uy*ny+uz*nz

                ux_perp = ux-nx*n_dot_u
                uy_perp = uy-ny*n_dot_u
                uz_perp = uz-nz*n_dot_u

                clust_i[0,t] = i
                clust_j[0,t] = j
                clust_k[0,t] = k
                clust_a[0,t] = a

                c[i,j,k,a,t] = 1

                i_c, m_c, n_c[t] = 0, 1, 1

                while i_c < m_c:

                    i_ = clust_i[i_c,t]
                    j_ = clust_j[i_c,t]
                    k_ = clust_k[i_c,t]
                    a_ = clust_a[i_c,t]

                    vx = Sx[i_,j_,k_,a_,t]
                    vy = Sy[i_,j_,k_,a_,t]
                    vz = Sz[i_,j_,k_,a_,t]

                    n_dot_v = vx*nx+vy*ny+vz*nz

                    vx_perp = vx-nx*n_dot_v
                    vy_perp = vy-ny*n_dot_v
                    vz_perp = vz-nz*n_dot_v

                    m_c = magnetic_cluster(Sx, Sy, Sz, nx, ny, nz,
                                           vx_perp, vy_perp, vz_perp,
                                           n_dot_v, J,
                                           clust_i, clust_j, clust_k, clust_a,
                                           h_eff, h_eff_ij, b, c, n_c, atm_ind,
                                           img_ind_i, img_ind_j, img_ind_k,
                                           pair_ind, pair_inv, pair_ij, beta,
                                           i_, j_, k_, a_, t)

                    b[i_,j_,k_,a_,t] = 1

                    n_c[t] = m_c

                    i_c += 1

                E[t] = boundary_energy(Sx, Sy, Sz, nx, ny, nz, J,
                                       clust_i, clust_j, clust_k,
                                       clust_a, c, n_c, atm_ind,
                                       img_ind_i, img_ind_j, img_ind_k,
                                       pair_ind, pair_inv, pair_ij, t)

                rate = annealing_cluster(Sx, Sy, Sz, nx, ny, nz, J, A, g, B,
                                         clust_i, clust_j, clust_k, clust_a,
                                         h_eff, b, c, n_c, atm_ind,
                                         img_ind_i, img_ind_j, img_ind_k,
                                         pair_ind, pair_ij, H, E, beta,
                                         count, total, t)

                if (rate > 0.0 and rate < 1.0):
                    factor = rate/(1.0-rate)
                    sigma[t] *= factor

                    # if (sigma[t] < 0.0001): sigma[t] = 0.0001
                    # if (sigma[t] > 1000): sigma[t] = 1000

                # print(sigma[t],rate,factor,count[t],total[t],H[t]) 

            replica_exchange(H, beta, sigma)

    return 1/(kB*np.copy(beta))