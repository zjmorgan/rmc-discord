#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport M_PI, fabs, log, exp, sqrt
from libc.math cimport sin, cos, tan
from libc.math cimport acos, atan, atan2
from libcpp.vector cimport vector

cdef extern from '<random>' namespace 'std' nogil:

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

cdef vector[vector[vector[vector[mt19937]]]] gen_ind

cdef uniform_real_distribution[double] dist
cdef uniform_int_distribution[Py_ssize_t] dist_u
cdef uniform_int_distribution[Py_ssize_t] dist_v
cdef uniform_int_distribution[Py_ssize_t] dist_w
cdef uniform_int_distribution[Py_ssize_t] dist_atm
cdef uniform_int_distribution[Py_ssize_t] dist_temp

cdef double MACHINE_EPSILON = np.finfo(float).eps

cdef Py_ssize_t seed = 20

cpdef void set_seed(Py_ssize_t s):

    global seed

    seed = s

cdef void initialize_random(Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm,
                            Py_ssize_t n_temp):

    cdef Py_ssize_t i, j, k, a, ind

    global gen, gen_ind, dist, dist_u, dist_v, dist_w, dist_atm, dist_temp

    cdef vector[vector[vector[vector[mt19937]]]] u
    cdef vector[vector[vector[mt19937]]] v
    cdef vector[vector[mt19937]] w
    cdef vector[mt19937] atm

    u.clear()
    for i in range(nu):
        v.clear()
        for j in range(nv):
            w.clear()
            for k in range(nw):
                atm.clear()
                for a in range(n_atm):
                    ind = 1+seed+a+n_atm*(k+nw*(j+nv*i))
                    atm.push_back(mt19937(ind))
                w.push_back(atm)
            v.push_back(w)
        u.push_back(v)

    gen_ind = u

    gen = mt19937(seed)

    dist = uniform_real_distribution[double](0.0,1.0)

    dist_u = uniform_int_distribution[Py_ssize_t](0,nu-1)
    dist_v = uniform_int_distribution[Py_ssize_t](0,nv-1)
    dist_w = uniform_int_distribution[Py_ssize_t](0,nw-1)
    dist_atm = uniform_int_distribution[Py_ssize_t](0,n_atm-1)
    dist_temp = uniform_int_distribution[Py_ssize_t](0,n_temp-1)

cdef bint iszero(double a) nogil:

    cdef double atol = 1e-08

    return fabs(a) <= atol

cdef Py_ssize_t sqrt_babylonian(Py_ssize_t n) nogil:

    cdef Py_ssize_t x = n
    cdef Py_ssize_t y = 1

    while (x > y):

        x = (x+y) // 2
        y = n // x

    return x

cdef double random_uniform() nogil:

    return dist(gen)

cdef double random_uniform_parallel(Py_ssize_t i,
                                    Py_ssize_t j,
                                    Py_ssize_t k,
                                    Py_ssize_t a) nogil:

    return dist(gen_ind[i][j][k][a])

cdef double alpha(double E, double beta) nogil:

    return exp(-beta*E)

cdef (Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t) random_original(Py_ssize_t nu,
                                  Py_ssize_t nv,
                                  Py_ssize_t nw,
                                  Py_ssize_t n_atm) nogil:

    cdef Py_ssize_t i = dist_u(gen)
    cdef Py_ssize_t j = dist_v(gen)
    cdef Py_ssize_t k = dist_w(gen)
    cdef Py_ssize_t a = dist_atm(gen)

    return i, j, k, a

cdef (double, double, double) random_vector_candidate() nogil:

    cdef double theta = 2*M_PI*random_uniform()
    cdef double phi = acos(1-2*random_uniform())

    cdef double sx = sin(phi)*cos(theta)
    cdef double sy = sin(phi)*sin(theta)
    cdef double sz = cos(phi)

    return sx, sy, sz

cdef (double, double, double) random_vector_length_candidate() nogil:

    cdef double theta = 2*M_PI*random_uniform()
    cdef double phi = acos(1-2*random_uniform())

    cdef double s = random_uniform()

    cdef double sx = s*sin(phi)*cos(theta)
    cdef double sy = s*sin(phi)*sin(theta)
    cdef double sz = s*cos(phi)

    return sx, sy, sz

cdef (double, double, double) ising_vector_candidate(double ux,
                                                     double uy,
                                                     double uz) nogil:

    return -ux, -uy, -uz

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
                                                        double sigma) nogil:

    cdef double vx, vy, vz
    cdef double wx, wy, wz, w, inv_w

    vx, vy, vz = random_gaussian_3d()

    wx = ux+sigma*vx
    wy = uy+sigma*vy
    wz = uz+sigma*vz

    w = sqrt(wx*wx+wy*wy+wz*wz)
    inv_w = 1.0/w

    return wx*inv_w, wy*inv_w, wz*inv_w

cdef (double,
      double,
      double) interpolated_vector_candidate(double ux,
                                            double uy,
                                            double uz,
                                            double sigma) nogil:

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
                           double [::1] sigma) nogil:

    cdef Py_ssize_t n_temp = H.shape[0]

    cdef Py_ssize_t i, j

    i, j = dist_temp(gen), dist_temp(gen)

    if (i != j):
        if (random_uniform() < alpha(H[j]-H[i], beta[i]-beta[j])):
            beta[i], beta[j] = beta[j], beta[i]
            sigma[i], sigma[j] = sigma[j], sigma[i]

cdef double energy_moment(double [:,:,:,::1] p,
                          double [:,::1] Q,
                          double vx,
                          double vy,
                          double vz,
                          double ux,
                          double uy,
                          double uz,
                          Py_ssize_t i,
                          Py_ssize_t t) nogil:

    cdef Py_ssize_t n = p.shape[0]

    cdef double dx = vx-ux
    cdef double dy = vy-uy
    cdef double dz = vz-uz

    cdef double px = p[i,0,0,t]+p[i,0,1,t]+p[i,0,2,t]
    cdef double py = p[i,1,0,t]+p[i,1,1,t]+p[i,1,2,t]
    cdef double pz = p[i,2,0,t]+p[i,2,1,t]+p[i,2,2,t]

    cdef Py_ssize_t k = i+n*i-(i+1)*i // 2

    cdef double E = 2*px*dx+Q[k,0]*dx*dx+2*Q[k,3]*dy*dz\
                  + 2*py*dy+Q[k,1]*dy*dy+2*Q[k,4]*dx*dz\
                  + 2*pz*dz+Q[k,2]*dz*dz+2*Q[k,5]*dx*dy

    return E

cdef void update_moment(double [:,:,:,::1] p,
                        double [:,::1] Q,
                        double vx,
                        double vy,
                        double vz,
                        double ux,
                        double uy,
                        double uz,
                        Py_ssize_t i,
                        Py_ssize_t t) nogil:

    cdef Py_ssize_t n = p.shape[0]

    cdef double dx = vx-ux
    cdef double dy = vy-uy
    cdef double dz = vz-uz

    cdef Py_ssize_t j, k

    for j in range(n):

        if j < i:
            k = i+n*j-(j+1)*j // 2
        else:
            k = j+n*i-(i+1)*i // 2

        p[j,0,0,t] += Q[k,0]*dx
        p[j,1,1,t] += Q[k,1]*dy
        p[j,2,2,t] += Q[k,2]*dz

        p[j,1,2,t] += Q[k,3]*dz
        p[j,2,1,t] += Q[k,3]*dy

        p[j,0,2,t] += Q[k,4]*dz
        p[j,2,0,t] += Q[k,4]*dx

        p[j,0,1,t] += Q[k,5]*dy
        p[j,1,0,t] += Q[k,5]*dx

cdef double energy_moment_cluster(double [:,:,:,::1] p,
                                  double [:,::1] Q,
                                  double [::1] clust_vx,
                                  double [::1] clust_vy,
                                  double [::1] clust_vz,
                                  double [::1] clust_ux,
                                  double [::1] clust_uy,
                                  double [::1] clust_uz,
                                  Py_ssize_t [::1] clust_ind,
                                  Py_ssize_t n_c,
                                  Py_ssize_t t) nogil:

    cdef Py_ssize_t n = p.shape[0]

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double dux, duy, duz
    cdef double dvx, dvy, dvz

    cdef double px, py, pz

    cdef Py_ssize_t i_c, j_c

    cdef Py_ssize_t i, j, k

    cdef double E = 0

    for i_c in range(n_c):

        i = clust_ind[i_c]

        ux = clust_ux[i]
        uy = clust_uy[i]
        uz = clust_uz[i]

        vx = clust_vx[i]
        vy = clust_vy[i]
        vz = clust_vz[i]

        dux = vx-ux
        duy = vy-uy
        duz = vz-uz

        px = p[i,0,0,t]+p[i,0,1,t]+p[i,0,2,t]
        py = p[i,1,0,t]+p[i,1,1,t]+p[i,1,2,t]
        pz = p[i,2,0,t]+p[i,2,1,t]+p[i,2,2,t]

        E += 2*(px*dux+py*duy+pz*duz)

        for j_c in range(i_c,n_c):

            j = clust_ind[j_c]

            ux = clust_ux[j]
            uy = clust_uy[j]
            uz = clust_uz[j]

            vx = clust_vx[j]
            vy = clust_vy[j]
            vz = clust_vz[j]

            dvx = vx-ux
            dvy = vy-uy
            dvz = vz-uz

            if j < i:
                k = i+n*j-(j+1)*j // 2
            else:
                k = j+n*i-(i+1)*i // 2

            if i_c == j_c:
                E += Q[k,0]*dux*dvx+Q[k,5]*dux*dvy+Q[k,4]*dux*dvz\
                   + Q[k,5]*duy*dvx+Q[k,1]*duy*dvy+Q[k,3]*duy*dvz\
                   + Q[k,4]*duz*dvx+Q[k,3]*duz*dvy+Q[k,2]*duz*dvz
            else:
                E += 2*(Q[k,0]*dux*dvx+Q[k,5]*dux*dvy+Q[k,4]*dux*dvz\
                      + Q[k,5]*duy*dvx+Q[k,1]*duy*dvy+Q[k,3]*duy*dvz\
                      + Q[k,4]*duz*dvx+Q[k,3]*duz*dvy+Q[k,2]*duz*dvz)

    return E

cdef void update_moment_cluster(double [:,:,:,::1] p,
                                double [:,::1] Q,
                                double [::1] clust_vx,
                                double [::1] clust_vy,
                                double [::1] clust_vz,
                                double [::1] clust_ux,
                                double [::1] clust_uy,
                                double [::1] clust_uz,
                                Py_ssize_t [::1] clust_ind,
                                Py_ssize_t n_c,
                                Py_ssize_t t) nogil:

    cdef Py_ssize_t n = p.shape[0]

    cdef double dux, duy, duz
    cdef double dvx, dvy, dvz

    cdef Py_ssize_t i, j, k

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef Py_ssize_t i_c

    for i_c in range(n_c):

        i = clust_ind[i_c]

        ux = clust_ux[i]
        uy = clust_uy[i]
        uz = clust_uz[i]

        vx = clust_vx[i]
        vy = clust_vy[i]
        vz = clust_vz[i]

        dux = vx-ux
        duy = vy-uy
        duz = vz-uz

        for j in range(n):

            if j < i:
                k = i+n*j-(j+1)*j // 2
            else:
                k = j+n*i-(i+1)*i // 2

            p[j,0,0,t] += Q[k,0]*dux
            p[j,1,1,t] += Q[k,1]*duy
            p[j,2,2,t] += Q[k,2]*duz

            p[j,1,2,t] += Q[k,3]*duz
            p[j,2,1,t] += Q[k,3]*duy

            p[j,0,2,t] += Q[k,4]*duz
            p[j,2,0,t] += Q[k,4]*dux

            p[j,0,1,t] += Q[k,5]*duy
            p[j,1,0,t] += Q[k,5]*dux

def dipole_dipole_interaction_energy(double [:,:,:,:,::1] Sx,
                                     double [:,:,:,:,::1] Sy,
                                     double [:,:,:,:,::1] Sz,
                                     double [:,::1] Q):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    cdef Py_ssize_t n_temp = Sx.shape[4]

    cdef Py_ssize_t n = nu*nv*nw*n_atm

    cdef Py_ssize_t i, j, k, t

    cdef Py_ssize_t iu, ju, ku, au
    cdef Py_ssize_t iv, jv, kv, av

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    e_np = np.zeros((n,3,3,n_temp))

    cdef double [:,:,:,::1] e = e_np

    for t in range(n_temp):

        for i in range(n):

            au = i % n_atm
            ku = i // n_atm % nw
            ju = i // n_atm // nw % nv
            iu = i // n_atm // nw // nv % nu

            ux = Sx[iu,ju,ku,au,t]
            uy = Sy[iu,ju,ku,au,t]
            uz = Sz[iu,ju,ku,au,t]

            for j in range(i,n):

                av = j % n_atm
                kv = j // n_atm % nw
                jv = j // n_atm // nw % nv
                iv = j // n_atm // nw // nv % nu

                vx = Sx[iv,jv,kv,av,t]
                vy = Sy[iv,jv,kv,av,t]
                vz = Sz[iv,jv,kv,av,t]

                k = j+n*i-(i+1)*i // 2

                if i != j:
                    e[i,0,0,t] += Q[k,0]*ux*vx
                    e[i,0,1,t] += Q[k,5]*ux*vy
                    e[i,0,2,t] += Q[k,4]*ux*vz

                    e[i,1,0,t] += Q[k,5]*uy*vx
                    e[i,1,1,t] += Q[k,1]*uy*vy
                    e[i,1,2,t] += Q[k,3]*uy*vz

                    e[i,2,0,t] += Q[k,4]*uz*vx
                    e[i,2,1,t] += Q[k,3]*uz*vy
                    e[i,2,2,t] += Q[k,2]*uz*vz

                    e[j,0,0,t] += Q[k,0]*vx*ux
                    e[j,0,1,t] += Q[k,5]*vx*uy
                    e[j,0,2,t] += Q[k,4]*vx*uz

                    e[j,1,0,t] += Q[k,5]*vy*ux
                    e[j,1,1,t] += Q[k,1]*vy*uy
                    e[j,1,2,t] += Q[k,3]*vy*uz

                    e[j,2,0,t] += Q[k,4]*vz*ux
                    e[j,2,1,t] += Q[k,3]*vz*uy
                    e[j,2,2,t] += Q[k,2]*vz*uz
                else:
                    e[i,0,0,t] += Q[k,0]*ux*vx
                    e[i,0,1,t] += Q[k,5]*ux*vy
                    e[i,0,2,t] += Q[k,4]*ux*vz

                    e[i,1,0,t] += Q[k,5]*uy*vx
                    e[i,1,1,t] += Q[k,1]*uy*vy
                    e[i,1,2,t] += Q[k,3]*uy*vz

                    e[i,2,0,t] += Q[k,4]*uz*vx
                    e[i,2,1,t] += Q[k,3]*uz*vy
                    e[i,2,2,t] += Q[k,2]*uz*vz

    return e_np

def dipole_dipole_interaction_potential(double [:,:,:,:,::1] Sx,
                                        double [:,:,:,:,::1] Sy,
                                        double [:,:,:,:,::1] Sz,
                                        double [:,::1] Q):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    cdef Py_ssize_t n_temp = Sx.shape[4]

    cdef Py_ssize_t n = nu*nv*nw*n_atm

    cdef Py_ssize_t i, j, k, t

    cdef Py_ssize_t iu, ju, ku, au
    cdef Py_ssize_t iv, jv, kv, av

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    p_np = np.zeros((n,3,3,n_temp))

    cdef double [:,:,:,::1] p = p_np

    for t in range(n_temp):

        for i in range(n):

            au = i % n_atm
            ku = i // n_atm % nw
            ju = i // n_atm // nw % nv
            iu = i // n_atm // nw // nv % nu

            ux = Sx[iu,ju,ku,au,t]
            uy = Sy[iu,ju,ku,au,t]
            uz = Sz[iu,ju,ku,au,t]

            for j in range(i,n):

                av = j % n_atm
                kv = j // n_atm % nw
                jv = j // n_atm // nw % nv
                iv = j // n_atm // nw // nv % nu

                vx = Sx[iv,jv,kv,av,t]
                vy = Sy[iv,jv,kv,av,t]
                vz = Sz[iv,jv,kv,av,t]

                k = j+n*i-(i+1)*i // 2

                if i != j:
                    p[i,0,0,t] += Q[k,0]*vx
                    p[i,0,1,t] += Q[k,5]*vy
                    p[i,0,2,t] += Q[k,4]*vz

                    p[i,1,0,t] += Q[k,5]*vx
                    p[i,1,1,t] += Q[k,1]*vy
                    p[i,1,2,t] += Q[k,3]*vz

                    p[i,2,0,t] += Q[k,4]*vx
                    p[i,2,1,t] += Q[k,3]*vy
                    p[i,2,2,t] += Q[k,2]*vz

                    p[j,0,0,t] += Q[k,0]*ux
                    p[j,0,1,t] += Q[k,5]*uy
                    p[j,0,2,t] += Q[k,4]*uz

                    p[j,1,0,t] += Q[k,5]*ux
                    p[j,1,1,t] += Q[k,1]*uy
                    p[j,1,2,t] += Q[k,3]*uz

                    p[j,2,0,t] += Q[k,4]*ux
                    p[j,2,1,t] += Q[k,3]*uy
                    p[j,2,2,t] += Q[k,2]*uz
                else:
                    p[i,0,0,t] += Q[k,0]*ux
                    p[i,0,1,t] += Q[k,5]*uy
                    p[i,0,2,t] += Q[k,4]*uz

                    p[i,1,0,t] += Q[k,5]*ux
                    p[i,1,1,t] += Q[k,1]*uy
                    p[i,1,2,t] += Q[k,3]*uz

                    p[i,2,0,t] += Q[k,4]*ux
                    p[i,2,1,t] += Q[k,3]*uy
                    p[i,2,2,t] += Q[k,2]*uz

    return p_np

def magnetic_energy(double [:,:,:,:,::1] Sx,
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

cdef (double, bint) annealing_vector(double [:,:,:,:,::1] Sx,
                                     double [:,:,:,:,::1] Sy,
                                     double [:,:,:,:,::1] Sz,
                                     double sx,
                                     double sy,
                                     double sz,
                                     double [::1] H,
                                     double E,
                                     double [::1] beta,
                                     double [::1] count,
                                     double [::1] total,
                                     Py_ssize_t i,
                                     Py_ssize_t j,
                                     Py_ssize_t k,
                                     Py_ssize_t a,
                                     Py_ssize_t t):

    cdef bint flip = False

    if (random_uniform() < alpha(E, beta[t])):

        Sx[i,j,k,a,t] = sx
        Sy[i,j,k,a,t] = sy
        Sz[i,j,k,a,t] = sz

        H[t] += E
        count[t] += 1

        flip = True

    total[t] += 1

    return count[t]/total[t], flip

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
                     bint [:,::1] pair_ij,
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

    cdef double Ej, E = 0

    cdef double wx, wy, wz

    cdef double dx, dy, dz

    dx, dy, dz = vx-ux, vy-uy, vz-uz

    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]

    for p in prange(n_pairs, nogil=True):

        i_ = (i+img_ind_i[a,p]+nu)%nu
        j_ = (j+img_ind_j[a,p]+nv)%nv
        k_ = (k+img_ind_k[a,p]+nw)%nw
        a_ = atm_ind[a,p]

        q = pair_ind[a,p]
        f = pair_ij[a,p]

        wx, wy, wz = Sx[i_,j_,k_,a_,t], Sy[i_,j_,k_,a_,t], Sz[i_,j_,k_,a_,t]

        if (f == 1):
            Ej = -(dx*(J[q,0,0]*wx+J[q,1,0]*wy+J[q,2,0]*wz)\
                 + dy*(J[q,0,1]*wx+J[q,1,1]*wy+J[q,2,1]*wz)\
                 + dz*(J[q,0,2]*wx+J[q,1,2]*wy+J[q,2,2]*wz))
        else:
            Ej = -(dx*(J[q,0,0]*wx+J[q,0,1]*wy+J[q,0,2]*wz)\
                 + dy*(J[q,1,0]*wx+J[q,1,1]*wy+J[q,1,2]*wz)\
                 + dz*(J[q,2,0]*wx+J[q,2,1]*wy+J[q,2,2]*wz))

        E += Ej

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
               double [:,::1] Q,
               long [:,::1] atm_ind,
               long [:,::1] img_ind_i,
               long [:,::1] img_ind_j,
               long [:,::1] img_ind_k,
               long [:,::1] pair_ind,
               bint [:,::1] pair_ij,
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

    cdef Py_ssize_t i_ind

    cdef bint long_range = np.any(Q)
    cdef bint flip

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double E

    cdef double [::1] H = np.zeros(n_temp)

    cdef double [:,:,:,:,:,::1] e = magnetic_energy(Sx, Sy, Sz, J, A, g, B,
                                                    atm_ind, img_ind_i,
                                                    img_ind_j, img_ind_k,
                                                    pair_ind, pair_ij)

    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    for p in range(n_pairs+2):
                        for t in range(n_temp):
                            H[t] += e[i,j,k,a,p,t]

    cdef double [:,:,:,::1] V = dipole_dipole_interaction_energy(Sx, Sy, Sz, Q)

    cdef double [:,:,:,::1] U = dipole_dipole_interaction_potential(Sx,
                                                                    Sy,
                                                                    Sz, Q)

    n = nu*nv*nw*n_atm

    if long_range:
        for i_ind in range(n):
            for t in range(n_temp):
                H[t] += V[i_ind,0,0,t]+V[i_ind,0,1,t]+V[i_ind,0,2,t]\
                      + V[i_ind,1,0,t]+V[i_ind,1,1,t]+V[i_ind,1,2,t]\
                      + V[i_ind,2,0,t]+V[i_ind,2,1,t]+V[i_ind,2,2,t]

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double [::1] sigma = np.full(n_temp, 1.)

    cdef double [::1] count = np.zeros(n_temp)
    cdef double [::1] total = np.zeros(n_temp)

    cdef double rate, factor

    cdef double [::1] beta = 1/(kB*np.copy(T_range))

    for _ in range(N):

        for _ in range(n):

            i, j, k, a = random_original(nu, nv, nw, n_atm)

            for t in range(n_temp):

                ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

                vx, vy, vz = gaussian_vector_candidate(ux, uy, uz, sigma[t])

                E = magnetic(Sx, Sy, Sz, vx, vy, vz, ux, uy, uz, J, A, g, B,
                             atm_ind, img_ind_i, img_ind_j, img_ind_k,
                             pair_ind, pair_ij, i, j, k, a, t)

                if long_range:

                    i_ind = a+n_atm*(k+nw*(j+nv*i))

                    E += energy_moment(U, Q, vx, vy, vz,
                                       ux, uy, uz, i_ind, t)

                rate, flip = annealing_vector(Sx, Sy, Sz, vx, vy, vz, H, E,
                                              beta, count, total,
                                              i, j, k, a, t)

                if long_range and flip:

                    update_moment(U, Q, vx, vy, vz, ux, uy, uz, i_ind, t)

                if (rate > 0.0 and rate < 1.0):
                    factor = rate/(1.0-rate)
                    sigma[t] *= factor

                    if (sigma[t] < 0.01): sigma[t] = 0.01
                    if (sigma[t] > 10): sigma[t] = 10

            replica_exchange(H, beta, sigma)

    return np.copy(H), 1/(kB*np.copy(beta))

# ---

cdef (double, bint) annealing_cluster(double [:,:,:,:,::1] Sx,
                                      double [:,:,:,:,::1] Sy,
                                      double [:,:,:,:,::1] Sz,
                                      double [::1] clust_sx,
                                      double [::1] clust_sy,
                                      double [::1] clust_sz,
                                      double [:,:,::1] J,
                                      double [:,::,::1] A,
                                      double [:,::,::1] g,
                                      double [::1] B,
                                      Py_ssize_t [::1] clust_i,
                                      Py_ssize_t [::1] clust_j,
                                      Py_ssize_t [::1] clust_k,
                                      Py_ssize_t [::1] clust_a,
                                      double [:,:,:,::1] h_eff,
                                      bint [:,:,:,::1] b,
                                      bint [:,:,:,::1] c,
                                      Py_ssize_t n_c,
                                      long [:,::1] atm_ind,
                                      long [:,::1] img_ind_i,
                                      long [:,::1] img_ind_j,
                                      long [:,::1] img_ind_k,
                                      long [:,::1] pair_ind,
                                      bint [:,::1] pair_ij,
                                      double [::1] H,
                                      double Eij,
                                      double [::1] beta,
                                      double [::1] count,
                                      double [::1] total,
                                      Py_ssize_t t):

    cdef bint flip = False

    cdef Py_ssize_t n_temp = H.shape[0]

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double Ec = 0
    cdef double Ek, Eb

    cdef Py_ssize_t p, i_c, i_ind

    cdef bint f

    cdef Py_ssize_t i, j, k, a
    cdef Py_ssize_t i_, j_, k_, a_

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double dx, dy, dz

    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]

    for i_c in prange(n_c, nogil=True):

        i = clust_i[i_c]
        j = clust_j[i_c]
        k = clust_k[i_c]
        a = clust_a[i_c]

        ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

        vx = clust_sx[i_c]
        vy = clust_sy[i_c]
        vz = clust_sz[i_c]

        dx = vx-ux
        dy = vy-uy
        dz = vz-uz

        Ek = -(vx*(A[a,0,0]*vx+A[a,0,1]*vy+A[a,0,2]*vz)+\
               vy*(A[a,1,0]*vx+A[a,1,1]*vy+A[a,1,2]*vz)+\
               vz*(A[a,2,0]*vx+A[a,2,1]*vy+A[a,2,2]*vz))\
             +(ux*(A[a,0,0]*ux+A[a,0,1]*uy+A[a,0,2]*uz)+\
               uy*(A[a,1,0]*ux+A[a,1,1]*uy+A[a,1,2]*uz)+\
               uz*(A[a,2,0]*ux+A[a,2,1]*uy+A[a,2,2]*uz))

        Eb = -(Bx*(g[a,0,0]*dx+g[a,0,1]*dy+g[a,0,2]*dz)+\
               By*(g[a,1,0]*dx+g[a,1,1]*dy+g[a,1,2]*dz)+\
               Bz*(g[a,2,0]*dx+g[a,2,1]*dy+g[a,2,2]*dz))

        Ec += Ek+Eb+h_eff[i,j,k,a]

        b[i,j,k,a] = 0
        c[i,j,k,a] = 0

        h_eff[i,j,k,a] = 0

    if (random_uniform() < alpha(Ec, beta[t])):

        for i_c in prange(n_c, nogil=True):

            i = clust_i[i_c]
            j = clust_j[i_c]
            k = clust_k[i_c]
            a = clust_a[i_c]

            vx = clust_sx[i_c]
            vy = clust_sy[i_c]
            vz = clust_sz[i_c]

            Sx[i,j,k,a,t] = vx
            Sy[i,j,k,a,t] = vy
            Sz[i,j,k,a,t] = vz

        H[t] += Ec+Eij
        count[t] += 1

        flip = True

    total[t] += 1

    return count[t]/total[t], flip

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
                                 Py_ssize_t [::1] clust_i,
                                 Py_ssize_t [::1] clust_j,
                                 Py_ssize_t [::1] clust_k,
                                 Py_ssize_t [::1] clust_a,
                                 Py_ssize_t [::1] pairs_i,
                                 Py_ssize_t [::1] pairs_j,
                                 Py_ssize_t [::1] pairs_k,
                                 Py_ssize_t [::1] pairs_a,
                                 bint [::1] activated,
                                 double [:,:,:,::1] h_eff,
                                 double [:,:,:,:,::1] h_eff_ij,
                                 bint [:,:,:,::1] b,
                                 bint [:,:,:,::1] c,
                                 Py_ssize_t n_c,
                                 long [:,::1] atm_ind,
                                 long [:,::1] img_ind_i,
                                 long [:,::1] img_ind_j,
                                 long [:,::1] img_ind_k,
                                 long [:,::1] pair_ind,
                                 long [:,::1] pair_inv,
                                 bint [:,::1] pair_ij,
                                 double [::1] beta,
                                 Py_ssize_t i,
                                 Py_ssize_t j,
                                 Py_ssize_t k,
                                 Py_ssize_t a,
                                 Py_ssize_t t):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef Py_ssize_t n = nu*nv*nw*n_atm

    cdef double E

    cdef Py_ssize_t i_, j_, k_, a_, p_, p, q

    cdef bint f

    cdef double vx, vy, vz
    cdef double vx_perp, vy_perp, vz_perp

    cdef double J_eff, J_eff_ij, J_eff_ji

    cdef double Jx_eff_ij, Jy_eff_ij, Jz_eff_ij
    cdef double Jx_eff_ji, Jy_eff_ji, Jz_eff_ji

    cdef double n_dot_v

    for p in prange(n_pairs, nogil=True):

        i_ = (i+img_ind_i[a,p]+nu)%nu
        j_ = (j+img_ind_j[a,p]+nv)%nv
        k_ = (k+img_ind_k[a,p]+nw)%nw
        a_ = atm_ind[a,p]
        p_ = pair_inv[a,p]

        pairs_i[p] = i_
        pairs_j[p] = j_
        pairs_k[p] = k_
        pairs_a[p] = a_

        activated[p] = 0

        if (b[i_,j_,k_,a_] == 0):

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

            h_eff_ij[i_,j_,k_,a_,p_] = 2*J_eff_ij*n_dot_v

            J_eff_ji = vx_perp*Jx_eff_ji+vy_perp*Jy_eff_ji+vz_perp*Jz_eff_ji

            h_eff_ij[i,j,k,a,p] = 2*J_eff_ji*n_dot_u

            if (c[i_,j_,k_,a_] == 0):

                J_eff = nx*Jx_eff_ij+ny*Jy_eff_ij+nz*Jz_eff_ij

                E = 2*J_eff*n_dot_u*n_dot_v

                if (random_uniform_parallel(i_,j_,k_,a_) >= alpha(E, beta[t])):

                    c[i_,j_,k_,a_] = 1

                    activated[p] = 1

    for p in range(n_pairs):

        h_eff[i,j,k,a] += h_eff_ij[i,j,k,a,p]

        if (activated[p] == 1):

            if (n_c < n):

                clust_i[n_c] = pairs_i[p]
                clust_j[n_c] = pairs_j[p]
                clust_k[n_c] = pairs_k[p]
                clust_a[n_c] = pairs_a[p]

                n_c += 1

    return n_c

cdef double boundary_energy(double [:,:,:,:,::1] Sx,
                            double [:,:,:,:,::1] Sy,
                            double [:,:,:,:,::1] Sz,
                            double nx,
                            double ny,
                            double nz,
                            double [:,:,::1] J,
                            Py_ssize_t [::1] clust_i,
                            Py_ssize_t [::1] clust_j,
                            Py_ssize_t [::1] clust_k,
                            Py_ssize_t [::1] clust_a,
                            bint [:,:,:,::1] c,
                            Py_ssize_t n_c,
                            long [:,::1] atm_ind,
                            long [:,::1] img_ind_i,
                            long [:,::1] img_ind_j,
                            long [:,::1] img_ind_k,
                            long [:,::1] pair_ind,
                            long [:,::1] pair_inv,
                            bint [:,::1] pair_ij,
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

    for i_c in range(n_c):

        i = clust_i[i_c]
        j = clust_j[i_c]
        k = clust_k[i_c]
        a = clust_a[i_c]

        ux, uy, uz = Sx[i,j,k,a,t], Sy[i,j,k,a,t], Sz[i,j,k,a,t]

        n_dot_u = ux*nx+uy*ny+uz*nz

        for p in prange(n_pairs, nogil=True):

            i_ = (i+img_ind_i[a,p]+nu)%nu
            j_ = (j+img_ind_j[a,p]+nv)%nv
            k_ = (k+img_ind_k[a,p]+nw)%nw
            a_ = atm_ind[a,p]

            if (c[i_,j_,k_,a_] == 0):

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
                       double [:,::1] Q,
                       long [:,::1] atm_ind,
                       long [:,::1] img_ind_i,
                       long [:,::1] img_ind_j,
                       long [:,::1] img_ind_k,
                       long [:,::1] pair_ind,
                       long [:,::1] pair_inv,
                       bint [:,::1] pair_ij,
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

    cdef Py_ssize_t i_ind

    cdef bint long_range = np.any(Q)
    cdef bint flip

    cdef Py_ssize_t n_pairs = atm_ind.shape[1]

    cdef double E

    cdef double [::1] H = np.zeros(n_temp)

    cdef double [:,:,:,:,:,::1] e = magnetic_energy(Sx, Sy, Sz, J, A, g, B,
                                                    atm_ind, img_ind_i,
                                                    img_ind_j, img_ind_k,
                                                    pair_ind, pair_ij)

    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    for p in range(n_pairs+2):
                        for t in range(n_temp):
                            H[t] += e[i,j,k,a,p,t]

    cdef double [:,:,:,::1] V = dipole_dipole_interaction_energy(Sx, Sy, Sz, Q)

    cdef double [:,:,:,::1] U = dipole_dipole_interaction_potential(Sx,
                                                                    Sy,
                                                                    Sz, Q)

    n = nu*nv*nw*n_atm

    if long_range:
        for i_ind in range(n):
            for t in range(n_temp):
                H[t] += V[i_ind,0,0,t]+V[i_ind,0,1,t]+V[i_ind,0,2,t]\
                      + V[i_ind,1,0,t]+V[i_ind,1,1,t]+V[i_ind,1,2,t]\
                      + V[i_ind,2,0,t]+V[i_ind,2,1,t]+V[i_ind,2,2,t]

    cdef Py_ssize_t i_c, m_c, n_c

    cdef bint [:,:,:,::1] b = np.zeros((nu,nv,nw,n_atm), dtype=np.intc)
    cdef bint [:,:,:,::1] c = np.zeros((nu,nv,nw,n_atm), dtype=np.intc)

    pairs_shape = (nu,nv,nw,n_atm,n_pairs)

    cdef double [:,:,:,::1] h_eff = np.zeros((nu,nv,nw,n_atm), dtype=float)
    cdef double [:,:,:,:,::1] h_eff_ij = np.zeros(pairs_shape, dtype=float)

    cdef Py_ssize_t [::1] clust_i = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_j = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_k = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_a = np.zeros(n, dtype=np.intp)

    cdef Py_ssize_t [::1] clust_ind = np.zeros(n, dtype=np.intp)

    cdef Py_ssize_t [::1] pairs_i = np.zeros(n_pairs, dtype=np.intp)
    cdef Py_ssize_t [::1] pairs_j = np.zeros(n_pairs, dtype=np.intp)
    cdef Py_ssize_t [::1] pairs_k = np.zeros(n_pairs, dtype=np.intp)
    cdef Py_ssize_t [::1] pairs_a = np.zeros(n_pairs, dtype=np.intp)

    cdef Py_ssize_t [::1] pairs_ind = np.zeros(n_pairs, dtype=np.intp)

    cdef bint [::1] activated = np.zeros(n_pairs, dtype=np.intc)

    cdef double ux, uy, uz
    cdef double vx, vy, vz

    cdef double [::1] clust_ux = np.zeros(n)
    cdef double [::1] clust_uy = np.zeros(n)
    cdef double [::1] clust_uz = np.zeros(n)

    cdef double [::1] clust_vx = np.zeros(n)
    cdef double [::1] clust_vy = np.zeros(n)
    cdef double [::1] clust_vz = np.zeros(n)

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

                nx, ny, nz = gaussian_vector_candidate(ux, uy, uz, sigma[t])

                clust_i[0] = i
                clust_j[0] = j
                clust_k[0] = k
                clust_a[0] = a

                c[i,j,k,a] = 1

                i_c, m_c = 0, 1

                n_c = m_c

                while (i_c < m_c):

                    i_ = clust_i[i_c]
                    j_ = clust_j[i_c]
                    k_ = clust_k[i_c]
                    a_ = clust_a[i_c]

                    vx = Sx[i_,j_,k_,a_,t]
                    vy = Sy[i_,j_,k_,a_,t]
                    vz = Sz[i_,j_,k_,a_,t]

                    n_dot_v = vx*nx+vy*ny+vz*nz

                    vx_perp = vx-nx*n_dot_v
                    vy_perp = vy-ny*n_dot_v
                    vz_perp = vz-nz*n_dot_v

                    b[i_,j_,k_,a_] = 1

                    clust_ind[i_c] = a_+n_atm*(k_+nw*(j_+nv*i_))

                    clust_ux[i_c] = vx
                    clust_uy[i_c] = vy
                    clust_uz[i_c] = vz

                    clust_vx[i_c] = vx_perp-nx*n_dot_v
                    clust_vy[i_c] = vy_perp-ny*n_dot_v
                    clust_vz[i_c] = vz_perp-nz*n_dot_v

                    m_c = magnetic_cluster(Sx, Sy, Sz, nx, ny, nz,
                                           vx_perp, vy_perp, vz_perp,
                                           n_dot_v, J,
                                           clust_i, clust_j, clust_k, clust_a,
                                           pairs_i, pairs_j, pairs_k, pairs_a,
                                           activated, h_eff, h_eff_ij,
                                           b, c, n_c, atm_ind,
                                           img_ind_i, img_ind_j, img_ind_k,
                                           pair_ind, pair_inv, pair_ij, beta,
                                           i_, j_, k_, a_, t)

                    n_c = m_c

                    i_c += 1

                E = boundary_energy(Sx, Sy, Sz, nx, ny, nz, J,
                                    clust_i, clust_j, clust_k, clust_a, c, n_c,
                                    atm_ind, img_ind_i, img_ind_j, img_ind_k,
                                    pair_ind, pair_inv, pair_ij, t)

                if long_range:

                    i_ind = a+n_atm*(k+nw*(j+nv*i))

                    E += energy_moment_cluster(U, Q,
                                               clust_vx, clust_vy, clust_vz,
                                               clust_ux, clust_uy, clust_uz,
                                               clust_ind, n_c, t)

                rate, flip = annealing_cluster(Sx, Sy, Sz,
                                               clust_vx, clust_vy, clust_vz,
                                               J, A, g, B,
                                               clust_i, clust_j, clust_k,
                                               clust_a, h_eff,
                                               b, c, n_c, atm_ind,
                                               img_ind_i, img_ind_j, img_ind_k,
                                               pair_ind, pair_ij,
                                               H, E, beta, count, total, t)

                if long_range and flip:

                    update_moment_cluster(U, Q,
                                          clust_vx, clust_vy, clust_vz,
                                          clust_ux, clust_uy, clust_uz,
                                          clust_ind, n_c, t)

                # if (rate > 0.0 and rate < 1.0):
                #     factor = rate/(1.0-rate)
                #     sigma[t] *= factor

                #     if (sigma[t] < 0.1): sigma[t] = 0.1
                #     if (sigma[t] > 10): sigma[t] = 10

                # print(sigma[t],rate,factor)

                for i in range(nu):
                    for j in range(nv):
                        for k in range(nw):
                            for a in range(n_atm):
                                c[i,j,k,a] = 0
                                b[i,j,k,a] = 0

            replica_exchange(H, beta, sigma)

    return np.copy(H), 1/(kB*np.copy(beta))