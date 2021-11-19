#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.math cimport M_PI, sin, cos, acos, fabs, log, exp, sqrt, atan2

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

cdef double MACHINE_EPSILON = np.finfo(float).eps
    
cdef void initialize_random(Py_ssize_t nu,
                            Py_ssize_t nv, 
                            Py_ssize_t nw,
                            Py_ssize_t n_atm):
    
    global gen, dist, dist_u, dist_v, dist_w, dist_atm
    
    gen = mt19937(20)
    
    dist = uniform_real_distribution[double](0.0,1.0)
    
    dist_u = uniform_int_distribution[Py_ssize_t](0,nu-1)
    dist_v = uniform_int_distribution[Py_ssize_t](0,nv-1)
    dist_w = uniform_int_distribution[Py_ssize_t](0,nw-1)
    dist_atm = uniform_int_distribution[Py_ssize_t](0,n_atm-1)
    
cdef bint iszero(double a) nogil:
    
    cdef double atol = 1e-08
    
    return fabs(a) <= atol

cdef double random():
    
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

    cdef double theta = 2*M_PI*random()
    cdef double phi = acos(1-2*random())

    cdef double sx = sin(phi)*cos(theta)
    cdef double sy = sin(phi)*sin(theta)
    cdef double sz = cos(phi)

    return sx, sy, sz
    
cdef (double, double, double) random_vector_length_candidate():

    cdef double theta = 2*M_PI*random()
    cdef double phi = acos(1-2*random())

    cdef double s = random()

    cdef double sx = s*sin(phi)*cos(theta)
    cdef double sy = s*sin(phi)*sin(theta)
    cdef double sz = s*cos(phi)

    return sx, sy, sz

cdef (double, double, double) ising_vector_candidate(double ux,
                                                     double uy,
                                                     double uz):

    return -ux, -uy, -uz

cdef void annealing_vector(double [:,:,:,::1] Sx,
                           double [:,:,:,::1] Sy,
                           double [:,:,:,::1] Sz,
                           double sx,
                           double sy,
                           double sz,
                           double E,
                           double beta,                           
                           Py_ssize_t i,
                           Py_ssize_t j,
                           Py_ssize_t k,
                           Py_ssize_t a):


    if (random() < alpha(E, beta)):
        Sx[i,j,k,a] = sx
        Sy[i,j,k,a] = sy
        Sz[i,j,k,a] = sz
        
cdef double magnetic(double [:,:,:,::1] Sx,
                     double [:,:,:,::1] Sy,
                     double [:,:,:,::1] Sz,
                     double vx,
                     double vy,
                     double vz,
                     double ux,
                     double uy,
                     double uz,
                     double [:,:,::1] J,
                     double [:,::1] A,
                     double [:,::1] g,
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
                     Py_ssize_t a):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    
    cdef Py_ssize_t n_pairs = atm_ind.shape[1]
    
    cdef Py_ssize_t i_, j_, k_, a_, t, p
    
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
                
        wx, wy, wz = Sx[i_,j_,k_,a_], Sy[i_,j_,k_,a_], Sz[i_,j_,k_,a_]
        
        t = pair_ind[a,p]
        
        f = pair_ij[a,p]
                
        if (f == 1):
            E -= dx*(J[t,0,0]*wx+J[t,1,0]*wy+J[t,2,0]*wz)\
              +  dy*(J[t,0,1]*wx+J[t,1,1]*wy+J[t,2,1]*wz)\
              +  dz*(J[t,0,2]*wx+J[t,1,2]*wy+J[t,2,2]*wz)
        else:
            E -= dx*(J[t,0,0]*wx+J[t,0,1]*wy+J[t,0,2]*wz)\
              +  dy*(J[t,1,0]*wx+J[t,1,1]*wy+J[t,1,2]*wz)\
              +  dz*(J[t,2,0]*wx+J[t,2,1]*wy+J[t,2,2]*wz)
              
    E -= vx*(A[0,0]*vx+A[0,1]*vy+A[0,2]*vz)\
      +  vy*(A[1,0]*vx+A[1,1]*vy+A[1,2]*vz)\
      +  vz*(A[2,0]*vx+A[2,1]*vy+A[2,2]*vz)
      
    E += ux*(A[0,0]*ux+A[0,1]*uy+A[0,2]*uz)\
      +  uy*(A[1,0]*ux+A[1,1]*uy+A[1,2]*uz)\
      +  uz*(A[2,0]*ux+A[2,1]*uy+A[2,2]*uz)
              
    E -= Bx*(g[0,0]*dx+g[0,1]*dy+g[0,2]*dz)\
      +  By*(g[1,0]*dx+g[1,1]*dy+g[1,2]*dz)\
      +  Bz*(g[2,0]*dx+g[2,1]*dy+g[2,2]*dz)

    return E

def heisenberg(double [:,:,:,::1] Sx,
               double [:,:,:,::1] Sy,
               double [:,:,:,::1] Sz,
               double [:,:,::1] J,
               double [:,::1] A,
               double [:,::1] g,
               double [::1] B,
               long [:,::1] atm_ind,
               long [:,::1] img_ind_i,
               long [:,::1] img_ind_j,
               long [:,::1] img_ind_k,
               long [:,::1] pair_ind,
               long [:,::1] pair_ij,
               double [::1] T_range,
               double kB,
               Py_ssize_t N,
               bint logarithmic,
               bint continuous):
        
    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    
    initialize_random(nu, nv, nw, n_atm)

    cdef Py_ssize_t i, j, k, a, t
    cdef double E
    
    cdef double ux, uy, uz
    cdef double vx, vy, vz
       
    cdef double constant
    
    if (N <= 1):
        constant = 0
    else:
        if logarithmic:
            constant = (log(1/kB/T_range[0])-log(1/kB/T_range[1]))/(N-1)
        else:
            constant = (1/kB/T_range[0]-1/kB/T_range[1])/(N-1)
            
    cdef double beta = 1/kB/T_range[0]
        
    for t in range(N):

        i, j, k, a = random_original(nu, nv, nw, n_atm)

        ux, uy, uz = Sx[i,j,k,a], Sy[i,j,k,a], Sz[i,j,k,a]
        
        if continuous:
            vx, vy, vz = random_vector_candidate()
        else:
            vx, vy, vz = ising_vector_candidate(ux, uy, uz)
                        
        E = magnetic(Sx, Sy, Sz, vx, vy, vz, ux, uy, uz, J, A, g, B, atm_ind, 
                     img_ind_i, img_ind_j, img_ind_k, pair_ind, pair_ij,
                     i, j, k, a)
        
        annealing_vector(Sx, Sy, Sz, vx, vy, vz, E, beta, i, j, k, a)
        
        if (logarithmic):
            beta = exp(log(beta)-constant)
        else:
            beta = beta-constant
            
def energy(double [:,:,:,::1] Sx,
           double [:,:,:,::1] Sy,
           double [:,:,:,::1] Sz,
           double [:,:,::1] J,
           double [:,::1] A,
           double [:,::1] g,
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
    
    cdef Py_ssize_t n_pairs = atm_ind.shape[1]
    
    cdef Py_ssize_t i, j, k, a
    
    cdef Py_ssize_t i_, j_, k_, a_, t, p
    
    cdef bint f

    e_np = np.zeros((nu,nv,nw,n_atm,n_pairs+2))
    
    cdef double [:,:,:,:,::1] e = e_np
        
    cdef double ux, uy, uz, vx, vy, vz
    
    cdef double Bx = B[0]
    cdef double By = B[1]
    cdef double Bz = B[2]
    
    for i in range(nu):
        for j in range(nv):
            for k in range(nw):
                for a in range(n_atm):
                    
                    ux, uy, uz = Sx[i,j,k,a], Sy[i,j,k,a], Sz[i,j,k,a]
                       
                    e[i,j,k,a,n_pairs] = -(ux*(A[0,0]*ux+A[0,1]*uy+A[0,2]*uz)\
                                         + uy*(A[1,0]*ux+A[1,1]*uy+A[1,2]*uz)\
                                         + uz*(A[2,0]*ux+A[2,1]*uy+A[2,2]*uz))
                                                
                    e[i,j,k,a,n_pairs+1] = \
                                         -(Bx*(g[0,0]*ux+g[0,1]*uy+g[0,2]*uz)\
                                         + By*(g[1,0]*ux+g[1,1]*uy+g[1,2]*uz)\
                                         + Bz*(g[2,0]*ux+g[2,1]*uy+g[2,2]*uz))
                                                
                    for p in range(n_pairs):
                        
                        i_ = (i+img_ind_i[a,p]+nu)%nu
                        j_ = (j+img_ind_j[a,p]+nv)%nv
                        k_ = (k+img_ind_k[a,p]+nw)%nw
                        a_ = atm_ind[a,p]
                                
                        vx = Sx[i_,j_,k_,a_] 
                        vy = Sy[i_,j_,k_,a_]
                        vz = Sz[i_,j_,k_,a_]
                        
                        t = pair_ind[a,p]
                        
                        f = pair_ij[a,p]
                        
                        if (f == 1):
                            e[i,j,k,a,p] = \
                                    -(ux*(J[t,0,0]*vx+J[t,1,0]*vy+J[t,2,0]*vz)\
                                    + uy*(J[t,0,1]*vx+J[t,1,1]*vy+J[t,2,1]*vz)\
                                    + uz*(J[t,0,2]*vx+J[t,1,2]*vy+J[t,2,2]*vz))
                        else:
                            e[i,j,k,a,p] = \
                                    -(ux*(J[t,0,0]*vx+J[t,0,1]*vy+J[t,0,2]*vz)\
                                    + uy*(J[t,1,0]*vx+J[t,1,1]*vy+J[t,1,2]*vz)\
                                    + uz*(J[t,2,0]*vx+J[t,2,1]*vy+J[t,2,2]*vz))                                       
    return e_np

# ---

cdef void annealing_cluster(double [:,:,:,::1] Sx,
                            double [:,:,:,::1] Sy,
                            double [:,:,:,::1] Sz,
                            double nx,
                            double ny,
                            double nz,
                            double [:,:,::1] J,
                            double [:,::1] A,
                            double [:,::1] g,
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
                            long [:,::1] pair_ij,
                            double beta,
                            bint continuous):
    
    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    
    cdef Py_ssize_t n_pairs = atm_ind.shape[1]
    
    cdef double E = 0
        
    cdef Py_ssize_t t, p, i_c
    
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

    for i_c in range(n_c):
        
        i = clust_i[i_c] 
        j = clust_j[i_c]
        k = clust_k[i_c]
        a = clust_a[i_c]
                
        ux, uy, uz = Sx[i,j,k,a], Sy[i,j,k,a], Sz[i,j,k,a]

        n_dot_u = nx*ux+ny*uy+nz*uz
                
        vx = ux-2*nx*n_dot_u
        vy = uy-2*ny*n_dot_u
        vz = uz-2*nz*n_dot_u      
                    
        E -= vx*(A[0,0]*vx+A[0,1]*vy+A[0,2]*vz)\
          +  vy*(A[1,0]*vx+A[1,1]*vy+A[1,2]*vz)\
          +  vz*(A[2,0]*vx+A[2,1]*vy+A[2,2]*vz)
          
        E += ux*(A[0,0]*ux+A[0,1]*uy+A[0,2]*uz)\
          +  uy*(A[1,0]*ux+A[1,1]*uy+A[1,2]*uz)\
          +  uz*(A[2,0]*ux+A[2,1]*uy+A[2,2]*uz)
          
        dx = vx-ux
        dy = vy-uy
        dz = vz-uz
          
        E -= Bx*(g[0,0]*dx+g[0,1]*dy+g[0,2]*dz)\
          +  By*(g[1,0]*dx+g[1,1]*dy+g[1,2]*dz)\
          +  Bz*(g[2,0]*dx+g[2,1]*dy+g[2,2]*dz)
        
        E += h_eff[i,j,k,a]
         
        b[i,j,k,a] = 0
        c[i,j,k,a] = 0
        
        h_eff[i,j,k,a] = 0

    if (random() < alpha(E, beta)):
                
        for i_c in range(n_c):
                    
            i = clust_i[i_c] 
            j = clust_j[i_c]
            k = clust_k[i_c]
            a = clust_a[i_c]
                        
            ux, uy, uz = Sx[i,j,k,a], Sy[i,j,k,a], Sz[i,j,k,a]
    
            if (continuous):
                
                n_dot_u = nx*ux+ny*uy+nz*uz
                
                vx = ux-2*nx*n_dot_u
                vy = uy-2*ny*n_dot_u
                vz = uz-2*nz*n_dot_u
                
            else:
                
                vx = -ux
                vy = -uy
                vz = -uz               
            
            Sx[i,j,k,a] = vx
            Sy[i,j,k,a] = vy
            Sz[i,j,k,a] = vz
                
cdef Py_ssize_t magnetic_cluster(double [:,:,:,::1] Sx,
                                 double [:,:,:,::1] Sy,
                                 double [:,:,:,::1] Sz,
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
                                 long [:,::1] pair_ij,
                                 double beta,
                                 Py_ssize_t i,
                                 Py_ssize_t j,
                                 Py_ssize_t k,
                                 Py_ssize_t a):

    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    
    cdef Py_ssize_t n_pairs = atm_ind.shape[1]
    
    cdef Py_ssize_t m_c = n_c
        
    cdef double E
    
    cdef Py_ssize_t i_, j_, k_, a_, t, p, p_
    
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
                      
        if (b[i_,j_,k_,a_] == 0):
        
            vx, vy, vz = Sx[i_,j_,k_,a_], Sy[i_,j_,k_,a_], Sz[i_,j_,k_,a_]
            
            n_dot_v = vx*nx+vy*ny+vz*nz
            
            vx_perp = vx-nx*n_dot_v
            vy_perp = vy-ny*n_dot_v
            vz_perp = vz-nz*n_dot_v
                    
            t = pair_ind[a,p]
            
            f = pair_ij[a,p]
                            
            if (f == 1):
                Jx_eff_ij = J[t,0,0]*nx+J[t,1,0]*ny+J[t,2,0]*nz
                Jy_eff_ij = J[t,0,1]*nx+J[t,1,1]*ny+J[t,2,1]*nz
                Jz_eff_ij = J[t,0,2]*nx+J[t,1,2]*ny+J[t,2,2]*nz
                
                Jx_eff_ji = J[t,0,0]*nx+J[t,0,1]*ny+J[t,0,2]*nz
                Jy_eff_ji = J[t,1,0]*nx+J[t,1,1]*ny+J[t,1,2]*nz
                Jz_eff_ji = J[t,2,0]*nx+J[t,2,1]*ny+J[t,2,2]*nz 
            else:
                Jx_eff_ij = J[t,0,0]*nx+J[t,0,1]*ny+J[t,0,2]*nz
                Jy_eff_ij = J[t,1,0]*nx+J[t,1,1]*ny+J[t,1,2]*nz
                Jz_eff_ij = J[t,2,0]*nx+J[t,2,1]*ny+J[t,2,2]*nz
                
                Jx_eff_ji = J[t,0,0]*nx+J[t,1,0]*ny+J[t,2,0]*nz
                Jy_eff_ji = J[t,0,1]*nx+J[t,1,1]*ny+J[t,2,1]*nz
                Jz_eff_ji = J[t,0,2]*nx+J[t,1,2]*ny+J[t,2,2]*nz
                
            J_eff_ij = ux_perp*Jx_eff_ij+uy_perp*Jy_eff_ij+uz_perp*Jz_eff_ij            
                    
            h_eff_ij[i_,j_,k_,a_,p_] = 2*J_eff_ij*n_dot_v
    
            J_eff_ji = vx_perp*Jx_eff_ji+vy_perp*Jy_eff_ji+vz_perp*Jz_eff_ji
            
            h_eff_ij[i,j,k,a,p] = 2*J_eff_ji*n_dot_u
            
            if (c[i_,j_,k_,a_] == 0):
                
                J_eff = nx*Jx_eff_ij+ny*Jy_eff_ij+nz*Jz_eff_ij
                
                E = 2*J_eff*n_dot_u*n_dot_v
                                                           
                if (random() < 1-alpha(E, beta)):
        
                    clust_i[m_c] = i_
                    clust_j[m_c] = j_
                    clust_k[m_c] = k_
                    clust_a[m_c] = a_
                    
                    c[i_,j_,k_,a_] = 1
                                                                    
                    m_c += 1
                    
            h_eff[i,j,k,a] += h_eff_ij[i,j,k,a,p]

        elif (b[i_,j_,k_,a_] == 1):
            
            h_eff[i,j,k,a] += h_eff_ij[i,j,k,a,p]

    return m_c

def heisenberg_cluster(double [:,:,:,::1] Sx,
                       double [:,:,:,::1] Sy,
                       double [:,:,:,::1] Sz,
                       double [:,:,::1] J,
                       double [:,::1] A,
                       double [:,::1] g,
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
                       Py_ssize_t N,
                       bint logarithmic,
                       bint continuous):
        
    cdef Py_ssize_t nu = Sx.shape[0]
    cdef Py_ssize_t nv = Sx.shape[1]
    cdef Py_ssize_t nw = Sx.shape[2]
    cdef Py_ssize_t n_atm = Sx.shape[3]
    
    initialize_random(nu, nv, nw, n_atm)

    cdef Py_ssize_t t, n
    cdef Py_ssize_t i, j, k, a,
    cdef Py_ssize_t i_, j_, k_, a_
    
    cdef Py_ssize_t n_pairs = atm_ind.shape[1]
        
    n = nu*nv*nw*n_atm
                    
    cdef Py_ssize_t i_c, n_c = 0
    
    cdef bint [:,:,:,::1] b = np.zeros((nu,nv,nw,n_atm), dtype=np.intc)
    cdef bint [:,:,:,::1] c = np.zeros((nu,nv,nw,n_atm), dtype=np.intc)
 
    cdef double [:,:,:,::1] h_eff = np.zeros((nu,nv,nw,n_atm), dtype=float)
    cdef double [:,:,:,:,::1] h_eff_ij = np.zeros((nu,nv,nw,n_atm,n_pairs), 
                                                  dtype=float)
    
    cdef Py_ssize_t [::1] clust_i = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_j = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_k = np.zeros(n, dtype=np.intp)
    cdef Py_ssize_t [::1] clust_a = np.zeros(n, dtype=np.intp)
    
    cdef double ux, uy, uz
    cdef double vx, vy, vz
    
    cdef double mx, my, mz
    cdef double nx, ny, nz
    
    cdef double ux_perp, uy_perp, uz_perp
    cdef double vx_perp, vy_perp, vz_perp

    cdef double u, n_dot_u, n_dot_v
            
    cdef double constant
        
    if (N <= 1):
        constant = 0
    else:
        if logarithmic:
            constant = (log(1/kB/T_range[0])-log(1/kB/T_range[1]))/(N-1)
        else:
            constant = (1/kB/T_range[0]-1/kB/T_range[1])/(N-1)
        
    cdef double beta = 1/kB/T_range[0]
            
    for t in range(N):
       
        i, j, k, a = random_original(nu, nv, nw, n_atm)
        
        ux, uy, uz = Sx[i,j,k,a], Sy[i,j,k,a], Sz[i,j,k,a]

        if (continuous):
            nx, ny, nz = random_vector_candidate()
        else:
            u = sqrt(ux*ux+uy*uy+uz*uz)
            nx, ny, nz = ux/u, uy/u, uz/u
            
        n_dot_u = ux*nx+uy*ny+uz*nz
    
        ux_perp = ux-nx*n_dot_u
        uy_perp = uy-ny*n_dot_u
        uz_perp = uz-nz*n_dot_u
                
        clust_i[0] = i
        clust_j[0] = j
        clust_k[0] = k
        clust_a[0] = a
        
        b[i,j,k,a] = 1
                
        n_c = 1
        
        n_c = magnetic_cluster(Sx, Sy, Sz, nx, ny, nz, 
                               ux_perp, uy_perp, uz_perp, n_dot_u, J, 
                               clust_i, clust_j, clust_k, clust_a,
                               h_eff, h_eff_ij, b, c, n_c, 
                               atm_ind, img_ind_i, img_ind_j, img_ind_k,
                               pair_ind, pair_inv, pair_ij, beta, 
                               i, j, k, a)
                
        i_c = 1
            
        while i_c < n_c:
            
            i_ = clust_i[i_c]
            j_ = clust_j[i_c]
            k_ = clust_k[i_c]
            a_ = clust_a[i_c]
                
            vx, vy, vz = Sx[i_,j_,k_,a_], Sy[i_,j_,k_,a_], Sz[i_,j_,k_,a_]
        
            n_dot_v = vx*nx+vy*ny+vz*nz
    
            vx_perp = vx-nx*n_dot_v
            vy_perp = vy-ny*n_dot_v
            vz_perp = vz-nz*n_dot_v
            
            b[i_,j_,k_,a_] = 1
            
            n_c = magnetic_cluster(Sx, Sy, Sz, nx, ny, nz, 
                                   vx_perp, vy_perp, vz_perp, n_dot_v, J, 
                                   clust_i, clust_j, clust_k, clust_a,
                                   h_eff, h_eff_ij, b, c, n_c,
                                   atm_ind, img_ind_i, img_ind_j, img_ind_k,
                                   pair_ind, pair_inv, pair_ij, beta,
                                   i_, j_, k_, a_)
                                    
            i_c += 1
                
        annealing_cluster(Sx, Sy, Sz, nx, ny, nz, J, A, g, B,
                          clust_i, clust_j, clust_k, clust_a,
                          h_eff, b, c, n_c,
                          atm_ind, img_ind_i, img_ind_j, img_ind_k,
                          pair_ind, pair_ij, beta, continuous)
                                
        if logarithmic:
            beta = exp(log(beta)-constant)
        else:
            beta = beta-constant