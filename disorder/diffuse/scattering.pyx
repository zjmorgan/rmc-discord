#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython
cimport openmp

from libc.math cimport M_PI, cos, sin, exp, sqrt, acos, fabs

from disorder.material import tables

import os, sys

def parallelism():
    
    threads = os.environ.get('OMP_NUM_THREADS')
    
    cdef Py_ssize_t i_thread, num_threads
    
    for i_thread in prange(1, nogil=True):
        num_threads = openmp.omp_get_max_threads()
        
    if (len(sys.argv) > 1):
        openmp.omp_set_num_threads(int(sys.argv[1]))
    elif (threads is None):
        if (num_threads > 8):
            openmp.omp_set_num_threads(8)
        else:
            openmp.omp_set_num_threads(num_threads)
    
    for i_thread in prange(1, nogil=True):
        num_threads = openmp.omp_get_max_threads()

    print('threads:', num_threads)
    
def threads():
    
    cdef Py_ssize_t thread_id = openmp.omp_get_thread_num()
    
    print('id:', thread_id)

cdef double complex cexp(double complex z) nogil:
    cdef double x = z.real
    cdef double y = z.imag
    return exp(x)*(cos(y)+1j*sin(y))

cdef double complex iexp(double y) nogil:
    return cos(y)+1j*sin(y)

def rebin0(double [:,:,::1] a, double [:,::1] comp):
    
    cdef Py_ssize_t m0 = a.shape[0]
    cdef Py_ssize_t m1 = a.shape[1]
    cdef Py_ssize_t m2 = a.shape[2]
    
    cdef Py_ssize_t n0 = comp.shape[0]
    
    ind_i_np, ind_l_np = np.ma.nonzero(comp)
    
    cdef Py_ssize_t [::] ind_i = ind_i_np
    cdef Py_ssize_t [::] ind_l = ind_l_np
      
    cdef Py_ssize_t n = ind_l.shape[0]
    
    b_np = np.zeros((n0,m1,m2), dtype=np.double)
    
    cdef double [:,:,::1] b = b_np
    
    cdef double weight
            
    cdef Py_ssize_t i, j, k, l, m

    with nogil:
        for m in prange(n):
            i = ind_i[m]
            l = ind_l[m]
            weight = comp[i,l]
            for j in range(m1):
                for k in range(m2):
                    b[i,j,k] += weight*a[l,j,k]
                        
    return b_np

def rebin1(double [:,:,::1] a, double [:,::1] comp):
    
    cdef Py_ssize_t m0 = a.shape[0]
    cdef Py_ssize_t m1 = a.shape[1]
    cdef Py_ssize_t m2 = a.shape[2]
    
    cdef Py_ssize_t n1 = comp.shape[0]
    
    ind_j_np, ind_l_np = np.ma.nonzero(comp)
    
    cdef Py_ssize_t [::] ind_j = ind_j_np
    cdef Py_ssize_t [::] ind_l = ind_l_np
      
    cdef Py_ssize_t n = ind_l.shape[0]
  
    b_np = np.zeros((m0,n1,m2), dtype=np.double)
    
    cdef double [:,:,::1] b = b_np
    
    cdef double weight
            
    cdef Py_ssize_t i, j, k, l, m
  
    with nogil:
        for m in prange(n):
            j = ind_j[m]
            l = ind_l[m]
            for i in range(m0):
                weight = comp[j,l]
                for k in range(m2):
                    b[i,j,k] += weight*a[i,l,k]
                    
    return b_np

def rebin2(double [:,:,::1] a, double [:,::1] comp):
    
    cdef Py_ssize_t m0 = a.shape[0]
    cdef Py_ssize_t m1 = a.shape[1]
    cdef Py_ssize_t m2 = a.shape[2]
    
    cdef Py_ssize_t n2 = comp.shape[0]
    
    ind_k_np, ind_l_np = np.ma.nonzero(comp)
    
    cdef Py_ssize_t [::] ind_k = ind_k_np
    cdef Py_ssize_t [::] ind_l = ind_l_np
      
    cdef Py_ssize_t n = ind_l.shape[0]
    
    b_np = np.zeros((m0,m1,n2), dtype=np.double)
    
    cdef double [:,:,::1] b = b_np
    
    cdef double weight
            
    cdef Py_ssize_t i, j, k, l, m
    
    with nogil:
        for m in prange(n):
            k = ind_k[m]
            l = ind_l[m]
            for i in range(m0):
                for j in range(m1):
                    weight = comp[k,l]
                    b[i,j,k] += weight*a[i,j,l]
                        
    return b_np
    
cpdef void extract(double complex [::1] B, 
                   double complex [::1] A, 
                   Py_ssize_t j, 
                   Py_ssize_t n) nogil:
    
    cdef Py_ssize_t m = B.shape[0]
        
    cdef Py_ssize_t i
    
    for i in prange(m):
        B[i] = A[j+n*i]
        
cpdef void insert(double complex [::1] A, 
                  double complex [::1] B, 
                  Py_ssize_t j, 
                  Py_ssize_t n) nogil:
    
    cdef Py_ssize_t m = B.shape[0]
    
    cdef Py_ssize_t i
    
    for i in prange(m):
        A[j+n*i] = B[i]
        
cpdef void take(double complex [::1] B, 
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
        
cpdef void give(double complex [::1] A, 
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
        
cpdef void copy(double complex [::1] B, double complex [::1] A) nogil:
    
    cdef Py_ssize_t m = A.shape[0]
        
    cdef Py_ssize_t i
    
    for i in prange(m):
        B[i] = A[i]
        
cpdef void get(double [::1] B, 
               double [::1] A, 
               Py_ssize_t j, 
               Py_ssize_t n) nogil:
    
    cdef Py_ssize_t m = B.shape[0]
        
    cdef Py_ssize_t i
    
    for i in prange(m):
        B[i] = A[j+n*i]
        
cpdef void put(double [::1] A, 
               double [::1] B, 
               Py_ssize_t j, 
               Py_ssize_t n) nogil:
    
    cdef Py_ssize_t m = B.shape[0]
    
    cdef Py_ssize_t i
    
    for i in prange(m):
        A[j+n*i] = B[i]
        
cpdef void detach(double [::1] B, 
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
        
cpdef void attach(double [::1] A, 
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
           
cdef void blur0(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t s
    
    if (nh-sigma > 0):
        s = sigma
    else:
        s = 0
    
    cdef double normal = 1./(2*s+1)
    
    cdef double value, f, g
    
    cdef Py_ssize_t u, j, k
    
    cdef Py_ssize_t i_j, i_jk
    
    cdef Py_ssize_t nkl = nk*nl
            
    for j in prange(nk):
        
        i_j = nl*j
        
        for k in range(nl):
            
            i_jk = k+i_j
            
            f = source[i_jk]
            g = source[i_jk+nkl*(nh-1)]
            
            value = (s+1)*f         
            for u in range(0,s,1):         
                value = value+source[i_jk+nkl*u]   
                
            for u in range(0,s+1,1):         
                value = value+source[i_jk+nkl*(u+s)]-f
                target[i_jk+nkl*u] = value*normal
                
            for u in range(s+1,nh-s,1):         
                value = value+source[i_jk+nkl*(u+s)]-source[i_jk+nkl*(u-s-1)]
                target[i_jk+nkl*u] = value*normal
                
            for u in range(nh-s,nh,1):         
                value = value+g-source[i_jk+nkl*(u-s-1)]
                target[i_jk+nkl*u] = value*normal
    
cdef void blur1(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t s
    
    if (nk-sigma > 0):
        s = sigma
    else:
        s = 0
    
    cdef double normal = 1./(2*s+1)
    
    cdef double value, f, g
    
    cdef Py_ssize_t i, v, k
    
    cdef Py_ssize_t i_i, i_ik
    
    cdef Py_ssize_t nkl = nk*nl
        
    for i in prange(nh):
        
        i_i = nkl*i
        
        for k in range(nl):
            
            i_ik = k+i_i
            
            f = source[i_ik]
            g = source[i_ik+nl*(nk-1)]
            
            value = (s+1)*f         
            for v in range(0,s,1):         
                value = value+source[i_ik+nl*v] 
                
            for v in range(0,s+1,1):         
                value = value+source[i_ik+nl*(v+s)]-f
                target[i_ik+nl*v] = value*normal
                
            for v in range(s+1,nk-s,1):         
                value = value+source[i_ik+nl*(v+s)]-source[i_ik+nl*(v-s-1)]
                target[i_ik+nl*v] = value*normal
                
            for v in range(nk-s,nk,1):         
                value = value+g-source[i_ik+nl*(v-s-1)]
                target[i_ik+nl*v] = value*normal
    
cdef void blur2(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t s
    
    if (nl-sigma > 0):
        s = sigma
    else:
        s = 0
    
    cdef double normal = 1./(2*s+1)
    
    cdef double value, f, g
    
    cdef Py_ssize_t i, j, w
    
    cdef Py_ssize_t i_i, i_ij
    
    cdef Py_ssize_t nkl = nk*nl
            
    for i in prange(nh):
        
        i_i = nkl*i
        
        for j in range(nk):
            
            i_ij = nl*j+i_i

            f = source[i_ij]
            g = source[i_ij+nl-1]
            
            value = (s+1)*f         
            for w in range(0,s,1):         
                value = value+source[i_ij+w] 
                
            for w in range(0,s+1,1):         
                value = value+source[i_ij+w+s]-f
                target[i_ij+w] = value*normal
                
            for w in range(s+1,nl-s,1):         
                value = value+source[i_ij+w+s]-source[i_ij+w-s-1]
                target[i_ij+w] = value*normal
                
            for w in range(nl-s,nl,1):         
                value = value+g-source[i_ij+w-s-1]
                target[i_ij+w] = value*normal
                
cdef void weight(double [::1] w, double [::1] u, double [::1] v) nogil:
    
    cdef Py_ssize_t n = w.shape[0]        
    
    cdef Py_ssize_t i
    
    for i in prange(n):
        w[i] = u[i]*v[i]
        
cpdef void gauss(double [::1] v, 
                 double [::1] u, 
                 long [::1] boxes, 
                 double [::1] a, 
                 double [::1] b, 
                 double [::1] c, 
                 double [::1] d, 
                 double [::1] e, 
                 double [::1] f, 
                 double [::1] g, 
                 double [::1] h, 
                 Py_ssize_t nh, 
                 Py_ssize_t nk, 
                 Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t n = boxes.shape[0]
    
    if (n == 3):
        
        blur0(a, u, boxes[0], nh, nk, nl)
        blur1(b, a, boxes[0], nh, nk, nl)
        blur2(c, b, boxes[0], nh, nk, nl)
        
        blur0(d, c, boxes[1], nh, nk, nl)
        blur1(e, d, boxes[1], nh, nk, nl)
        blur2(f, e, boxes[1], nh, nk, nl)
        
        blur0(g, f, boxes[2], nh, nk, nl)
        blur1(h, g, boxes[2], nh, nk, nl)
        blur2(v, h, boxes[2], nh, nk, nl)
    
    else:
        
        blur0(a, u, boxes[0], nh, nk, nl)
        blur1(b, a, boxes[1], nh, nk, nl)
        blur2(c, b, boxes[2], nh, nk, nl)
        
        blur0(d, c, boxes[3], nh, nk, nl)
        blur1(e, d, boxes[4], nh, nk, nl)
        blur2(f, e, boxes[5], nh, nk, nl)
        
        blur0(g, f, boxes[6], nh, nk, nl)
        blur1(h, g, boxes[7], nh, nk, nl)
        blur2(v, h, boxes[8], nh, nk, nl)
        
cpdef void blur(double [::1] v, 
                double [::1] u, 
                long [::1] boxes, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t n = boxes.shape[0]
    
    if (n == 3):
        
        blur0(v, u, boxes[0], nh, nk, nl)
        blur1(u, v, boxes[0], nh, nk, nl)
        blur2(v, u, boxes[0], nh, nk, nl)
        
        blur0(u, v, boxes[1], nh, nk, nl)
        blur1(v, u, boxes[1], nh, nk, nl)
        blur2(u, v, boxes[1], nh, nk, nl)
        
        blur0(v, u, boxes[2], nh, nk, nl)
        blur1(u, v, boxes[2], nh, nk, nl)
        blur2(v, u, boxes[2], nh, nk, nl)
    
    else:
        
        blur0(v, u, boxes[0], nh, nk, nl)
        blur1(u, v, boxes[1], nh, nk, nl)
        blur2(v, u, boxes[2], nh, nk, nl)
        
        blur0(u, v, boxes[3], nh, nk, nl)
        blur1(v, u, boxes[4], nh, nk, nl)
        blur2(u, v, boxes[5], nh, nk, nl)
        
        blur0(v, u, boxes[6], nh, nk, nl)
        blur1(u, v, boxes[7], nh, nk, nl)
        blur2(v, u, boxes[8], nh, nk, nl)
        
cpdef void filtering(double [::1] t, 
                     double [::1] s, 
                     double [::1] v_inv, 
                     long [::1] boxes, 
                     double [::1] a, 
                     double [::1] b, 
                     double [::1] c, 
                     double [::1] d, 
                     double [::1] e, 
                     double [::1] f, 
                     double [::1] g, 
                     double [::1] h, 
                     double [::1] i, 
                     Py_ssize_t nh, 
                     Py_ssize_t nk, 
                     Py_ssize_t nl) nogil:
    
    cdef Py_ssize_t n = boxes.shape[0]
    
    if (n == 3):
        
        blur0(a, s, boxes[0], nh, nk, nl)
        blur1(b, a, boxes[0], nh, nk, nl)
        blur2(c, b, boxes[0], nh, nk, nl)
        
        blur0(d, c, boxes[1], nh, nk, nl)
        blur1(e, d, boxes[1], nh, nk, nl)
        blur2(f, e, boxes[1], nh, nk, nl)
        
        blur0(g, f, boxes[2], nh, nk, nl)
        blur1(h, g, boxes[2], nh, nk, nl)
        blur2(i, h, boxes[2], nh, nk, nl)
    
    else:
        
        blur0(a, s, boxes[0], nh, nk, nl)
        blur1(b, a, boxes[1], nh, nk, nl)
        blur2(c, b, boxes[2], nh, nk, nl)
        
        blur0(d, c, boxes[3], nh, nk, nl)
        blur1(e, d, boxes[4], nh, nk, nl)
        blur2(f, e, boxes[5], nh, nk, nl)
        
        blur0(g, f, boxes[6], nh, nk, nl)
        blur1(h, g, boxes[7], nh, nk, nl)
        blur2(i, h, boxes[8], nh, nk, nl)
        
    weight(t, i, v_inv)
                
cpdef void intensity(double [::1] I, 
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
        
cpdef void unmask(double [::1] I_calc, 
                  double [::1] I, 
                  long [::1] i_unmask) nogil:
    
    cdef Py_ssize_t n_hkl = I_calc.shape[0]
    
    cdef Py_ssize_t i_hkl
        
    for i_hkl in prange(n_hkl):
                
        I_calc[i_hkl] = I[i_unmask[i_hkl]]
   
def length(ions, Py_ssize_t n_hkl):
    
    cdef Py_ssize_t n_atm = len(ions)
    
    b_np = np.zeros(n_hkl*n_atm, dtype=complex)
    
    cdef double complex [::1] b = b_np
    
    cdef double complex bc
    
    cdef Py_ssize_t i_hkl, i
            
    for i, ion in enumerate(ions):
    
        bc = tables.bc.get(ion)
        
        for i_hkl in range(n_hkl):
     
            b[i+n_atm*i_hkl] = bc
            
    return b_np

def form(ions, double [::1] Q, electron=False):
    
    cdef Py_ssize_t n_atm = len(ions)
    
    cdef Py_ssize_t n_hkl = Q.shape[0]
    
    f_np = np.zeros(n_hkl*n_atm, dtype=np.double)
    
    cdef double [::1] f = f_np
    
    cdef double s, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
        
    cdef Py_ssize_t i_hkl, i
            
    for i, ion in enumerate(ions):
        
        if (electron is False):
    
            a1, b1, a2, b2, a3, b3, a4, b4, c = tables.X.get(ion)
            
            for i_hkl in range(n_hkl):
                
                s = Q[i_hkl]/4/M_PI
         
                f[i+n_atm*i_hkl] = a1*exp(-b1*s**2)\
                                 + a2*exp(-b2*s**2)\
                                 + a3*exp(-b3*s**2)\
                                 + a4*exp(-b4*s**2)\
                                 + c
                                 
        else:
    
            a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = tables.E.get(ion)
            
            for i_hkl in range(n_hkl):
                
                s = Q[i_hkl]/4/M_PI
         
                f[i+n_atm*i_hkl] = a1*exp(-b1*s**2)\
                                 + a2*exp(-b2*s**2)\
                                 + a3*exp(-b3*s**2)\
                                 + a4*exp(-b4*s**2)\
                                 + a5*exp(-b5*s**2)
            
    return f_np
     
def phase(double [::1] Qx, 
          double [::1] Qy, 
          double [::1] Qz, 
          double [::1] rx, 
          double [::1] ry, 
          double [::1] rz):
    
    cdef Py_ssize_t n_hkl = Qx.shape[0]

    cdef Py_ssize_t n_uvw = rx.shape[0]

    factor_np = np.zeros(n_hkl*n_uvw, dtype=complex)
    
    cdef double complex [::1] factor = factor_np
       
    cdef Py_ssize_t i_hkl, i_uvw
        
    for i_hkl in range(n_hkl):  
            for i_uvw in range(n_uvw):  
                
                factor[i_uvw+n_uvw*i_hkl] = iexp(Qx[i_hkl]*rx[i_uvw]+\
                                                 Qy[i_hkl]*ry[i_uvw]+\
                                                 Qz[i_hkl]*rz[i_uvw])

    return factor_np
                                                                                
cpdef (double, double) goodness(double [::1] calc, 
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
                        
cpdef void products_molecule(double [:,::1] V, 
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

cpdef void magnetic(double [::1] I,
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
     
cpdef void occupational(double [::1] I,
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
     
cpdef void displacive(double [::1] I,
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
        
cpdef void spin(double complex [::1] Sx_k_cand,
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
        
cpdef void composition(double complex [::1] A_k_cand,
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
                
cpdef void composition_molecule(double complex [::1] A_k_cand,
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

cpdef void expansion(double complex [::1] U_k_cand,
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
                    
cpdef void expansion_molecule(double complex [::1] U_k_cand,
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
                                                 
cpdef void relaxation(double complex [::1] A_k_cand,
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
                    
cpdef void relaxation_molecule(double complex [::1] A_k_cand,
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
                                                 
cpdef void extension(double complex [::1] U_k_cand,
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
                    
cpdef void extension_molecule(double complex [::1] U_k_cand,
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
                                                 
cpdef void moment(double complex [::1] Fx_cand,
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
           
cpdef void occupancy(double complex [::1] F_cand,
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
        
cpdef void displacement(double complex [::1] F_cand,
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
                          
cpdef void displacement_molecule(double complex [::1] F_cand,
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
                               
cpdef void nonmagnetic(double complex [::1] F_cand,
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
                          
cpdef void nonmagnetic_molecule(double complex [::1] F_cand,
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
