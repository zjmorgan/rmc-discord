#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython
cimport openmp

from libc.math cimport M_PI, cos, sin, exp, sqrt, acos, fabs

import os, sys

def rebin0(double [:,:,::1] a, double [:,::1] comp):
    
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
    
# cdef void sort(double [::1] array2sort, int n) nogil:

#     cdef double temp
    
#     cdef Py_ssize_t i, j
    
#     for i in range(0, n-1):
#         for j in range(0, n-1-i):
#             if (array2sort[j] > array2sort[j+1]):
#                 temp = array2sort[j]
#                 array2sort[j] = array2sort[j+1]
#                 array2sort[j+1] = temp

cdef void sort(double [::1] data, long [::1] order, int n) nogil:

    cdef double temp
    
    cdef Py_ssize_t temp_ind, i, j
    
    i = 1
    while (i < n):
        temp = data[i]
        temp_ind = i
        j = i-1
        while (j >= 0 and data[j] > temp):
            data[j+1] = data[j]
            order[j+1] = order[j]
            j = j-1
        data[j+1] = temp
        order[j+1] = temp_ind
        i = i+1
        
cdef void argsort(long [::1] data, int n) nogil:

    cdef Py_ssize_t temp
    
    cdef Py_ssize_t i, j
    
    i = 1
    while (i < n):
        temp = data[i]
        j = i-1
        while (j >= 0 and data[j] > temp):
            data[j+1] = data[j]
            j = j-1
        data[j+1] = temp
        i = i+1
        
cdef void copy(long [::1] data, long [::1] order, int n) nogil:
    
    cdef Py_ssize_t i
    
    for i in range(n):
        
        data[i] = order[i]

def median(double [:,:,::1] a, Py_ssize_t size):
    
    cdef Py_ssize_t n0 = a.shape[0]
    cdef Py_ssize_t n1 = a.shape[1]
    cdef Py_ssize_t n2 = a.shape[2]
  
    b_np = np.zeros((n0,n1,n2), dtype=np.double)
    
    cdef double [:,:,::1] b = b_np
    
    cdef Py_ssize_t rank = size // 2
    
    cdef Py_ssize_t window_size = size**3
    
    cdef Py_ssize_t size_sq = size**2
    
    cdef Py_ssize_t med = window_size // 2
    
    window_np = np.zeros(window_size, dtype=np.double)

    cdef double [::1] window = window_np
    
    window_order_np = np.arange(window_size, dtype=int)
    argument_order_np = np.arange(window_size, dtype=int)
    
    cdef long [::1] window_order = window_order_np
    cdef long [::1] argument_order = argument_order_np

    cdef Py_ssize_t i, j, k, l, m, n
    
    cdef Py_ssize_t j_l, j_lm, j_lmn
    
    cdef Py_ssize_t i_l, j_m, k_n

    with nogil:
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                                            
                    for l in range(2*rank+1):
                        i_l = i+(l-rank)
                        if (i_l < 0):
                            i_l = 0
                        elif (i_l > n0-1):
                            i_l = n0-1
                        j_l = size_sq*l
                        for m in range(2*rank+1):
                            j_m = j+(m-rank)
                            if (j_m < 0):
                                j_m = 0
                            elif (j_m > n1-1):
                                j_m = n1-1
                            j_lm = j_l+size*m
                            for n in range(2*rank+1):
                                k_n = k+(n-rank)
                                if (k_n < 0):
                                    k_n = 0
                                elif (k_n > n2-1):
                                    k_n = n2-1
                                j_lmn = j_lm+n
                    
                                window[j_lmn] = a[i_l,j_m,k_n]
                                    
                    sort(window, window_order, window_size)
                    copy(argument_order, window_order, window_size)
                    argsort(argument_order, window_size)
                    
                    b[i,j,k] = window[med]
            
    return b_np