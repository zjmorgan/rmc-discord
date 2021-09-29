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

cdef void sort(double [:,::1] data, 
               long long [:,::1] order, 
               int n, 
               Py_ssize_t thread_id) nogil:

    cdef double temp
    
    cdef Py_ssize_t temp_ind, i, j
    
    i = 1
    while (i < n):
        temp = data[thread_id,i]
        temp_ind = order[thread_id,i]
        j = i-1
        while (j >= 0 and data[thread_id,j] > temp):
            data[thread_id,j+1] = data[thread_id,j]
            order[thread_id,j+1] = order[thread_id,j]
            j = j-1
        data[thread_id,j+1] = temp
        order[thread_id,j+1] = temp_ind
        i = i+1
        
def padding(data, rank):
    n0, n1, n2 = data.shape
    padded = np.zeros((n0+2*rank,n1+2*rank,n2+2*rank))
    padded[rank:-rank,rank:-rank,rank:-rank] = data
    return padded

def median(double [:,:,::1] a, Py_ssize_t size):
    
    cdef Py_ssize_t thread_id, num_threads = openmp.omp_get_max_threads()
        
    cdef Py_ssize_t n0 = a.shape[0]
    cdef Py_ssize_t n1 = a.shape[1]
    cdef Py_ssize_t n2 = a.shape[2]
      
    b_np = np.zeros((n0,n1,n2), dtype=np.double)
    
    cdef double [:,:,::1] b = b_np
    
    cdef Py_ssize_t rank = size // 2
    
    cdef Py_ssize_t window_size = size**3
    
    cdef Py_ssize_t size_sq = size**2
    
    cdef Py_ssize_t med = window_size // 2
    
    a_np = np.copy(a, order='c')
    
    cdef double [:,:,::1] a_pad = padding(a_np, rank)
    
    window_np = np.zeros((num_threads,window_size), dtype=np.double)

    temp_window_np = window_np.copy()

    cdef double [:,::1] window = window_np
    cdef double [:,::1] temp_window = temp_window_np
    
    order = np.arange(window_size)
    
    dim = (size,size,size)
    
    order_i_np, order_j_np, order_k_np = np.unravel_index(order, dim)
    
    order_i_np = np.tile(order_i_np, num_threads)
    order_j_np = np.tile(order_j_np, num_threads)
    order_k_np = np.tile(order_k_np, num_threads)
    
    order_i_np = order_i_np.reshape(num_threads,window_size)
    order_j_np = order_j_np.reshape(num_threads,window_size)
    order_k_np = order_k_np.reshape(num_threads,window_size)
    
    temp_order_i_np = order_i_np.copy()
    temp_order_j_np = order_j_np.copy()
    temp_order_k_np = order_k_np.copy()
    
    cdef long long [:,::1] order_i = order_i_np
    cdef long long [:,::1] order_j = order_j_np
    cdef long long [:,::1] order_k = order_k_np
    cdef long long [:,::1] temp_order_i = temp_order_i_np
    cdef long long [:,::1] temp_order_j = temp_order_j_np
    cdef long long [:,::1] temp_order_k = temp_order_k_np
    
    cdef Py_ssize_t i, j, k, l, m, n, p, q, r
    
    cdef Py_ssize_t sliding_size = window_size-size_sq
    
    cdef Py_ssize_t j_l, j_lm, j_lmn
    
    cdef Py_ssize_t i_l, j_m, k_n
    
    cdef Py_ssize_t i_l_max, j_m_max, k_n_max

    i_l_max, j_m_max, k_n_max = n0-1, n1-1, n2-1
    
    cdef Py_ssize_t m_max = 2*rank
    cdef Py_ssize_t n_max = 2*rank
    
    cdef Py_ssize_t wind_indx

    with nogil:
        for i in prange(n0):
            thread_id = openmp.omp_get_thread_num()
            for j in range(n1):
                for k in range(n2):
    
                    if (k == 0):
                        
                        for l in range(size):
                            i_l = i+l
                            j_l = size_sq*l
                            for m in range(size):
                                j_m = j+m
                                j_lm = j_l+size*m
                                for n in range(size):
                                    k_n = k+n
                                    j_lmn = j_lm+n
                                    window[thread_id,j_lmn] = a_pad[i_l,j_m,k_n]  
                                    order_k[thread_id,j_lmn] = n
                                        
                        sort(window, order_k, window_size, thread_id)
                    
                        b[i,j,k] = window[thread_id,med]
                        
                        # for p in range(window_size):
                        #     wcache[thread_id,p] = window[thread_id,p]
                        #     wcache_order[thread_id,p] = window_order[thread_id,p]
                    
                    # elif (k == 0):
                        
                    #     p = 0
                    #     while (p < sliding_size):
                    #         if (wcache_order[thread_id,p] // size % size == 0):
                    #             q = p
                    #             r = p+1
                    #             while (r < window_size):
                    #                 if (wcache_order[thread_id,r] // size \
                    #                                                % size != 0):
                    #                     wcache[thread_id,q] = wcache[thread_id,r]
                    #                     wcache_order[thread_id,q] = \
                    #                     wcache_order[thread_id,r]
                    #                     q = q+1
                    #                 r = r+1
                    #         wcache_order[thread_id,p] = \
                    #         wcache_order[thread_id,p]-size     
                                        
                    #         window[thread_id,p] = wcache[thread_id,p] 
                    #         window_order[thread_id,p] = wcache_order[thread_id,p] 
                    #         p = p+1
                            
                    #     m = m_max
                    #     j_m = j+m
                    #     p = 0
                    #     for l in range(size):
                    #         i_l = i+l
                    #         j_l = size_sq*l
                    #         j_lm = j_l+size*m
                    #         for n in range(size):
                    #             k_n = k+n
                    #             j_lmn = j_lm+n
                    #             wind_indx = sliding_size+p
                    #             window[thread_id,wind_indx] = a_pad[i_l,j_m,k_n]  
                    #             window_order[thread_id,wind_indx] = j_lmn
                    #             p = p+1
    
                    #     sort(window, window_order, window_size, thread_id)
                        
                    #     b[i,j,k] = window[thread_id,med]
                        
                    #     for p in range(window_size):
                    #         wcache[thread_id,p] = window[thread_id,p]
                    #         wcache_order[thread_id,p] = window_order[thread_id,p]     
                            
                    else:
                        
                        p = 0
                        while (p < sliding_size):
                            if (order_k[thread_id,p] == 0):
                                q = p
                                r = p+1
                                while (r < window_size):
                                    if (order_k[thread_id,r] != 0):
                                        window[thread_id,q] = \
                                        window[thread_id,r]
                                        order_k[thread_id,q] = \
                                        order_k[thread_id,r]
                                        q = q+1
                                    r = r+1
                            order_k[thread_id,p] = order_k[thread_id,p]-1       
                            p = p+1
                            
                        p = 0
                        n = n_max
                        k_n = k+n                       
                        for l in range(size):
                            i_l = i+l
                            for m in range(size):
                                j_m = j+m
                                wind_indx = sliding_size+p
                                window[thread_id,wind_indx] = a_pad[i_l,j_m,k_n]  
                                order_k[thread_id,wind_indx] = n 
                                p = p+1
                                        
                        sort(window, order_k, window_size, thread_id)
                    
                        b[i,j,k] = window[thread_id,med]
        
    return b_np