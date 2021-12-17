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
        for m in range(n):
            i = ind_i[m]
            l = ind_l[m]
            weight = comp[i,l]
            for j in range(m1):
                for k in prange(m2):
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
        for m in range(n):
            j = ind_j[m]
            l = ind_l[m]
            for i in range(m0):
                weight = comp[j,l]
                for k in prange(m2):
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
        for m in range(n):
            k = ind_k[m]
            l = ind_l[m]
            for i in range(m0):
                for j in prange(m1):
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

def boxblur(sigma, n):

    sigma = np.asarray(sigma)

    l = np.floor(np.sqrt(12*sigma**2/n+1))

    if (np.size(sigma) == 1):
        if (l % 2 == 0):
            l -= 1

    else:
        l[np.mod(l,2) == 0] -= 1

    u = l+2

    m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)

    if (np.size(sigma) == 1):

        sizes = np.zeros(n, dtype=int)

        for i in range(n):

            if (i < m):
                sizes[i] = l
            else:
                sizes[i] = u

    else:

        sizes = np.zeros((n, np.size(sigma)), dtype=int)

        for i in range(n):
            for j in range(np.size(sigma)):

                if (i < m[j]):
                    sizes[i,j] = l[j]
                else:
                    sizes[i,j] = u[j]

    return ((np.array(sizes)-1)/2).astype(int).flatten()

def gaussian(mask, sigma):

    v = np.ones(mask.shape)
    v[mask] = 0

    v = v.flatten()

    a = np.zeros(mask.size)
    b = np.zeros(mask.size)
    c = np.zeros(mask.size)
    d = np.zeros(mask.size)
    e = np.zeros(mask.size)
    f = np.zeros(mask.size)
    g = np.zeros(mask.size)
    h = np.zeros(mask.size)

    w = np.zeros(mask.size)

    boxes = boxblur(sigma, 3)

    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]

    if (np.isclose(sigma, 0).all()):

        return v

    else:

        gauss(w, v, boxes, a, b, c, d, e, f, g, h, nh, nk, nl)

        veil = np.isclose(w,0)

        if (np.sum(veil) > 0):

            x = np.zeros(mask.size)

            gauss(x, v, boxes, a, b, c, d, e, f, g, h, nh, nk, nl)

            w[veil] = x[veil].copy()

        veil = np.isclose(w,0)

        w_inv = np.zeros(mask.size)
        w_inv[~veil] = 1/w[~veil]

        return w_inv

def boxfilter(I, mask, sigma, v_inv):

    a = np.zeros(mask.size)
    b = np.zeros(mask.size)
    c = np.zeros(mask.size)
    d = np.zeros(mask.size)
    e = np.zeros(mask.size)
    f = np.zeros(mask.size)
    g = np.zeros(mask.size)
    h = np.zeros(mask.size)

    i = np.zeros(mask.size)

    boxes = boxblur(sigma, 3)

    nh, nk, nl = mask.shape[0], mask.shape[1], mask.shape[2]

    if np.isclose(sigma, 0).all():

        return  I

    else:

        u = I.copy()
        u[mask] = 0

        gauss(i, u.flatten(), boxes, a, b, c, d, e, f, g, h, nh, nk, nl)

        return (v_inv*i).reshape(nh,nk,nl)

def blurring(I, sigma):

    i = np.zeros(I.size)

    boxes = boxblur(sigma, 3)

    nh, nk, nl = I.shape[0], I.shape[1], I.shape[2]

    if np.isclose(sigma, 0).all():

        return I

    else:

        blur(i, I.flatten(), boxes, nh, nk, nl)

        return i.reshape(nh,nk,nl)

cdef void sort(double [:,::1] data,
               long long [:,::1] order,
               int n,
               Py_ssize_t thread_id) nogil:

    cdef double temp

    cdef Py_ssize_t temp_ind, i, j, k

    i = 1
    while (i < n):
        temp = data[thread_id,i]
        temp_ind = order[thread_id,i]
        j = i-1
        while (j >= 0 and data[thread_id,j] > temp):
            k = j+1
            data[thread_id,k] = data[thread_id,j]
            order[thread_id,k] = order[thread_id,j]
            j = j-1
        k = j+1
        data[thread_id,k] = temp
        order[thread_id,k] = temp_ind
        i = i+1

cdef void copysort(double [:,::1] data,
                   double [:,::1] copy_data,
                   long long [:,::1] order,
                   long long [:,::1] copy_order,
                   long long [:,::1] temp_order,
                   int n,
                   Py_ssize_t thread_id) nogil:

    cdef double temp

    cdef Py_ssize_t temp_ind, temp_temp_ind, i, j, k

    i = 1
    while (i < n):
        temp = data[thread_id,i]
        temp_ind = order[thread_id,i]
        temp_temp_ind = temp_order[thread_id,i]
        j = i-1
        while (j >= 0 and data[thread_id,j] > temp):
            k = j+1
            data[thread_id,k] = data[thread_id,j]
            order[thread_id,k] = order[thread_id,j]
            copy_data[thread_id,k] = copy_data[thread_id,j]
            copy_order[thread_id,k] = copy_order[thread_id,j]
            temp_order[thread_id,k] = temp_order[thread_id,j]
            j = j-1
        k = j+1
        data[thread_id,k] = temp
        order[thread_id,k] = temp_ind
        copy_data[thread_id,k] = temp
        copy_order[thread_id,k] = temp_ind
        temp_order[thread_id,k] = temp_temp_ind
        i = i+1

def padding(data, rank):
    n0, n1, n2 = data.shape
    for _ in range(rank):
        data = np.concatenate((data[:1,:,:],data,data[-1:,:,:]), axis=0)
        data = np.concatenate((data[:,:1,:],data,data[:,-1:,:]), axis=1)
        data = np.concatenate((data[:,:,:1],data,data[:,:,-1:]), axis=2)
    return data

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

    cdef double [:,:,::1] A = padding(a_np, rank)

    window_np = np.zeros((num_threads,window_size), dtype=np.double)

    temp_wind_np = window_np.copy()

    cdef double [:,::1] window = window_np
    cdef double [:,::1] temp_wind = temp_wind_np

    order = np.arange(window_size)

    dim = (size,size,size)

    order_i_np, order_j_np, order_k_np = np.unravel_index(order, dim)

    order_j_np = np.tile(order_j_np, num_threads)
    order_k_np = np.tile(order_k_np, num_threads)

    order_j_np = order_j_np.reshape(num_threads,window_size)
    order_k_np = order_k_np.reshape(num_threads,window_size)

    temp_ord_j_np = order_j_np.copy()
    temp_ord_k_np = order_k_np.copy()

    cdef long long [:,::1] order_k = order_k_np

    cdef long long [:,::1] temp_ord_j = temp_ord_j_np
    cdef long long [:,::1] temp_ord_k = temp_ord_k_np

    cdef Py_ssize_t i, j, k, l, m, n, p, q, r

    cdef Py_ssize_t sliding_size = window_size-size_sq

    cdef Py_ssize_t j_l, j_lm, j_lmn

    cdef Py_ssize_t i_l, j_m, k_n

    cdef Py_ssize_t m_max = 2*rank
    cdef Py_ssize_t n_max = 2*rank

    cdef Py_ssize_t wind_indx

    with nogil:
        for i in prange(n0):
            thread_id = openmp.omp_get_thread_num()
            for j in range(n1):
                for k in range(n2):

                    if (j == 0 and k == 0):

                        for l in range(size):
                            i_l = i+l
                            j_l = size_sq*l
                            for m in range(size):
                                j_m = j+m
                                j_lm = j_l+size*m
                                for n in range(size):
                                    k_n = k+n
                                    j_lmn = j_lm+n
                                    window[thread_id,j_lmn] = A[i_l,j_m,k_n]
                                    temp_wind[thread_id,j_lmn] = A[i_l,j_m,k_n]
                                    order_k[thread_id,j_lmn] = n
                                    temp_ord_k[thread_id,j_lmn] = n
                                    temp_ord_j[thread_id,j_lmn] = m

                        copysort(window,
                                 temp_wind,
                                 order_k,
                                 temp_ord_k,
                                 temp_ord_j,
                                 window_size,
                                 thread_id)

                    elif (k == 0):

                        p = 0
                        q = 0
                        while (p < window_size):
                            if (temp_ord_j[thread_id,p] != 0):
                                window[thread_id,q] = temp_wind[thread_id,p]
                                temp_wind[thread_id,q] = temp_wind[thread_id,p]
                                order_k[thread_id,q] = temp_ord_k[thread_id,p]
                                temp_ord_k[thread_id,q] = \
                                temp_ord_k[thread_id,p]
                                temp_ord_j[thread_id,q] = \
                                temp_ord_j[thread_id,p]-1
                                q = q+1
                            p = p+1

                        p = 0
                        m = m_max
                        j_m = j+m
                        for l in range(size):
                            i_l = i+l
                            for n in range(size):
                                k_n = k+n
                                wind_indx = sliding_size+p
                                window[thread_id,wind_indx] = A[i_l,j_m,k_n]
                                temp_wind[thread_id,wind_indx] = A[i_l,j_m,k_n]
                                order_k[thread_id,wind_indx] = n
                                temp_ord_k[thread_id,wind_indx] = n
                                temp_ord_j[thread_id,wind_indx] = m
                                p = p+1

                        copysort(window,
                                 temp_wind,
                                 order_k,
                                 temp_ord_k,
                                 temp_ord_j,
                                 window_size,
                                 thread_id)

                    else:

                        p = 0
                        q = 0
                        while (p < window_size):
                            if (order_k[thread_id,p] != 0):
                                window[thread_id,q] = window[thread_id,p]
                                order_k[thread_id,q] = order_k[thread_id,p]-1
                                q = q+1
                            p = p+1

                        p = 0
                        n = n_max
                        k_n = k+n
                        for l in range(size):
                            i_l = i+l
                            for m in range(size):
                                j_m = j+m
                                wind_indx = sliding_size+p
                                window[thread_id,wind_indx] = A[i_l,j_m,k_n]
                                order_k[thread_id,wind_indx] = n
                                p = p+1

                        sort(window, order_k, window_size, thread_id)

                    b[i,j,k] = window[thread_id,med]

    return b_np