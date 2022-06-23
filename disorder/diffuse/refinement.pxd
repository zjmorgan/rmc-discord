#cython: boundscheck=False, wraparound=True, nonecheck=False, cdivision=True
#cython: language_level=3

cimport cython

cdef double random_uniform_nonzero() nogil

cdef double random_uniform() nogil

cdef double random_gaussian() nogil

cdef (double, double, double) random_gaussian_3d() nogil

cdef bint iszero(double a) nogil

cdef double complex cexp(double complex z) nogil

cdef double complex iexp(double y) nogil