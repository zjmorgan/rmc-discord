#cython: language_level=3

cimport cython
    
cpdef double scalar(double A, double value, bint fixed) nogil

cpdef (double, double, double) vector(double A, 
                                      double B, 
                                      double C, 
                                      double value,
                                      bint fixed) nogil