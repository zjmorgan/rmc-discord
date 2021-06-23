#cython: language_level=3

cimport cython

cdef double f(double x, double y) nogil
    
cpdef double scalar(double A, double value, bint fixed) nogil

cpdef (double, double, double) vector(double A, 
                                      double B, 
                                      double C, 
                                      double delta, 
                                      double value,
                                      bint fixed,
                                      double [::1] T) nogil
    
cpdef void scalars(double [::1] B, 
                   double [::1] A, 
                   double value, 
                   bint fixed) nogil              
                   
cpdef void vectors(double [::1] D, 
                   double [::1] E, 
                   double [::1] F, 
                   double [::1] A, 
                   double [::1] B, 
                   double [::1] C, 
                   double delta, 
                   double value, 
                   bint fixed) nogil