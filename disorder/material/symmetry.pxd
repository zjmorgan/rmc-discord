#cython: language_level=3

cimport cython
                
cpdef void friedel(signed short [:,::1] pair,
                   signed short [:,:,::1] coordinate)

cpdef (double, double, double) transform(double x, 
                                         double y, 
                                         double z, 
                                         Py_ssize_t sym, 
                                         Py_ssize_t op) nogil