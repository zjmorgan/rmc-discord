#cython: language_level=3

cimport cython
    
cpdef double composition(double A, double value, bint fixed) nogil

cpdef (double, double, double) moment(double A, 
                                      double B, 
                                      double C, 
                                      double value,
                                      bint fixed,
                                      bint rotate) nogil

cpdef (double, double, double) displacement(double A, 
                                            double B, 
                                            double C, 
                                            double D, 
                                            double E, 
                                            double F, 
                                            bint fixed,
                                            bint isotropic) nogil