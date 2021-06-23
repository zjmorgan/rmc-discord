#cython: language_level=3

cimport cython
    
cpdef (double, Py_ssize_t) scalar(double [::1] A) nogil
        
cpdef (double, double, double, Py_ssize_t) vector(double [::1] A,
                                                  double [::1] B,
                                                  double [::1] C) nogil
    
cpdef Py_ssize_t scalars(double [::1] B, 
                         long [::1] i,
                         double [::1] A, 
                         long [:,::1] structure) nogil
        
cpdef Py_ssize_t vectors(double [::1] D, 
                         double [::1] E, 
                         double [::1] F,
                         long [::1] i,
                         double [::1] A, 
                         double [::1] B, 
                         double [::1] C, 
                         long [:,::1] structure) nogil