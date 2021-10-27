#cython: language_level=3

cimport cython
           
cdef void blur0(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil
    
cdef void blur1(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil
    
cdef void blur2(double [::1] target, 
                double [::1] source, 
                Py_ssize_t sigma, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil
                
cdef void weight(double [::1] w, double [::1] u, double [::1] v) nogil
    
        
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
                 Py_ssize_t nl) nogil
        
cpdef void blur(double [::1] v, 
                double [::1] u, 
                long [::1] boxes, 
                Py_ssize_t nh, 
                Py_ssize_t nk, 
                Py_ssize_t nl) nogil
        
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
                     Py_ssize_t nl) nogil
    
cdef void sort(double [:,::1] data, 
               long long [:,::1] order, 
               int n, 
               Py_ssize_t thread_id) nogil

cdef void copysort(double [:,::1] data, 
                   double [:,::1] copy_data, 
                   long long [:,::1] order, 
                   long long [:,::1] copy_order, 
                   long long [:,::1] temp_order, 
                   int n, 
                   Py_ssize_t thread_id) nogil