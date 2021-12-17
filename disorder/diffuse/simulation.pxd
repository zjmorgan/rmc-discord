#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

cimport cython

cdef void initialize_random(Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm,
                            Py_ssize_t n_temp)

cdef bint iszero(double a) nogil

cdef double random_uniform() nogil

cdef double alpha(double E, double beta) nogil

cdef (Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t,
      Py_ssize_t) random_original(Py_ssize_t nu,
                                  Py_ssize_t nv,
                                  Py_ssize_t nw,
                                  Py_ssize_t n_atm) nogil

cdef (double, double, double) random_vector_candidate() nogil

cdef (double, double, double) random_vector_length_candidate() nogil

cdef (double, double, double) ising_vector_candidate(double ux,
                                                     double uy,
                                                     double uz) nogil

cdef double random_gaussian() nogil

cdef (double, double, double) random_gaussian_3d() nogil

cdef (double, double, double) gaussian_vector_candidate(double ux,
                                                        double uy,
                                                        double uz,
                                                        double sigma) nogil

cdef (double, double, double) interpolated_vector_candidate(double ux,
                                                            double uy,
                                                            double uz,
                                                            double sigma) nogil

cdef void replica_exchange(double [::1] H,
                           double [::1] beta,
                           double [::1] sigma) nogil