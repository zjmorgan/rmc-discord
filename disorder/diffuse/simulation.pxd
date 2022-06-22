#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

cimport cython

cdef void initialize_random(Py_ssize_t nu,
                            Py_ssize_t nv,
                            Py_ssize_t nw,
                            Py_ssize_t n_atm,
                            Py_ssize_t n_temp)

cdef bint iszero(double a) nogil

cdef Py_ssize_t sqrt_babylonian(Py_ssize_t n) nogil

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

cdef double energy_moment(double [:,:,:,::1] p,
                          double [:,::1] Q,
                          double vx,
                          double vy,
                          double vz,
                          double ux,
                          double uy,
                          double uz,
                          Py_ssize_t i,
                          Py_ssize_t t) nogil

cdef void update_moment(double [:,:,:,::1] p,
                        double [:,::1] Q,
                        double vx,
                        double vy,
                        double vz,
                        double ux,
                        double uy,
                        double uz,
                        Py_ssize_t i,
                        Py_ssize_t t) nogil

cdef double energy_moment_cluster(double [:,:,:,::1] p,
                                  double [:,::1] Q,
                                  double [::1] clust_vx,
                                  double [::1] clust_vy,
                                  double [::1] clust_vz,
                                  double [::1] clust_ux,
                                  double [::1] clust_uy,
                                  double [::1] clust_uz,
                                  Py_ssize_t [::1] clust_ind,
                                  Py_ssize_t n_c,
                                  Py_ssize_t t) nogil

cdef void update_moment_cluster(double [:,:,:,::1] p,
                                double [:,::1] Q,
                                double [::1] clust_vx,
                                double [::1] clust_vy,
                                double [::1] clust_vz,
                                double [::1] clust_ux,
                                double [::1] clust_uy,
                                double [::1] clust_uz,
                                Py_ssize_t [::1] clust_ind,
                                Py_ssize_t n_c,
                                Py_ssize_t t) nogil