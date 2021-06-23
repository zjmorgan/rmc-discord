#cython: language_level=3

cimport cython
    
cpdef void extract(double complex [::1] B, 
                   double complex [::1] A, 
                   Py_ssize_t j, 
                   Py_ssize_t n) nogil
        
cpdef void insert(double complex [::1] A, 
                  double complex [::1] B, 
                  Py_ssize_t j, 
                  Py_ssize_t n) nogil
        
cpdef void take(double complex [::1] B, 
                double complex [::1] A, 
                long [::1] indices, 
                Py_ssize_t n) nogil
        
cpdef void give(double complex [::1] A, 
                double complex [::1] B, 
                long [::1] indices, 
                Py_ssize_t n) nogil
        
cpdef void copy(double complex [::1] B, double complex [::1] A) nogil
        
cpdef void get(double [::1] B, 
               double [::1] A, 
               Py_ssize_t j, 
               Py_ssize_t n) nogil
        
cpdef void put(double [::1] A, 
               double [::1] B, 
               Py_ssize_t j, 
               Py_ssize_t n) nogil
        
cpdef void detach(double [::1] B, 
                  double [::1] A, 
                  long [::1] indices, 
                  Py_ssize_t n) nogil
        
cpdef void attach(double [::1] A, 
                  double [::1] B, 
                  long [::1] indices, 
                  Py_ssize_t n) nogil
           
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
                
cpdef void intensity(double [::1] I, 
                     double [::1] I_calc, 
                     long [::1] inverses,
                     long [::1] i_mask) nogil
        
cpdef void unmask(double [::1] I_calc, 
                  double [::1] I, 
                  long [::1] i_unmask) nogil
                                                                                
cpdef (double, double) goodness(double [::1] calc, 
                                double [::1] exp, 
                                double [::1] inv_error_sq) nogil       

cpdef void products(double [::1] V, 
                    double Vx, 
                    double Vy, 
                    double Vz, 
                    Py_ssize_t p) nogil

cpdef void products_molecule(double [:,::1] V, 
                             double [::1] Vx, 
                             double [::1] Vy, 
                             double [::1] Vz, 
                             Py_ssize_t p) nogil 

cpdef void magnetic(double [::1] I,
                    double [::1] Qx_norm, 
                    double [::1] Qy_norm, 
                    double [::1] Qz_norm, 
                    double complex [::1] Fx, 
                    double complex [::1] Fy, 
                    double complex [::1] Fz, 
                    Py_ssize_t n_xyz) nogil
     
cpdef void occupational(double [::1] I,
                        double complex [::1] F, 
                        Py_ssize_t n_xyz) nogil
     
cpdef void displacive(double [::1] I,
                      double complex [::1] F, 
                      double complex [::1] F_nuc,
                      long [::1] bragg,
                      Py_ssize_t n_xyz) nogil
        
cpdef void spin(double complex [::1] Sx_k_cand,
                double complex [::1] Sy_k_cand,
                double complex [::1] Sz_k_cand,
                double Sx_cand, 
                double Sy_cand, 
                double Sz_cand, 
                double complex [::1] Sx_k_orig,
                double complex [::1] Sy_k_orig,
                double complex [::1] Sz_k_orig,
                double Sx_orig,
                double Sy_orig,
                double Sz_orig,
                double complex [::1] space_factor,
                Py_ssize_t i,
                Py_ssize_t nu, 
                Py_ssize_t nv, 
                Py_ssize_t nw, 
                Py_ssize_t n_atm) nogil
        
cpdef void composition(double complex [::1] A_k_cand,
                       double A_cand, 
                       double complex [::1] A_k_orig,
                       double A_orig,
                       double complex [::1] space_factor,
                       Py_ssize_t i,
                       Py_ssize_t nu, 
                       Py_ssize_t nv, 
                       Py_ssize_t nw, 
                       Py_ssize_t n_atm) nogil
                
cpdef void composition_molecule(double complex [::1] A_k_cand,
                                double [::1] A_cand, 
                                double complex [::1] A_k_orig,
                                double [::1] A_orig,
                                double complex [::1] space_factor,
                                long [::1] i_atm,
                                Py_ssize_t nu, 
                                Py_ssize_t nv, 
                                Py_ssize_t nw, 
                                Py_ssize_t n_atm) nogil

cpdef void expansion(double complex [::1] U_k_cand,
                     double [::1] U_cand, 
                     double complex [::1] U_k_orig,
                     double [::1] U_orig,
                     double complex [::1] space_factor,
                     Py_ssize_t i,
                     Py_ssize_t nu, 
                     Py_ssize_t nv, 
                     Py_ssize_t nw, 
                     Py_ssize_t n_atm) nogil
                    
cpdef void expansion_molecule(double complex [::1] U_k_cand,
                              double [::1] U_cand, 
                              double complex [::1] U_k_orig,
                              double [::1] U_orig,
                              double complex [::1] space_factor,
                              long [::1] i_atm,
                              Py_ssize_t nu, 
                              Py_ssize_t nv, 
                              Py_ssize_t nw, 
                              Py_ssize_t n_atm) nogil
                                                 
cpdef void relaxation(double complex [::1] A_k_cand,
                      double A_cand, 
                      double complex [::1] A_k_orig,
                      double A_orig,
                      double [::1] U,
                      double complex [::1] space_factor,
                      Py_ssize_t i,
                      Py_ssize_t nu, 
                      Py_ssize_t nv, 
                      Py_ssize_t nw, 
                      Py_ssize_t n_atm) nogil
                    
cpdef void relaxation_molecule(double complex [::1] A_k_cand,
                               double [::1] A_cand, 
                               double complex [::1] A_k_orig,
                               double [::1] A_orig,
                               double [::1] U,
                               double complex [::1] space_factor,
                               long [::1] i_atm,
                               Py_ssize_t nu, 
                               Py_ssize_t nv, 
                               Py_ssize_t nw, 
                               Py_ssize_t n_atm) nogil
                                                 
cpdef void extension(double complex [::1] U_k_cand,
                     double complex [::1] A_k_cand,
                     double [::1] U_cand, 
                     double complex [::1] U_k_orig,
                     double complex [::1] A_k_orig,
                     double [::1] U_orig,
                     double A,
                     double complex [::1] space_factor,
                     Py_ssize_t i,
                     Py_ssize_t nu, 
                     Py_ssize_t nv, 
                     Py_ssize_t nw, 
                     Py_ssize_t n_atm) nogil
                    
cpdef void extension_molecule(double complex [::1] U_k_cand,
                              double complex [::1] A_k_cand,
                              double [::1] U_cand, 
                              double complex [::1] U_k_orig,
                              double complex [::1] A_k_orig,
                              double [::1] U_orig,
                              double [::1] A_orig,
                              double complex [::1] space_factor,
                              long [::1] i_atm,
                              Py_ssize_t nu, 
                              Py_ssize_t nv, 
                              Py_ssize_t nw, 
                              Py_ssize_t n_atm) nogil
                                                 
cpdef void moment(double complex [::1] Fx_cand,
                  double complex [::1] Fy_cand,
                  double complex [::1] Fz_cand,
                  double complex [::1] prod_x_cand,
                  double complex [::1] prod_y_cand,
                  double complex [::1] prod_z_cand,
                  double complex [::1] Sx_k_cand,
                  double complex [::1] Sy_k_cand,
                  double complex [::1] Sz_k_cand,
                  double complex [::1] Fx_orig,
                  double complex [::1] Fy_orig,
                  double complex [::1] Fz_orig,
                  double complex [::1] prod_x_orig,
                  double complex [::1] prod_y_orig,
                  double complex [::1] prod_z_orig,
                  double complex [::1] factors,
                  Py_ssize_t j,
                  long [::1] i_dft, 
                  Py_ssize_t n_atm) nogil
           
cpdef void occupancy(double complex [::1] F_cand,
                     double complex [::1] prod_cand,
                     double complex [::1] A_k_cand,
                     double complex [::1] F_orig,
                     double complex [::1] prod_orig,
                     double complex [::1] factors,
                     Py_ssize_t j,
                     long [::1] i_dft, 
                     Py_ssize_t n_atm) nogil
        
cpdef void displacement(double complex [::1] F_cand,
                        double complex [::1] F_nuc_cand,
                        double complex [::1] prod_cand,
                        double complex [::1] prod_nuc_cand,
                        double complex [::1] V_k_cand,
                        double complex [::1] V_k_nuc_cand,
                        double complex [::1] U_k_cand,
                        double complex [::1] F_orig,
                        double complex [::1] F_nuc_orig,
                        double complex [::1] prod_orig,
                        double complex [::1] prod_nuc_orig,
                        double complex [::1] V_k_orig,
                        double complex [::1] V_k_nuc_orig,
                        double complex [::1] U_k_orig,
                        double [::1] Q_k,
                        double complex [::1] factors,
                        double complex [::1] coeffs,
                        long [::1] even,
                        long [::1] bragg,
                        long [::1] i_dft, 
                        Py_ssize_t p,
                        Py_ssize_t j,
                        Py_ssize_t n_atm) nogil
                          
cpdef void displacement_molecule(double complex [::1] F_cand,
                                 double complex [::1] F_nuc_cand,
                                 double complex [::1] prod_cand,
                                 double complex [::1] prod_nuc_cand,
                                 double complex [::1] V_k_cand,
                                 double complex [::1] V_k_nuc_cand,
                                 double complex [::1] U_k_cand,
                                 double complex [::1] F_orig,
                                 double complex [::1] F_nuc_orig,
                                 double complex [::1] prod_orig,
                                 double complex [::1] prod_nuc_orig,
                                 double complex [::1] V_k_orig,
                                 double complex [::1] V_k_nuc_orig,
                                 double complex [::1] U_k_orig,
                                 double [::1] Q_k,
                                 double complex [::1] factors,
                                 double complex [::1] coeffs,
                                 long [::1] even,
                                 long [::1] bragg,
                                 long [::1] i_dft, 
                                 Py_ssize_t p,
                                 long [::1] j_atm,
                                 Py_ssize_t n_atm) nogil
                               
cpdef void nonmagnetic(double complex [::1] F_cand,
                       double complex [::1] F_nuc_cand,
                       double complex [::1] prod_cand,
                       double complex [::1] prod_nuc_cand,
                       double complex [::1] V_k_cand,
                       double complex [::1] V_k_nuc_cand,
                       double complex [::1] U_k_cand,
                       double complex [::1] A_k_cand,
                       double complex [::1] F_orig,
                       double complex [::1] F_nuc_orig,
                       double complex [::1] prod_orig,
                       double complex [::1] prod_nuc_orig,
                       double complex [::1] V_k_orig,
                       double complex [::1] V_k_nuc_orig,
                       double complex [::1] U_k_orig,
                       double complex [::1] A_k_orig,
                       double [::1] Q_k,
                       double complex [::1] factors,
                       double complex [::1] coeffs,
                       long [::1] even,
                       long [::1] bragg,
                       long [::1] i_dft, 
                       Py_ssize_t p,
                       Py_ssize_t j,
                       Py_ssize_t n_atm) nogil
                          
cpdef void nonmagnetic_molecule(double complex [::1] F_cand,
                                double complex [::1] F_nuc_cand,
                                double complex [::1] prod_cand,
                                double complex [::1] prod_nuc_cand,
                                double complex [::1] V_k_cand,
                                double complex [::1] V_k_nuc_cand,
                                double complex [::1] U_k_cand,
                                double complex [::1] A_k_cand,
                                double complex [::1] F_orig,
                                double complex [::1] F_nuc_orig,
                                double complex [::1] prod_orig,
                                double complex [::1] prod_nuc_orig,
                                double complex [::1] V_k_orig,
                                double complex [::1] V_k_nuc_orig,
                                double complex [::1] U_k_orig,
                                double complex [::1] A_k_orig,
                                double [::1] Q_k,
                                double complex [::1] factors,
                                double complex [::1] coeffs,
                                long [::1] even,
                                long [::1] bragg,
                                long [::1] i_dft, 
                                Py_ssize_t p,
                                long [::1] j_atm,
                                Py_ssize_t n_atm) nogil