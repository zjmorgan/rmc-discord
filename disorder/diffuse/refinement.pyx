#cython: boundscheck=False, wraparound=True, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np
    
from disorder.diffuse cimport original
from disorder.diffuse cimport candidate
from disorder.diffuse cimport scattering

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp, log

cdef double random() nogil:
    
    return float(rand())/RAND_MAX

cpdef void magnetic(double [::1] Sx,
                    double [::1] Sy,
                    double [::1] Sz,
                    double [::1] Qx_norm,
                    double [::1] Qy_norm,
                    double [::1] Qz_norm,
                    double complex [::1] Sx_k,
                    double complex [::1] Sy_k,
                    double complex [::1] Sz_k,
                    double complex [::1] Sx_k_orig,
                    double complex [::1] Sy_k_orig,
                    double complex [::1] Sz_k_orig,
                    double complex [::1] Sx_k_cand,
                    double complex [::1] Sy_k_cand,
                    double complex [::1] Sz_k_cand,
                    double complex [::1] Fx,
                    double complex [::1] Fy,
                    double complex [::1] Fz,
                    double complex [::1] Fx_orig,
                    double complex [::1] Fy_orig,
                    double complex [::1] Fz_orig,
                    double complex [::1] Fx_cand,
                    double complex [::1] Fy_cand,
                    double complex [::1] Fz_cand,
                    double complex [::1] prod_x,
                    double complex [::1] prod_y,
                    double complex [::1] prod_z,
                    double complex [::1] prod_x_orig,     
                    double complex [::1] prod_y_orig,     
                    double complex [::1] prod_z_orig,  
                    double complex [::1] prod_x_cand,
                    double complex [::1] prod_y_cand,
                    double complex [::1] prod_z_cand,   
                    double complex [::1] space_factor,
                    double complex [::1] factors,
                    double [::1] moment,
                    double [::1] I_calc,
                    double [::1] I_exp,
                    double [::1] inv_sigma_sq,
                    double [::1] I_raw, 
                    double [::1] I_flat,
                    double [::1] I_ref, 
                    double [::1] v_inv, 
                    double [::1] a_filt,
                    double [::1] b_filt,
                    double [::1] c_filt,
                    double [::1] d_filt,
                    double [::1] e_filt,
                    double [::1] f_filt,
                    double [::1] g_filt,
                    double [::1] h_filt,
                    double [::1] i_filt,
                    long [::1] boxes,
                    long [::1] i_dft,
                    long [::1] inverses,
                    long [::1] i_mask,
                    long [::1] i_unmask,
                    list acc_moves,
                    list acc_temps,
                    list rej_moves,
                    list rej_temps,
                    list chi_sq,
                    list energy,
                    list temperature,
                    list scale,
                    double constant,
                    double delta,
                    bint fixed,
                    double [::1] T, 
                    Py_ssize_t nh,
                    Py_ssize_t nk, 
                    Py_ssize_t nl,
                    Py_ssize_t nu,
                    Py_ssize_t nv,
                    Py_ssize_t nw,
                    Py_ssize_t n_atm,
                    Py_ssize_t n,
                    Py_ssize_t N):
    
    cdef double temp, inv_temp
    
    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor
    
    cdef double Sx_orig, Sy_orig, Sz_orig, Sx_cand, Sy_cand, Sz_cand, mu
    
    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)
        
    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef Py_ssize_t i, j, s
    
    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]

    with nogil:
    
        for s in range(N):
                                       
            temp = exp(log(temp)-constant)  
            
            inv_temp = 1/temp
                        
            Sx_orig, Sy_orig, Sz_orig, i = original.vector(Sx, Sy, Sz)
                
            j = i % n_atm
            
            mu = moment[j]
    
            Sx_cand, Sy_cand, Sz_cand = candidate.vector(Sx_orig, 
                                                         Sy_orig, 
                                                         Sz_orig, 
                                                         mu,
                                                         fixed)
                                    
            scattering.copy(Fx_orig, Fx)
            scattering.copy(Fy_orig, Fy)
            scattering.copy(Fz_orig, Fz)
            
            scattering.extract(prod_x_orig, prod_x, j, n_atm)
            scattering.extract(prod_y_orig, prod_y, j, n_atm)
            scattering.extract(prod_z_orig, prod_z, j, n_atm)
            
            scattering.extract(Sx_k_orig, Sx_k, j, n_atm)
            scattering.extract(Sy_k_orig, Sy_k, j, n_atm)
            scattering.extract(Sz_k_orig, Sz_k, j, n_atm)
                        
            Sx[i], Sy[i], Sz[i] = Sx_cand, Sy_cand, Sz_cand 
    
            scattering.spin(Sx_k_cand,
                            Sy_k_cand,
                            Sz_k_cand,
                            Sx_cand, 
                            Sy_cand, 
                            Sz_cand, 
                            Sx_k_orig,
                            Sy_k_orig,
                            Sz_k_orig,
                            Sx_orig,
                            Sy_orig,
                            Sz_orig,
                            space_factor,
                            i,
                            nu,
                            nv,
                            nw, 
                            n_atm)
            
            scattering.insert(Sx_k, Sx_k_cand, j, n_atm)
            scattering.insert(Sy_k, Sy_k_cand, j, n_atm)
            scattering.insert(Sz_k, Sz_k_cand, j, n_atm)
            
            scattering.moment(Fx_cand,
                              Fy_cand,
                              Fz_cand,
                              prod_x_cand,
                              prod_y_cand,
                              prod_z_cand,
                              Sx_k_cand,
                              Sy_k_cand,
                              Sz_k_cand,
                              Fx_orig,
                              Fy_orig,
                              Fz_orig,
                              prod_x_orig,
                              prod_y_orig,
                              prod_z_orig,
                              factors,
                              j,
                              i_dft, 
                              n_atm)
    
            scattering.insert(prod_x, prod_x_cand, j, n_atm)
            scattering.insert(prod_y, prod_y_cand, j, n_atm)
            scattering.insert(prod_z, prod_z_cand, j, n_atm)
    
            scattering.copy(Fx, Fx_cand)
            scattering.copy(Fy, Fy_cand)
            scattering.copy(Fz, Fz_cand)
            
            scattering.magnetic(I_calc,
                                Qx_norm, 
                                Qy_norm, 
                                Qz_norm, 
                                Fx_cand, 
                                Fy_cand, 
                                Fz_cand, 
                                n)
            
            scattering.intensity(I_raw, I_calc, inverses, i_mask)
                    
            scattering.filtering(I_flat,
                                 I_raw, 
                                 v_inv, 
                                 boxes,
                                 a_filt,
                                 b_filt,
                                 c_filt,
                                 d_filt,
                                 e_filt,
                                 f_filt,
                                 g_filt,
                                 h_filt,
                                 i_filt, 
                                 nh, 
                                 nk, 
                                 nl)
                    
            scattering.unmask(I_ref, I_flat, i_unmask)
                        
            chi_sq_cand, scale_factor = scattering.goodness(I_ref, 
                                                            I_exp, 
                                                            inv_sigma_sq)
            
            delta_chi_sq = chi_sq_cand-chi_sq_orig
            
            if (delta_chi_sq > 0):
                if (random() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand
                    
                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    Sx[i], Sy[i], Sz[i] = Sx_orig, Sy_orig, Sz_orig 
                    
                    scattering.insert(Sx_k, Sx_k_orig, j, n_atm)
                    scattering.insert(Sy_k, Sy_k_orig, j, n_atm)
                    scattering.insert(Sz_k, Sz_k_orig, j, n_atm)
                            
                    scattering.insert(prod_x, prod_x_orig, j, n_atm)
                    scattering.insert(prod_y, prod_y_orig, j, n_atm)
                    scattering.insert(prod_z, prod_z_orig, j, n_atm)
    
                    scattering.copy(Fx, Fx_orig)
                    scattering.copy(Fy, Fy_orig)
                    scattering.copy(Fz, Fz_orig)
            
                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand
                
                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp
                
            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor
        
    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)

cpdef void occupational(double [::1] A_r,
                        double complex [::1] A_k,
                        double complex [::1] A_k_orig,
                        double complex [::1] A_k_cand,
                        double complex [::1] F,
                        double complex [::1] F_orig,
                        double complex [::1] F_cand,
                        double complex [::1] prod,
                        double complex [::1] prod_orig,
                        double complex [::1] prod_cand,     
                        double complex [::1] space_factor,
                        double complex [::1] factors,
                        double [::1] occupancy,
                        double [::1] I_calc,
                        double [::1] I_exp,
                        double [::1] inv_sigma_sq,
                        double [::1] I_raw, 
                        double [::1] I_flat,
                        double [::1] I_ref, 
                        double [::1] v_inv, 
                        double [::1] a_filt,
                        double [::1] b_filt,
                        double [::1] c_filt,
                        double [::1] d_filt,
                        double [::1] e_filt,
                        double [::1] f_filt,
                        double [::1] g_filt,
                        double [::1] h_filt,
                        double [::1] i_filt,
                        long [::1] boxes,
                        long [::1] i_dft,
                        long [::1] inverses,
                        long [::1] i_mask,
                        long [::1] i_unmask,
                        list acc_moves,
                        list acc_temps,
                        list rej_moves,
                        list rej_temps,
                        list chi_sq,
                        list energy,
                        list temperature,
                        list scale,
                        double constant,
                        bint fixed,
                        Py_ssize_t nh,
                        Py_ssize_t nk, 
                        Py_ssize_t nl,
                        Py_ssize_t nu,
                        Py_ssize_t nv,
                        Py_ssize_t nw,
                        Py_ssize_t n_atm,
                        Py_ssize_t n,
                        Py_ssize_t N):
        
    cdef double temp, inv_temp
    
    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor
    
    cdef double A_r_orig, A_r_cand, occ
    
    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)
        
    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef Py_ssize_t i, j, s
    
    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]

    with nogil:
    
        for s in range(N):
                                       
            temp = exp(log(temp)-constant)  
            
            inv_temp = 1/temp
                        
            A_r_orig, i = original.scalar(A_r)
                
            j = i % n_atm
            
            occ = occupancy[j]
                    
            A_r_cand = candidate.scalar(A_r_orig, occ, fixed)
                            
            scattering.copy(F_orig, F)
            
            scattering.extract(prod_orig, prod, j, n_atm)
            
            scattering.extract(A_k_orig, A_k, j, n_atm)
                
            A_r[i] = A_r_cand
              
            scattering.composition(A_k_cand,
                                   A_r_cand, 
                                   A_k_orig,
                                   A_r_orig,
                                   space_factor,
                                   i,
                                   nu,
                                   nv,
                                   nw, 
                                   n_atm)  
            
            scattering.insert(A_k, A_k_cand, j, n_atm)
            
            scattering.occupancy(F_cand, 
                                 prod_cand,
                                 A_k_cand,
                                 F_orig,
                                 prod_orig,
                                 factors,
                                 j,
                                 i_dft, 
                                 n_atm)
            
            scattering.insert(prod, prod_cand, j, n_atm)
            
            scattering.copy(F, F_cand)
            
            scattering.occupational(I_calc, F_cand, n)
            
            scattering.intensity(I_raw, I_calc, inverses, i_mask)
                    
            scattering.filtering(I_flat,
                                 I_raw, 
                                 v_inv, 
                                 boxes,
                                 a_filt,
                                 b_filt,
                                 c_filt,
                                 d_filt,
                                 e_filt,
                                 f_filt,
                                 g_filt,
                                 h_filt,
                                 i_filt, 
                                 nh, 
                                 nk, 
                                 nl)
                    
            scattering.unmask(I_ref, I_flat, i_unmask)
                        
            chi_sq_cand, scale_factor = scattering.goodness(I_ref, 
                                                            I_exp, 
                                                            inv_sigma_sq)
            
            delta_chi_sq = chi_sq_cand-chi_sq_orig
            
            if (delta_chi_sq > 0):
                if (random() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand
                    
                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    A_r[i] = A_r_orig
                    
                    scattering.insert(A_k, A_k_orig, j, n_atm)
                            
                    scattering.insert(prod, prod_orig, j, n_atm)
            
                    scattering.copy(F, F_orig)
            
                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand
                
                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp
                    
            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor
        
    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)
    
cpdef void displacive(double [::1] Ux,
                      double [::1] Uy,
                      double [::1] Uz,
                      double [::1] U_r,
                      double [::1] U_r_orig,
                      double [::1] U_r_cand,
                      double complex [::1] U_k,
                      double complex [::1] U_k_orig,
                      double complex [::1] U_k_cand,
                      double complex [::1] V_k,
                      double complex [::1] V_k_nuc,
                      double complex [::1] V_k_orig,
                      double complex [::1] V_k_nuc_orig,
                      double complex [::1] V_k_cand,
                      double complex [::1] V_k_nuc_cand,
                      double complex [::1] F,
                      double complex [::1] F_nuc,
                      double complex [::1] F_orig,
                      double complex [::1] F_nuc_orig,
                      double complex [::1] F_cand,
                      double complex [::1] F_nuc_cand,
                      double complex [::1] prod,
                      double complex [::1] prod_nuc,
                      double complex [::1] prod_orig,     
                      double complex [::1] prod_nuc_orig,    
                      double complex [::1] prod_cand,
                      double complex [::1] prod_nuc_cand,
                      double complex [::1] space_factor,
                      double complex [::1] factors,
                      double complex [::1] coeffs,
                      double [::1] Q_k,
                      double [::1] displacement,
                      double [::1] I_calc,
                      double [::1] I_exp,
                      double [::1] inv_sigma_sq,
                      double [::1] I_raw, 
                      double [::1] I_flat,
                      double [::1] I_ref, 
                      double [::1] v_inv, 
                      double [::1] a_filt,
                      double [::1] b_filt,
                      double [::1] c_filt,
                      double [::1] d_filt,
                      double [::1] e_filt,
                      double [::1] f_filt,
                      double [::1] g_filt,
                      double [::1] h_filt,
                      double [::1] i_filt,
                      long [::1] bragg,
                      long [::1] even,
                      long [::1] boxes,
                      long [::1] i_dft,
                      long [::1] inverses,
                      long [::1] i_mask,
                      long [::1] i_unmask,
                      list acc_moves,
                      list acc_temps,
                      list rej_moves,
                      list rej_temps,
                      list chi_sq,
                      list energy,
                      list temperature,
                      list scale,
                      double constant,
                      double delta,
                      bint fixed,
                      double [::1] T, 
                      Py_ssize_t p,
                      Py_ssize_t nh,
                      Py_ssize_t nk, 
                      Py_ssize_t nl,
                      Py_ssize_t nu,
                      Py_ssize_t nv,
                      Py_ssize_t nw,
                      Py_ssize_t n_atm,
                      Py_ssize_t n,
                      Py_ssize_t N):

    cdef double temp, inv_temp
    
    cdef double delta_chi_sq, chi_sq_cand, chi_sq_orig, scale_factor
    
    cdef double Ux_orig, Uy_orig, Uz_orig, Ux_cand, Uy_cand, Uz_cand, disp
    
    acc_moves_np = np.full(N, np.nan)
    acc_temps_np = np.full(N, np.nan)
    rej_moves_np = np.full(N, np.nan)
    rej_temps_np = np.full(N, np.nan)
    chi_sq_np = np.zeros(N, dtype=np.double)
    energy_np = np.zeros(N, dtype=np.double)
    temperature_np = np.zeros(N, dtype=np.double)
    scale_np = np.zeros(N, dtype=np.double)
        
    cdef double [::1] acc_moves_arr = acc_moves_np
    cdef double [::1] acc_temps_arr = acc_temps_np
    cdef double [::1] rej_moves_arr = rej_moves_np
    cdef double [::1] rej_temps_arr = rej_temps_np
    cdef double [::1] chi_sq_arr = chi_sq_np
    cdef double [::1] energy_arr = energy_np
    cdef double [::1] temperature_arr = temperature_np
    cdef double [::1] scale_arr = scale_np

    cdef Py_ssize_t i, j, s
    
    chi_sq_orig = chi_sq[-1]
    temp = temperature[-1]
        
    with nogil:
    
        for s in range(N):
                                       
            temp = exp(log(temp)-constant)  
            
            inv_temp = 1/temp
                        
            Ux_orig, Uy_orig, Uz_orig, i = original.vector(Ux, Uy, Uz)
                
            j = i % n_atm
            
            disp = displacement[j]

            Ux_cand, Uy_cand, Uz_cand = candidate.vector(Ux_orig, 
                                                         Uy_orig, 
                                                         Uz_orig, 
                                                         disp,
                                                         fixed)

            scattering.copy(F_orig, F)
            scattering.copy(F_nuc_orig, F_nuc)
            
            scattering.extract(prod_orig, prod, j, n_atm)
            scattering.extract(prod_nuc_orig, prod_nuc, j, n_atm)
            
            scattering.extract(V_k_orig, V_k, j, n_atm)
            scattering.extract(V_k_nuc_orig, V_k_nuc, j, n_atm)
            
            Ux[i], Uy[i], Uz[i] = Ux_cand, Uy_cand, Uz_cand
            
            scattering.get(U_r_orig, U_r, i, n)
           
            scattering.extract(U_k_orig, U_k, j, n_atm)
            
            scattering.products(U_r_cand, Ux_cand, Uy_cand, Uz_cand, p)
                    
            scattering.put(U_r, U_r_cand, i, n)
            
            scattering.expansion(U_k_cand,
                                 U_r_cand, 
                                 U_k_orig,
                                 U_r_orig,
                                 space_factor,
                                 i,
                                 nu,
                                 nv,
                                 nw, 
                                 n_atm)   
                            
            scattering.insert(U_k, U_k_cand, j, n_atm)
            
            scattering.displacement(F_cand,
                                    F_nuc_cand,
                                    prod_cand,
                                    prod_nuc_cand,
                                    V_k_cand,
                                    V_k_nuc_cand,
                                    U_k_cand,
                                    F_orig,
                                    F_nuc_orig,
                                    prod_orig,
                                    prod_nuc_orig,
                                    V_k_orig,
                                    V_k_nuc_orig,
                                    U_k_orig,
                                    Q_k,
                                    factors,
                                    coeffs,
                                    even,
                                    bragg,
                                    i_dft, 
                                    p,
                                    j,
                                    n_atm)
            
            scattering.insert(V_k, V_k_cand, j, n_atm)
            scattering.insert(V_k_nuc, V_k_nuc_cand, j, n_atm)
            
            scattering.insert(prod, prod_cand, j, n_atm)
            scattering.insert(prod_nuc, prod_nuc_cand, j, n_atm)
    
            scattering.copy(F, F_cand)
            scattering.copy(F_nuc, F_nuc_cand)
            
            scattering.displacive(I_calc, F_cand, F_nuc_cand, bragg, n)
            
            scattering.intensity(I_raw, I_calc, inverses, i_mask)
                    
            scattering.filtering(I_flat,
                                 I_raw, 
                                 v_inv, 
                                 boxes,
                                 a_filt,
                                 b_filt,
                                 c_filt,
                                 d_filt,
                                 e_filt,
                                 f_filt,
                                 g_filt,
                                 h_filt,
                                 i_filt, 
                                 nh, 
                                 nk, 
                                 nl)
                    
            scattering.unmask(I_ref, I_flat, i_unmask)
            
            chi_sq_cand, scale_factor = scattering.goodness(I_ref, 
                                                            I_exp, 
                                                            inv_sigma_sq)
            
            delta_chi_sq = chi_sq_cand-chi_sq_orig
            
            if (delta_chi_sq > 0):
                if (random() < exp(-inv_temp*delta_chi_sq)):
                    chi_sq_orig = chi_sq_cand
                    
                    acc_moves_arr[s] = chi_sq_orig
                    acc_temps_arr[s] = temp
                else:
                    Ux[i], Uy[i], Uz[i] = Ux_orig, Uy_orig, Uz_orig 
    
                    scattering.put(U_r, U_r_orig, i, n)
                    
                    scattering.insert(U_k, U_k_orig, j, n_atm)
                    
                    scattering.insert(V_k, V_k_orig, j, n_atm)
                    scattering.insert(V_k_nuc, V_k_nuc_orig, j, n_atm)
                            
                    scattering.insert(prod, prod_orig, j, n_atm)
                    scattering.insert(prod_nuc, prod_nuc_orig, j, n_atm)
    
                    scattering.copy(F, F_orig)
                    scattering.copy(F_nuc, F_nuc_orig)  
            
                    rej_moves_arr[s] = chi_sq_orig
                    rej_temps_arr[s] = temp
            else:
                chi_sq_orig = chi_sq_cand
                
                acc_moves_arr[s] = chi_sq_orig
                acc_temps_arr[s] = temp
                
            chi_sq_arr[s] = chi_sq_orig
            energy_arr[s] = delta_chi_sq
            temperature_arr[s] = temp
            scale_arr[s] = scale_factor
        
    acc_moves.extend(acc_moves_arr)
    acc_temps.extend(acc_temps_arr)
    rej_moves.extend(rej_moves_arr)
    rej_temps.extend(rej_temps_arr)
    chi_sq.extend(chi_sq_arr)
    energy.extend(energy_arr)
    temperature.extend(temperature_arr)
    scale.extend(scale_arr)