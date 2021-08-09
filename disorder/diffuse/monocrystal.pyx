#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from disorder.material cimport symmetry
from disorder.material import tables

from libc.math cimport M_PI, cos, sin, exp, sqrt, fabs, fmod, remainder

cdef (double, double, double) transform(double x, 
                                        double y, 
                                        double z, 
                                        Py_ssize_t sym, 
                                        Py_ssize_t op) nogil:
      
    return symmetry.transform(x, y, z, sym, op)

cdef double round2even(double x) nogil:
    x -= remainder(x, 1.0)
    return x

cdef double complex iexp(double y) nogil:
    return cos(y)+1j*sin(y)

cdef bint iszero(double a) nogil:
    
    cdef double atol = 1e-04
    
    return fabs(a) <= atol

cdef bint nuclear(double h, double k, double l, int centering) nogil:
    
    cdef bint cond = 0
    
    cdef int H, K, L
    
    H, K, L = int(round2even(h)), int(round2even(k)), int(round2even(l))
    #H, K, L = int(h), int(k), int(l)
    
    if (centering == 0):
        cond = 0
    elif (centering == 1):
        cond = (H % 1 == 0) & (K % 1 == 0) & (L % 1 == 0)
    elif (centering == 2):
        cond = ((H+K+L) % 2 == 0)
    elif (centering == 3):
        cond = ((H+K) % 2 == 0) \
             & ((K+L) % 2 == 0) \
             & ((L+H) % 2 == 0)
    elif (centering == 4):
        cond = ((-H+K+L) % 3 == 0)
    elif (centering == 5):
        cond = ((H+K) % 2 == 0)
    elif (centering == 6):
        cond = ((K+L) % 2 == 0)
    elif (centering == 7):
        cond = ((L+H) % 2 == 0)

    return cond

def magnetic(double [::1] Sx, 
             double [::1] Sy,
             double [::1] Sz,
             double [::1] ux,
             double [::1] uy,
             double [::1] uz,
             ions,
             h_range,
             k_range,
             l_range,
             long [::1] indices,
             symop,
             double [:,:] T,
             double [:,:] B,
             double [:,:] R,
             double [:,:,:] domains,
             double [:] variants,
             Py_ssize_t nh,
             Py_ssize_t nk,
             Py_ssize_t nl,
             Py_ssize_t nu,
             Py_ssize_t nv,
             Py_ssize_t nw,
             Py_ssize_t Nu,
             Py_ssize_t Nv,
             Py_ssize_t Nw,
             double [::1] g):
    
    cdef Py_ssize_t n_atm = len(ions)
        
    cdef Py_ssize_t n_hkl = indices.shape[0]
    
    cdef Py_ssize_t sym = symop[0]
    cdef Py_ssize_t n_ops = symop[1]
    
    cdef Py_ssize_t n_vars = variants.shape[0]

    cdef Py_ssize_t n_uvw = nu*nv*nw
    cdef Py_ssize_t n = n_uvw*n_atm
    
    cdef double factor
    
    if (n == 0):
        factor = 0
    else:
        factor = 1.0/n
    
    cdef double x_, y_, z_   
    cdef double h_, k_, l_   

    cdef double x, y, z
    cdef double h, k, l
   
    cdef double Qh, Qk, Ql
    cdef double Qx, Qy, Qz
    
    cdef double Qx_norm, Qy_norm, Qz_norm, Q
    
    cdef double complex phase_factor, factors
    
    cdef double [::1] K2 = 2/np.copy(g, order='C')-1
    
    cdef double form_factor, j0, j2, s_, s_sq
    
    cdef double [::1] A0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] B0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] C0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] D0 = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] A2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] B2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] C2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] D2 = np.zeros(n_atm, dtype=float)

    cdef double [::1] a0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b0 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c0 = np.zeros(n_atm, dtype=float)
    
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c2 = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
        
    Sx_np = np.copy(Sx, order='C')
    Sy_np = np.copy(Sy, order='C')
    Sz_np = np.copy(Sz, order='C')
    
    Sx_np = Sx_np.reshape(nu,nv,nw,n_atm)
    Sy_np = Sy_np.reshape(nu,nv,nw,n_atm)
    Sz_np = Sz_np.reshape(nu,nv,nw,n_atm)
    
    cdef double h_min_, k_min_, l_min_
    cdef double h_max_, k_max_, l_max_
    
    cdef double h_step_, k_step_, l_step_
        
    h_min_, k_min_, l_min_ = h_range[0], k_range[0], l_range[0]
    
    if (nh == 1):
        h_max_ = h_range[0]
        h_step_ = 0
    else:
        h_max_ = h_range[1]
        h_step_ = (h_max_-h_min_)/(nh-1)
        
    if (nk == 1):
        k_max_ = k_range[0]
        k_step_ = 0
    else:
        k_max_ = k_range[1]
        k_step_ = (k_max_-k_min_)/(nk-1)
        
    if (nl == 1):
        l_max_ = l_range[0]
        l_step_ = 0
    else:
        l_max_ = l_range[1]
        l_step_ = (l_max_-l_min_)/(nl-1)
        
    cdef Py_ssize_t Fu = Nu // nu
    cdef Py_ssize_t Fv = Nv // nv
    cdef Py_ssize_t Fw = Nw // nw
    
    cdef double f_H, f_K, f_L
    
    cdef double f_Nu = float(Nu)
    cdef double f_Nv = float(Nv)
    cdef double f_Nw = float(Nw)
    
    Sx_k_np = idft(Sx_np, Nu, Nv, Nw).flatten()*n_uvw
    Sy_k_np = idft(Sy_np, Nu, Nv, Nw).flatten()*n_uvw
    Sz_k_np = idft(Sz_np, Nu, Nv, Nw).flatten()*n_uvw
   
    cdef double complex [::1] Sx_k = Sx_k_np
    cdef double complex [::1] Sy_k = Sy_k_np
    cdef double complex [::1] Sz_k = Sz_k_np
    
    cdef Py_ssize_t F_uvw = Fu*Fv*Fw
    cdef Py_ssize_t N_uvw = Nu*Nv*Nw 
    
    cdef double complex prod_x, prod_y, prod_z
    cdef double complex Fx, Fy, Fz
    
    cdef double complex Fx_perp, Fy_perp, Fz_perp
    cdef double complex Q_norm_dot_F

    cdef double Fx_perp_real, Fy_perp_real, Fz_perp_real
    cdef double Fx_perp_imag, Fy_perp_imag, Fz_perp_imag
    
    cdef Py_ssize_t H, K, L, iH, iK, iL
    cdef Py_ssize_t i_dft, i_ind, j_dft
    
    cdef Py_ssize_t op, var
    cdef Py_ssize_t i, j
    cdef Py_ssize_t u, v, w
    
    cdef double M_TAU = 2*np.pi
    cdef double complex M_I = 1j
    
    cdef double inv_M_SP = 1/(4*np.pi)
        
    for j in range(n_atm):
        
        ion = ions[j]
        
        if (tables.j0.get(ion) is None):
            A0[j], a0[j], \
            B0[j], b0[j], \
            C0[j], c0[j], \
            D0[j] = 0, 0, 0, 0, 0, 0, 0
            A2[j], a2[j], \
            B2[j], b2[j], \
            C2[j], c2[j], \
            D2[j] = 0, 0, 0, 0, 0, 0, 0
        else:
            A0[j], a0[j], \
            B0[j], b0[j], \
            C0[j], c0[j], \
            D0[j] = tables.j0.get(ion)
            A2[j], a2[j], \
            B2[j], b2[j], \
            C2[j], c2[j], \
            D2[j] = tables.j2.get(ion)
                            
    for i in prange(n_hkl, nogil=True):
             
        i_ind = indices[i]
        
        w = i_ind % nl 
        v = i_ind // nl % nk
        u = i_ind // nl // nk % nh
        
        h_ = h_min_+h_step_*u
        k_ = k_min_+k_step_*v
        l_ = l_min_+l_step_*w
        
        x = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
        y = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
        z = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
        
        for var in range(n_vars):
            
            x_ = domains[var,0,0]*x+domains[var,0,1]*y+domains[var,0,2]*z
            y_ = domains[var,1,0]*x+domains[var,1,1]*y+domains[var,1,2]*z
            z_ = domains[var,2,0]*x+domains[var,2,1]*y+domains[var,2,2]*z
        
            for op in range(n_ops):
                                        
                h, k, l = transform(x_, y_, z_, sym, op)
                            
                f_H, f_K, f_L = h*Nu, k*Nv, l*Nw
                
                H = int(round2even(f_H))
                K = int(round2even(f_K))
                L = int(round2even(f_L))
                
                iH = (H % Nu+Nu) % Nu
                iK = (K % Nv+Nv) % Nv
                iL = (L % Nw+Nw) % Nw
                
                if (iH < Fu and not iszero(fmod(f_H, f_Nu)) and Nu-iH > Fu):
                    iH = iH+Fu
                if (iK < Fv and not iszero(fmod(f_K, f_Nv)) and Nv-iK > Fv):
                    iK = iK+Fv
                if (iL < Fw and not iszero(fmod(f_L, f_Nw)) and Nw-iL > Fw):
                    iL = iL+Fw
                    
                i_dft = iL+Nw*(iK+Nv*iH)
                
                Qh = M_TAU*(B[0,0]*h+B[0,1]*k+B[0,2]*l)
                Qk = M_TAU*(B[1,0]*h+B[1,1]*k+B[1,2]*l)
                Ql = M_TAU*(B[2,0]*h+B[2,1]*k+B[2,2]*l)
                
                Qx = R[0,0]*Qh+R[0,1]*Qk+R[0,2]*Ql
                Qy = R[1,0]*Qh+R[1,1]*Qk+R[1,2]*Ql
                Qz = R[2,0]*Qh+R[2,1]*Qk+R[2,2]*Ql
                
                Q = sqrt(Qx*Qx+Qy*Qy+Qz*Qz)
                
                if iszero(Q):
                    Qx_norm, Qy_norm, Qz_norm = 0, 0, 0
                else:
                    Qx_norm, Qy_norm, Qz_norm = Qx/Q, Qy/Q, Qz/Q   
                
                s_ = Q*inv_M_SP
                s_sq = s_*s_
                    
                prod_x = 0
                prod_y = 0
                prod_z = 0
                
                for j in range(n_atm):
                    
                    j0 = A0[j]*exp(-a0[j]*s_sq)\
                       + B0[j]*exp(-b0[j]*s_sq)\
                       + C0[j]*exp(-c0[j]*s_sq)\
                       + D0[j]
                    
                    j2 = (A2[j]*exp(-a2[j]*s_sq)\
                       +  B2[j]*exp(-b2[j]*s_sq)\
                       +  C2[j]*exp(-c2[j]*s_sq)\
                       +  D2[j])*s_sq
                        
                    form_factor = j0+K2[j]*j2
                    
                    if (form_factor < 0):
                        form_factor = 0
                        
                    phase_factor = iexp(Qx*ux[j]+Qy*uy[j]+Qz*uz[j])
                    
                    factors = form_factor*phase_factor
                    
                    j_dft = j+n_atm*i_dft
                                 
                    prod_x = prod_x+factors*Sx_k[j_dft]
                    prod_y = prod_y+factors*Sy_k[j_dft]
                    prod_z = prod_z+factors*Sz_k[j_dft]
                
                Fx = prod_x
                Fy = prod_y
                Fz = prod_z
            
                Q_norm_dot_F = Qx_norm*Fx+Qy_norm*Fy+Qz_norm*Fz
            
                Fx_perp = Fx-Q_norm_dot_F*Qx_norm
                Fy_perp = Fy-Q_norm_dot_F*Qy_norm
                Fz_perp = Fz-Q_norm_dot_F*Qz_norm
                
                Fx_perp_real, Fx_perp_imag = Fx_perp.real, Fx_perp.imag
                Fy_perp_real, Fy_perp_imag = Fy_perp.real, Fy_perp.imag
                Fz_perp_real, Fz_perp_imag = Fz_perp.real, Fz_perp.imag
            
                I[i] += (Fx_perp_real*Fx_perp_real+Fx_perp_imag*Fx_perp_imag\
                     +   Fy_perp_real*Fy_perp_real+Fy_perp_imag*Fy_perp_imag\
                     +   Fz_perp_real*Fz_perp_real+Fz_perp_imag*Fz_perp_imag)\
                     *   factor*variants[var]

    return I_np

def occupational(double [::1] A_r, 
                 double [::1] occupancy,
                 double [::1] ux,
                 double [::1] uy,
                 double [::1] uz,
                 atms,
                 h_range,
                 k_range,
                 l_range,
                 long [::1] indices,
                 symop,
                 double [:,:] T,
                 double [:,:] B,
                 double [:,:] R,
                 double [:,:,:] domains,
                 double [:] variants,
                 Py_ssize_t nh,
                 Py_ssize_t nk,
                 Py_ssize_t nl,
                 Py_ssize_t nu,
                 Py_ssize_t nv,
                 Py_ssize_t nw,
                 Py_ssize_t Nu,
                 Py_ssize_t Nv,
                 Py_ssize_t Nw,
                 technique='Neutron'):
    
    cdef Py_ssize_t n_atm = len(atms)
        
    cdef Py_ssize_t n_hkl = indices.shape[0]
    
    cdef Py_ssize_t sym = symop[0]
    cdef Py_ssize_t n_ops = symop[1]
    
    cdef Py_ssize_t n_vars = variants.shape[0]
    
    cdef Py_ssize_t n_uvw = nu*nv*nw
    cdef Py_ssize_t n = n_uvw*n_atm
    
    cdef double factor
    
    if (n == 0):
        factor = 0
    else:
        factor = 1.0/n
        
    cdef double x_, y_, z_
    cdef double h_, k_, l_   

    cdef double x, y, z
    cdef double h, k, l
   
    cdef double Qh, Qk, Ql
    cdef double Qx, Qy, Qz
        
    cdef double complex phase_factor, factors
    
    cdef double Q, s_, s_sq
    
    cdef double occ
    
    cdef double complex scattering_length
    
    cdef double complex [::1] b = np.zeros(n_atm, dtype=complex)
    
    cdef double [::1] a1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c  = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
    
    A_r_np = np.copy(A_r, order='C')
    
    A_r_np = A_r_np.reshape(nu,nv,nw,n_atm)
    
    cdef double h_min_, k_min_, l_min_
    cdef double h_max_, k_max_, l_max_
    
    cdef double h_step_, k_step_, l_step_
                
    h_min_, k_min_, l_min_ = h_range[0], k_range[0], l_range[0]
    
    if (nh == 1):
        h_max_ = h_range[0]
        h_step_ = 0
    else:
        h_max_ = h_range[1]
        h_step_ = (h_max_-h_min_)/(nh-1)
        
    if (nk == 1):
        k_max_ = k_range[0]
        k_step_ = 0
    else:
        k_max_ = k_range[1]
        k_step_ = (k_max_-k_min_)/(nk-1)
        
    if (nl == 1):
        l_max_ = l_range[0]
        l_step_ = 0
    else:
        l_max_ = l_range[1]
        l_step_ = (l_max_-l_min_)/(nl-1)
        
    cdef Py_ssize_t Fu = Nu // nu
    cdef Py_ssize_t Fv = Nv // nv
    cdef Py_ssize_t Fw = Nw // nw
        
    cdef double f_H, f_K, f_L
    
    cdef double f_Nu = float(Nu)
    cdef double f_Nv = float(Nv)
    cdef double f_Nw = float(Nw)
    
    A_k_np = idft(A_r_np, Nu, Nv, Nw).flatten()*n_uvw
                          
    cdef double complex [::1] A_k = A_k_np
    
    cdef Py_ssize_t F_uvw = Fu*Fv*Fw
    cdef Py_ssize_t N_uvw = Nu*Nv*Nw

    cdef double complex prod
    cdef double complex F

    cdef double F_real, F_imag
    
    cdef Py_ssize_t H, K, L, iH, iK, iL
    cdef Py_ssize_t i_dft, i_ind, j_dft
    
    cdef Py_ssize_t op, var
    cdef Py_ssize_t i, j
    cdef Py_ssize_t u, v, w
    
    cdef double M_TAU = 2*np.pi
    cdef double complex M_I = 1j
    
    cdef double inv_M_SP = 1/(4*np.pi)
        
    for j in range(n_atm):
        
        atm = atms[j]
        
        if (technique == 'Neutron'):
            
            b[j] = tables.bc.get(atm)
        
        else:
        
            a1[j], b1[j], \
            a2[j], b2[j], \
            a3[j], b3[j], \
            a4[j], b4[j], \
            c[j] = tables.X.get(atm)
                            
    for i in prange(n_hkl, nogil=True):
            
        i_ind = indices[i]
        
        w = i_ind % nl 
        v = i_ind // nl % nk
        u = i_ind // nl // nk % nh
        
        h_ = h_min_+h_step_*u
        k_ = k_min_+k_step_*v
        l_ = l_min_+l_step_*w
        
        x = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
        y = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
        z = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
            
        for var in range(n_vars):
            
            x_ = domains[var,0,0]*x+domains[var,0,1]*y+domains[var,0,2]*z
            y_ = domains[var,1,0]*x+domains[var,1,1]*y+domains[var,1,2]*z
            z_ = domains[var,2,0]*x+domains[var,2,1]*y+domains[var,2,2]*z
        
            for op in range(n_ops):
                                        
                h, k, l = transform(x_, y_, z_, sym, op)
                        
                f_H, f_K, f_L = h*Nu, k*Nv, l*Nw
                
                H = int(round2even(f_H))
                K = int(round2even(f_K))
                L = int(round2even(f_L))
                
                iH = (H % Nu+Nu) % Nu
                iK = (K % Nv+Nv) % Nv
                iL = (L % Nw+Nw) % Nw
                
                if (iH < Fu and not iszero(fmod(f_H, f_Nu)) and Nu-iH > Fu):
                    iH = iH+Fu
                if (iK < Fv and not iszero(fmod(f_K, f_Nv)) and Nv-iK > Fv):
                    iK = iK+Fv
                if (iL < Fw and not iszero(fmod(f_L, f_Nw)) and Nw-iL > Fw):
                    iL = iL+Fw
                    
                i_dft = iL+Nw*(iK+Nv*iH)
                
                Qh = M_TAU*(B[0,0]*h+B[0,1]*k+B[0,2]*l)
                Qk = M_TAU*(B[1,0]*h+B[1,1]*k+B[1,2]*l)
                Ql = M_TAU*(B[2,0]*h+B[2,1]*k+B[2,2]*l)
                
                Qx = R[0,0]*Qh+R[0,1]*Qk+R[0,2]*Ql
                Qy = R[1,0]*Qh+R[1,1]*Qk+R[1,2]*Ql
                Qz = R[2,0]*Qh+R[2,1]*Qk+R[2,2]*Ql
                
                prod = 0
                
                for j in range(n_atm):
                    
                    occ = occupancy[j]
                    
                    if (technique == 'Neutron'):
                        
                        scattering_length = b[j]
                        
                    else:
                        
                        Q = sqrt(Qx*Qx+Qy*Qy+Qz*Qz)
                        
                        s_ = Q*inv_M_SP
                        s_sq = s_*s_
                                                
                        scattering_length = a1[j]*iexp(-b1[j]*s_sq)\
                                          + a2[j]*iexp(-b2[j]*s_sq)\
                                          + a3[j]*iexp(-b3[j]*s_sq)\
                                          + a4[j]*iexp(-b4[j]*s_sq)\
                                          + c[j]
                        
                    phase_factor = iexp(Qx*ux[j]+Qy*uy[j]+Qz*uz[j])
                    
                    factors = occ*scattering_length*phase_factor
                    
                    j_dft = j+n_atm*i_dft
                                 
                    prod = prod+factors*A_k[j_dft]
                
                F = prod
      
                F_real, F_imag = F.real, F.imag
    
                I[i] += (F_real*F_real+F_imag*F_imag)*factor*variants[var]

    return I_np

def displacive(double [::1] U_r,
               double complex [::1] coeffs,
               double [::1] ux,
               double [::1] uy,
               double [::1] uz,
               atms,
               h_range,
               k_range,
               l_range,
               long [::1] indices,
               symop,
               double [:,:] T,
               double [:,:] B,
               double [:,:] R,
               double [:,:,:] domains,
               double [:] variants,
               Py_ssize_t nh,
               Py_ssize_t nk,
               Py_ssize_t nl,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               Py_ssize_t Nu,
               Py_ssize_t Nv,
               Py_ssize_t Nw,
               Py_ssize_t p,
               long [::1] even,
               Py_ssize_t centering,
               technique='Neutron'):
    
    cdef Py_ssize_t n_atm = len(atms)
        
    cdef Py_ssize_t n_hkl = indices.shape[0]
    
    cdef Py_ssize_t n_prod = coeffs.shape[0]
    cdef Py_ssize_t n_even = even.shape[0]
    
    cdef Py_ssize_t sym = symop[0]
    cdef Py_ssize_t n_ops = symop[1]
    
    cdef Py_ssize_t n_vars = variants.shape[0]

    cdef Py_ssize_t n_uvw = nu*nv*nw
    cdef Py_ssize_t n = n_uvw*n_atm
    
    cdef double factor
    
    if (n == 0):
        factor = 0
    else:
        factor = 1.0/n
 
    cdef double x_, y_, z_   
    cdef double h_, k_, l_   

    cdef double x, y, z
    cdef double h, k, l
   
    cdef double Qh, Qk, Ql
    cdef double Qx, Qy, Qz
        
    cdef double complex phase_factor, factors
    
    cdef double Q, s_, s_sq
    
    cdef double occ
        
    cdef double complex scattering_length
    
    cdef double complex [::1] b = np.zeros(n_atm, dtype=complex)
    
    cdef double [::1] a1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c  = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
    
    U_r_np = np.copy(U_r, order='C')
    
    U_r_np = U_r_np.reshape(n_prod,nu,nv,nw,n_atm)
        
    cdef double h_min_, k_min_, l_min_
    cdef double h_max_, k_max_, l_max_
    
    cdef double h_step_, k_step_, l_step_
                
    h_min_, k_min_, l_min_ = h_range[0], k_range[0], l_range[0]
    
    if (nh == 1):
        h_max_ = h_range[0]
        h_step_ = 0
    else:
        h_max_ = h_range[1]
        h_step_ = (h_max_-h_min_)/(nh-1)
        
    if (nk == 1):
        k_max_ = k_range[0]
        k_step_ = 0
    else:
        k_max_ = k_range[1]
        k_step_ = (k_max_-k_min_)/(nk-1)
        
    if (nl == 1):
        l_max_ = l_range[0]
        l_step_ = 0
    else:
        l_max_ = l_range[1]
        l_step_ = (l_max_-l_min_)/(nl-1)
                
    cdef Py_ssize_t Fu = Nu // nu
    cdef Py_ssize_t Fv = Nv // nv
    cdef Py_ssize_t Fw = Nw // nw
    
    cdef double f_H, f_K, f_L
    
    cdef double f_Nu = float(Nu)
    cdef double f_Nv = float(Nv)
    cdef double f_Nw = float(Nw)
    
    U_k_np = idft_many(U_r_np, Nu, Nv, Nw).flatten()*n_uvw

    cdef double complex [::1] U_k = U_k_np

    cdef Py_ssize_t F_uvw = Fu*Fv*Fw
    cdef Py_ssize_t N_uvw = Nu*Nv*Nw
            
    cdef double Q_k
    
    cdef long [:,::1] exponents = np.zeros((n_prod,3), dtype=np.int)

    cdef double complex prod
    cdef double complex F

    cdef double F_real, F_imag
    
    cdef Py_ssize_t H, K, L, iH, iK, iL
    cdef Py_ssize_t i_dft, i_ind, j_dft
    
    cdef Py_ssize_t op, var
    cdef Py_ssize_t i, j
    cdef Py_ssize_t u, v, w
    
    cdef Py_ssize_t f, g, q, r, s, t
    
    cdef double M_TAU = 2*np.pi
    cdef double complex M_I = 1j
    
    cdef double inv_M_SP = 1/(4*np.pi)
        
    for j in range(n_atm):
        
        atm = atms[j]
        
        if (technique == 'Neutron'):
            
            b[j] = tables.bc.get(atm)
        
        else:
        
            a1[j], b1[j], \
            a2[j], b2[j], \
            a3[j], b3[j], \
            a4[j], b4[j], \
            c[j] = tables.X.get(atm)
            
    g = 0
    for f in range(p+1):
        for t in range(f+1):
            for s in range(f+1):
                for r in range(f+1):
                    if (r+s+t == f):
                        exponents[g,0] = r
                        exponents[g,1] = s
                        exponents[g,2] = t
                        g += 1
    
    cdef long [::1] odd = np.setdiff1d(np.union1d(even, np.arange(n_prod)), \
                                       np.intersect1d(even, np.arange(n_prod)))

    cdef Py_ssize_t n_odd = odd.shape[0]

    for i in prange(n_hkl, nogil=True):
            
        i_ind = indices[i]
        
        w = i_ind % nl 
        v = i_ind // nl % nk
        u = i_ind // nl // nk % nh
        
        h_ = h_min_+h_step_*u
        k_ = k_min_+k_step_*v
        l_ = l_min_+l_step_*w
        
        x = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
        y = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
        z = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
            
        for var in range(n_vars):
            
            x_ = domains[var,0,0]*x+domains[var,0,1]*y+domains[var,0,2]*z
            y_ = domains[var,1,0]*x+domains[var,1,1]*y+domains[var,1,2]*z
            z_ = domains[var,2,0]*x+domains[var,2,1]*y+domains[var,2,2]*z
        
            for op in range(n_ops):
                                        
                h, k, l = transform(x_, y_, z_, sym, op)
                
                f_H, f_K, f_L = h*Nu, k*Nv, l*Nw
                
                H = int(round2even(f_H))
                K = int(round2even(f_K))
                L = int(round2even(f_L))
                
                iH = (H % Nu+Nu) % Nu
                iK = (K % Nv+Nv) % Nv
                iL = (L % Nw+Nw) % Nw
                
                if (iH < Fu and not iszero(fmod(f_H, f_Nu)) and Nu-iH > Fu):
                    iH = iH+Fu
                if (iK < Fv and not iszero(fmod(f_K, f_Nv)) and Nv-iK > Fv):
                    iK = iK+Fv
                if (iL < Fw and not iszero(fmod(f_L, f_Nw)) and Nw-iL > Fw):
                    iL = iL+Fw
                                       
                i_dft = iL+Nw*(iK+Nv*iH)
                
                Qh = M_TAU*(B[0,0]*h+B[0,1]*k+B[0,2]*l)
                Qk = M_TAU*(B[1,0]*h+B[1,1]*k+B[1,2]*l)
                Ql = M_TAU*(B[2,0]*h+B[2,1]*k+B[2,2]*l)
                
                Qx = R[0,0]*Qh+R[0,1]*Qk+R[0,2]*Ql
                Qy = R[1,0]*Qh+R[1,1]*Qk+R[1,2]*Ql
                Qz = R[2,0]*Qh+R[2,1]*Qk+R[2,2]*Ql
                
                prod = 0
                
                for j in range(n_atm): 
                    
                    if (technique == 'Neutron'):
                        
                        scattering_length = b[j]
                        
                    else:
                        
                        Q = sqrt(Qx*Qx+Qy*Qy+Qz*Qz)
                        
                        s_ = Q*inv_M_SP
                        s_sq = s_*s_
                                                
                        scattering_length = a1[j]*iexp(-b1[j]*s_sq)\
                                          + a2[j]*iexp(-b2[j]*s_sq)\
                                          + a3[j]*iexp(-b3[j]*s_sq)\
                                          + a4[j]*iexp(-b4[j]*s_sq)\
                                          + c[j]
                                          
                    phase_factor = iexp(Qx*ux[j]+Qy*uy[j]+Qz*uz[j])
                    
                    factors = scattering_length*phase_factor
                    
                    if ((iH <= Fu and iK <= Fv and iL <= Fw) and \
                        nuclear(h, k, l, centering)):
                    
                        for g in range(n_odd):
                            
                            q = odd[g]
                        
                            j_dft = j+n_atm*(i_dft+N_uvw*q)
                            
                            Q_k = Qx**exponents[q,0]\
                                * Qy**exponents[q,1]\
                                * Qz**exponents[q,2]
    
                            prod = prod+factors*coeffs[q]*U_k[j_dft]*Q_k
                                                    
                    else:
                        
                        for q in range(n_prod):
                            
                            j_dft = j+n_atm*(i_dft+N_uvw*q)
    
                            Q_k = Qx**exponents[q,0]\
                                * Qy**exponents[q,1]\
                                * Qz**exponents[q,2]
                                                         
                            prod = prod+factors*coeffs[q]*U_k[j_dft]*Q_k
                    
                F = prod
                  
                F_real, F_imag = F.real, F.imag
    
                I[i] += (F_real*F_real+F_imag*F_imag)*factor*variants[var]

    return I_np

def structural(double [::1] occupancy,
               double [::1] ux,
               double [::1] uy,
               double [::1] uz,
               atms,
               h_range,
               k_range,
               l_range,
               long [::1] indices,
               symop,
               double [:,:] T,
               double [:,:] B,
               double [:,:] R,
               double [:,:,:] domains,
               double [:] variants,
               Py_ssize_t nh,
               Py_ssize_t nk,
               Py_ssize_t nl,
               Py_ssize_t nu,
               Py_ssize_t nv,
               Py_ssize_t nw,
               Py_ssize_t Nu,
               Py_ssize_t Nv,
               Py_ssize_t Nw,
               technique='Neutron'):
    
    cdef Py_ssize_t n_atm = len(atms)
        
    cdef Py_ssize_t n_hkl = indices.shape[0]
    
    cdef Py_ssize_t sym = symop[0]
    cdef Py_ssize_t n_ops = symop[1]
    
    cdef Py_ssize_t n_vars = variants.shape[0]
    
    cdef Py_ssize_t n_uvw = nu*nv*nw
    cdef Py_ssize_t n = n_uvw*n_atm
    
    cdef double factor
    
    if (n == 0):
        factor = 0
    else:
        factor = 1.0/n
        
    cdef double x_, y_, z_
    cdef double h_, k_, l_   

    cdef double x, y, z
    cdef double h, k, l
   
    cdef double Qh, Qk, Ql
    cdef double Qx, Qy, Qz
        
    cdef double complex phase_factor, factors
    
    cdef double Q, s_, s_sq
    
    cdef double occ
        
    cdef double complex scattering_length
    
    cdef double complex [::1] b = np.zeros(n_atm, dtype=complex)
    
    cdef double [::1] a1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b1 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b2 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b3 = np.zeros(n_atm, dtype=float)
    cdef double [::1] a4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] b4 = np.zeros(n_atm, dtype=float)
    cdef double [::1] c  = np.zeros(n_atm, dtype=float)
    
    I_np = np.zeros(n_hkl, dtype=float)
    
    cdef double [::1] I = I_np
        
    A_r_np = np.ones((nu,nv,nw,n_atm), dtype=float)
    
    cdef double h_min_, k_min_, l_min_
    cdef double h_max_, k_max_, l_max_
    
    cdef double h_step_, k_step_, l_step_
                
    h_min_, k_min_, l_min_ = h_range[0], k_range[0], l_range[0]
    
    if (nh == 1):
        h_max_ = h_range[0]
        h_step_ = 0
    else:
        h_max_ = h_range[1]
        h_step_ = (h_max_-h_min_)/(nh-1)
        
    if (nk == 1):
        k_max_ = k_range[0]
        k_step_ = 0
    else:
        k_max_ = k_range[1]
        k_step_ = (k_max_-k_min_)/(nk-1)
        
    if (nl == 1):
        l_max_ = l_range[0]
        l_step_ = 0
    else:
        l_max_ = l_range[1]
        l_step_ = (l_max_-l_min_)/(nl-1)
        
    cdef Py_ssize_t Fu = Nu // nu
    cdef Py_ssize_t Fv = Nv // nv
    cdef Py_ssize_t Fw = Nw // nw
    
    cdef double f_H, f_K, f_L
    
    cdef double f_Nu = float(Nu)
    cdef double f_Nv = float(Nv)
    cdef double f_Nw = float(Nw)
    
    A_k_np = idft(A_r_np, Nu, Nv, Nw).flatten()*n_uvw
                          
    cdef double complex [::1] A_k = A_k_np
    
    cdef Py_ssize_t F_uvw = Fu*Fv*Fw
    cdef Py_ssize_t N_uvw = Nu*Nv*Nw

    cdef double complex prod
    cdef double complex F

    cdef double F_real, F_imag
    
    cdef Py_ssize_t H, K, L, iH, iK, iL
    cdef Py_ssize_t i_dft, i_ind, j_dft
    
    cdef Py_ssize_t op, var
    cdef Py_ssize_t i, j
    cdef Py_ssize_t u, v, w
    
    cdef double M_TAU = 2*np.pi
    cdef double complex M_I = 1j
        
    cdef double inv_M_SP = 1/(4*np.pi)

    for j in range(n_atm):
        
        atm = atms[j]
        
        if (technique == 'Neutron'):
            
            b[j] = tables.bc.get(atm)
        
        else:
        
            a1[j], b1[j], \
            a2[j], b2[j], \
            a3[j], b3[j], \
            a4[j], b4[j], \
            c[j] = tables.X.get(atm)
                            
    for i in prange(n_hkl, nogil=True):
            
        i_ind = indices[i]
        
        w = i_ind % nl 
        v = i_ind // nl % nk
        u = i_ind // nl // nk % nh
        
        h_ = h_min_+h_step_*u
        k_ = k_min_+k_step_*v
        l_ = l_min_+l_step_*w
        
        x = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
        y = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
        z = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
            
        for var in range(n_vars):
            
            x_ = domains[var,0,0]*x+domains[var,0,1]*y+domains[var,0,2]*z
            y_ = domains[var,1,0]*x+domains[var,1,1]*y+domains[var,1,2]*z
            z_ = domains[var,2,0]*x+domains[var,2,1]*y+domains[var,2,2]*z
        
            for op in range(n_ops):
                                        
                h, k, l = transform(x_, y_, z_, sym, op)

                f_H, f_K, f_L = h*Nu, k*Nv, l*Nw
                
                H = int(round2even(f_H))
                K = int(round2even(f_K))
                L = int(round2even(f_L))
                
                iH = (H % Nu+Nu) % Nu
                iK = (K % Nv+Nv) % Nv
                iL = (L % Nw+Nw) % Nw
                
                if (iH < Fu and not iszero(fmod(f_H, f_Nu)) and Nu-iH > Fu):
                    iH = iH+Fu
                if (iK < Fv and not iszero(fmod(f_K, f_Nv)) and Nv-iK > Fv):
                    iK = iK+Fv
                if (iL < Fw and not iszero(fmod(f_L, f_Nw)) and Nw-iL > Fw):
                    iL = iL+Fw
                                       
                i_dft = iL+Nw*(iK+Nv*iH)
                
                Qh = M_TAU*(B[0,0]*h+B[0,1]*k+B[0,2]*l)
                Qk = M_TAU*(B[1,0]*h+B[1,1]*k+B[1,2]*l)
                Ql = M_TAU*(B[2,0]*h+B[2,1]*k+B[2,2]*l)
                
                Qx = R[0,0]*Qh+R[0,1]*Qk+R[0,2]*Ql
                Qy = R[1,0]*Qh+R[1,1]*Qk+R[1,2]*Ql
                Qz = R[2,0]*Qh+R[2,1]*Qk+R[2,2]*Ql
                
                prod = 0
                
                for j in range(n_atm):
                    
                    occ = occupancy[j]
                                            
                    if (technique == 'Neutron'):
                        
                        scattering_length = b[j]
                        
                    else:
                        
                        Q = sqrt(Qx*Qx+Qy*Qy+Qz*Qz)
                        
                        s_ = Q*inv_M_SP
                        s_sq = s_*s_
                                                
                        scattering_length = a1[j]*iexp(-b1[j]*s_sq)\
                                          + a2[j]*iexp(-b2[j]*s_sq)\
                                          + a3[j]*iexp(-b3[j]*s_sq)\
                                          + a4[j]*iexp(-b4[j]*s_sq)\
                                          + c[j]
                                          
                    phase_factor = iexp(Qx*ux[j]+Qy*uy[j]+Qz*uz[j])
                    
                    factors = occ*scattering_length*phase_factor
                    
                    j_dft = j+n_atm*i_dft
                                 
                    prod = prod+factors*A_k[j_dft]
                
                F = prod
      
                F_real, F_imag = F.real, F.imag
    
                I[i] += (F_real*F_real+F_imag*F_imag)*factor*variants[var]

    return I_np

def idft_many(X, Nu, Nv, Nw):
    
    nc = X.shape[0]
    nu = X.shape[1]
    nv = X.shape[2]
    nw = X.shape[3]
    n_atm = X.shape[4]
        
    U = np.zeros((nc,Nu,Nv,Nw,n_atm), dtype=complex)
    
    for i in range(nc):
        for j in range(n_atm):
            V = X[i,:,:,:,j].copy()
            
            W = V.copy()
            for _ in range(Nu // nu-1):
                W = np.concatenate((W,V*1),axis=0)
            W = np.concatenate((W,V[:(Nu%nu),:,:]*1),axis=0)
            
            V = W.copy()
            for _ in range(Nv // nv-1):
                W = np.concatenate((W,V*1),axis=1)
            W = np.concatenate((W,V[:,:(Nv%nv),:]*1),axis=1)
            
            V = W.copy()                
            for _ in range(Nw // nw-1):
                W = np.concatenate((W,V*1),axis=2)
            W = np.concatenate((W,V[:,:,:(Nw%nw)]*1),axis=2)
                
            U[i,:,:,:,j] = W.copy()
     
    x = np.fft.ifftn(U, axes=(1,2,3))
    
    for i in range(1, Nu // nu):
        x[:,i::(Nu//nu),:,:,:] += x[:,0::(Nu//nu),:,:,:]#*(Nu//nu-i)/(Nu//nu)
        
    for j in range(1, Nv // nv):
        x[:,:,j::(Nv//nv),:,:] += x[:,:,0::(Nv//nv),:,:]#*(Nv//nv-j)/(Nv//nv)
        
    for k in range(1, Nw // nw):
        x[:,:,:,k::(Nw//nw),:] += x[:,:,:,0::(Nw//nw),:]#*(Nw//nw-k)/(Nw//nw)
    
    return x

def idft(X, Nu, Nv, Nw):
    
    nu = X.shape[0]
    nv = X.shape[1]
    nw = X.shape[2]
    n_atm = X.shape[3]
        
    U = np.zeros((Nu,Nv,Nw,n_atm), dtype=complex)
    
    for j in range(n_atm):
        V = X[:,:,:,j].copy()
        
        W = V.copy()
        for _ in range(Nu // nu-1):
            W = np.concatenate((W,V*1),axis=0)
        W = np.concatenate((W,V[:(Nu%nu),:,:]*1),axis=0)
        
        V = W.copy()
        for _ in range(Nv // nv-1):
            W = np.concatenate((W,V*1),axis=1)
        W = np.concatenate((W,V[:,:(Nv%nv),:]*1),axis=1)
        
        V = W.copy()                
        for _ in range(Nw // nw-1):
            W = np.concatenate((W,V*1),axis=2)
        W = np.concatenate((W,V[:,:,:(Nw%nw)]*1),axis=2)
            
        U[:,:,:,j] = W.copy()
             
    x = np.fft.ifftn(U, axes=(0,1,2))
    
    for i in range(1, Nu // nu):
        x[i::(Nu//nu),:,:,:] += x[0::(Nu//nu),:,:,:]#*(Nu//nu-i)/(Nu//nu)
        
    for j in range(1, Nv // nv):
        x[:,j::(Nv//nv),:,:] += x[:,0::(Nv//nv),:,:]#*(Nv//nv-j)/(Nv//nv)
        
    for k in range(1, Nw // nw):
        x[:,:,k::(Nw//nw),:] += x[:,:,0::(Nw//nw),:]#*(Nw//nw-k)/(Nw//nw)
    
    return x