#!/usr/bin/env/python3

import re
import os

import numpy as np

from disorder.diffuse import experimental, space, scattering, monocrystal
from disorder.diffuse import magnetic, occupational, displacive, refinement
from disorder.material import crystal, symmetry, tables

import disorder.correlation.functions as correlations

from shutil import copyfile

import pyvista as pv

class Model:

    def __init__(self):
        pass
    
    def supercell_size(self, n_atm, nu, nv, nw):
        
        return n_atm*nu*nv*nw
    
    def unitcell_size(self, nu, nv, nw):
        
        return nu*nv*nw
    
    def ion_symbols(self, keys):
        
        return np.array([re.sub(r'[\d.+-]+$', '', key) for key in keys])

    def iso_symbols(self, keys):
        
        return np.array([re.sub(r'^\d+\s*', '', key) for key in keys])
    
    def remove_symbols(self, keys):
        
        return np.array([re.sub(r'[a-zA-Z]', '', key) for key in keys])

    def sort_keys(self, col0, col1, keys):
        
        keys = np.array([key for key in keys])
        sort = np.lexsort(np.array((col0, col1)))        
        return keys[sort]
    
    def get_isotope(self, element, nucleus):
        
        nucleus[nucleus == '-'] = ''
        return np.array([pre+atm for atm, pre in zip(element, nucleus)])
    
    def get_ion(self, element, charge):
        
        charge[charge == '-'] = ''
        return np.array([atm+app for atm, app in zip(element, charge)])
    
    def get_neutron_scattering_length_keys(self):
        
        bc_keys = tables.bc.keys()
        bc_atm = self.iso_symbols(bc_keys)
        bc_nuc = self.remove_symbols(bc_keys)
        return self.sort_keys(bc_nuc,bc_atm,bc_keys)
 
    def get_xray_form_factor_keys(self):
        
        X_keys = tables.X.keys()
        X_atm = self.ion_symbols(X_keys)
        X_ion = self.remove_symbols(X_keys)
        return self.sort_keys(X_ion,X_atm,X_keys)

    def get_magnetic_form_factor_keys(self):
        
        j0_keys = tables.j0.keys()
        j0_atm = self.ion_symbols(j0_keys)
        j0_ion = self.remove_symbols(j0_keys)
        return self.sort_keys(j0_ion,j0_atm,j0_keys) 
    
    def load_unit_cell(self, folder, filename):
        
        return crystal.unitcell(folder=folder, 
                                filename=filename,
                                occupancy=True,
                                displacement=True,
                                moment=True,
                                site=True,
                                operator=True,
                                magnetic_operator=True)
    
    def load_space_group(self, folder, filename):
                
        return crystal.group(folder=folder, filename=filename)        
    
    def load_lattice_parameters(self, folder, filename):
        
        return crystal.parameters(folder=folder, filename=filename)
    
    def find_laue(self, folder, filename):
        
        return crystal.laue(folder, filename)
    
    def find_lattice(self, a, b, c, alpha, beta, gamma):
    
        return crystal.lattice(a, b, c, alpha, beta, gamma)
    
    def crystal_matrices(self, a, b, c, alpha, beta, gamma):
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
    
        C, D = crystal.orthogonalized(a, b, c, alpha, beta, gamma)
        
        return A, B, R, C, D
    
    def crystal_reciprocal_matrices(self, a, b, c, alpha, beta, gamma):
        
        a_, b_, c_, \
        alpha_, beta_, gamma_ = crystal.reciprocal(a, b, c, 
                                                   alpha, beta, gamma)
      
        A_, B_, R_ = crystal.matrices(a_, b_, c_, alpha_, beta_, gamma_)
    
        C_, D_ = crystal.orthogonalized(a_, b_, c_, alpha_, beta_, gamma_)
        
        return A_, B_, R_, C_, D_
    
    def anisotropic_parameters(self, displacement, D):
        
        if (displacement.shape[1] == 6):
            U11, U22, U33, U23, U13, U12 = np.round(displacement.T, 4)
        else:
            Uiso = np.round(displacement.flatten(), 4)                
            uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
            U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
            U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]
            
        return U11, U22, U33, U23, U13, U12
    
    def atomic_displacement_parameters(self, U11, U22, U33, U23, U13, U12, D):

        U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
        n = np.size(U11)
        
        U = U.reshape(3,3,n)
        
        Uiso, U1, U2, U3 = [], [], [], []
        for i in range(n):
            Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D.T))
            Up.sort()
            U1.append(Up[0].real)
            U2.append(Up[1].real)
            U3.append(Up[2].real)
            Uiso.append(np.mean(Up).real)
        
        return np.array(Uiso), np.array(U1), np.array(U2), np.array(U3)
    
    def magnetic_moments(self, mu1, mu2, mu3, C):
        
        M = np.array([mu1,mu2,mu3])
        n = np.size(mu1)
        
        M = M.reshape(3,n)

        mu = []
        for i in range(n):
            mu.append(np.linalg.norm(np.dot(C, M[:,i])))
        
        return np.array(mu)
    
    def magnetic_symmetry(self, operator, moment):
                        
        return symmetry.evaluate_mag(operator, moment)
    
    def symmetry(self, operator, coordinate):
        
        coord = symmetry.evaluate(operator, coordinate)
                
        return [c+(c < 0)-(c > 1) for c in coord]
    
    def reverse_symmetry(self, operator, coordinate):
        
        rev_operator = symmetry.reverse(operator)[0]
        
        coord = symmetry.evaluate(rev_operator, coordinate)
        
        return [c+(c < 0)-(c > 1) for c in coord]
    
    def save_crystal(self, filename, fname):
        
        if (filename != fname):
            copyfile(filename, fname)
            
    def save_supercell(self, fname, atm, occ, disp, mom,
                       u, v, w, nu, nv, nw, folder, filename):
        
        crystal.supercell(atm, occ, disp, mom, u, v, w, nu, nv, nw,
                          fname, folder=folder, filename=filename)
        
    def save_disorder(self, fname, Sx, Sy, Sz, delta, Ux, Uy, Uz, rx, ry, rz,
                      nu, nv, nw, atm, A, folder, filename):
        
        crystal.disordered(delta, Ux, Uy, Uz, Sx, Sy, Sz, 
                           rx, ry, rz, nu, nv, nw, atm, A, fname, 
                           folder=folder, filename=filename, 
                           ulim=[0,nu], vlim=[0,nv], wlim=[0,nw])
    
    def load_data(self, fname):
        
        if fname.endswith('.nxs'):
            signal, sigma_sq, \
            h_range, k_range, l_range, \
            nh, nk, nl = experimental.data(fname)
        elif fname.endswith('.npz'):
            npzfile = np.load(fname, allow_pickle=True)
            signal = npzfile['signal']
            sigma_sq = npzfile['sigma_sq']
            limits = npzfile['limits']
            min_h, max_h, nh, min_k, max_k, nk, min_l, max_l, nl = limits
            h_range = [min_h, max_h]
            k_range = [min_k, max_k]
            l_range = [min_l, max_l]
        
        return signal, sigma_sq, h_range, k_range, l_range, nh, nk, nl
    
    def save_data(self, fname, signal, sigma_sq, 
                  h_range, k_range, l_range, nh, nk, nl):
        
        min_h, max_h = h_range
        min_k, max_k = k_range
        min_l, max_l = l_range
        
        limits = np.array([min_h, max_h, nh, 
                           min_k, max_k, nk, 
                           min_l, max_l, nl], dtype=object)
        
        np.savez('{}-intensity.npz'.format(fname), 
                 signal=signal, sigma_sq=sigma_sq, limits=limits)
        
    def load_region_of_interest(self, fname):
        
        signal = np.load('{}-intensity-roi.npy'.format(fname))
        sigma_sq = np.load('{}-error-roi.npy'.format(fname))
        
        return signal, sigma_sq
      
    def save_region_of_interest(self, fname, signal, sigma_sq):
        
        np.save('{}-intensity-roi.npy'.format(fname), signal)
        np.save('{}-error-roi.npy'.format(fname), sigma_sq)
    
    def rebin_parameters(self, size, minimum, maximum, centered=True):
        
        if (size > 0):
            step = (maximum-minimum)/(size-1)
       
            if centered:         
                round_min = round(minimum)
                round_max = round(maximum)  
                offset_min = int(np.round((round_min-minimum)/step, 4))
                offset_max = int(np.round((round_max-minimum)/step, 4))
                scale = experimental.factors(offset_max-offset_min)
                mask = np.isclose(np.mod(1/(step*scale), 1), 0)
                scale = scale[mask]
            else:
                scale = experimental.factors(size-1)            
            
            mask = step*scale <= 1
            scale = scale[mask]
            
            steps = np.round(step*scale, 4)
            sizes = (size-1) // scale+1
             
            return steps, sizes
    
        else:
            return np.array([]), np.array([])
    
    def slice_value(self, minimum, maximum, size, index):
        
        if (index > size):
            return np.round(maximum, 4)
        elif (index < 0 or size <= 1):
            return np.round(minimum, 4)
        else:
            step = (maximum-minimum)/(size-1)
            return np.round(minimum+step*index, 4)
    
    def slice_index(self, minimum, maximum, size, value):
              
        if (value > maximum):
            return size-1
        elif (value < minimum or size <= 1):
            return 0
        else:
            step = (maximum-minimum)/(size-1)
            return int(round((value-minimum)/step))
        
    def step_value(self, minimum, maximum, size):
        
        return (maximum-minimum)/(size-1) if (size > 1) else 0
    
    def size_value(self, minimum, maximum, step):
        
        return int(round((maximum-minimum)/step))+1 if (step > 0) else 1
    
    def minimum_value(self, size, step, maximum):
        
        return maximum-step*(size-1)
    
    def maximum_value(self, size, step, minimum):
        
        return minimum+step*(size-1)
            
    def matrix_transform(self, B, layer='l', T=np.eye(3)):
    
        M = np.eye(3)
        
        Bp = np.linalg.cholesky(np.dot(T.T,np.dot(np.dot(B.T,B),T))).T
        
        if (layer == 'h'):
            Q = Bp[1:3,1:3].copy()
        elif (layer == 'k'):
            Q = Bp[0:3:2,0:3:2].copy()
        elif (layer == 'l'):
            Q = Bp[0:2,0:2].copy()     
                       
        Q /= Q[1,1]
        
        scale = 1/Q[0,0]
        Q[0,:] *= scale
        
        M[0:2,0:2] = Q
        
        return M, scale
    
    def mask_array(self, array):
        
        return np.ma.masked_less_equal(np.ma.masked_invalid(array, copy=False),
                                       0, copy=False)
        
    def crop(self, array, h_slice, k_slice, l_slice):
        
        return experimental.crop(array, h_slice, k_slice, l_slice)
    
    def rebin(self, array, binsize):
        
        return experimental.rebin(array, binsize)
    
    def crop_parameters(self, xmin, xmax, minimum, maximum, size):
        
        binning = np.linspace(minimum, maximum, size)
        i_min = np.where(binning <= xmin)[0][-1]
        i_max = np.where(binning <= xmax)[0][-1]
        
        return i_min, i_max

    def punch(self, array, radius_h, radius_k, radius_l, 
              step_h, step_k, step_l, h_range, k_range, l_range,
              centering, outlier, punch):
        
        return experimental.punch(array, radius_h, radius_k, radius_l,
                   step_h, step_k, step_l, h_range, k_range, l_range,
                   centering=centering, outlier=outlier, punch=punch)
    
    def get_mask(self, signal, error_sq):
        
        return experimental.mask(signal, error_sq)
    
    def get_refinement_data(self, signal, error_sq, mask):
        
            return signal[~mask], 1/error_sq[~mask]
    
    def reciprocal_mapping(self, h_range, k_range, l_range, nu, nv, nw, mask):
        
        nh, nk, nl = mask.shape
        
        h, k, l, \
        H, K, L, \
        indices, inverses, \
        operators = crystal.bragg(h_range, k_range, l_range, 
                                  nh, nk, nl, nu, nv, nw)
        
        i_mask, i_unmask = space.indices(inverses, mask)
            
        return h, k, l, H, K, L, indices, inverses, i_mask, i_unmask 
    
    def reciprocal_space_coordinate_transform(self, h, k, l, B, R):
            
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        return Qx, Qy, Qz, Qx_norm, Qy_norm, Qz_norm, Q 
    
    def real_space_coordinate_transform(self, u, v, w, atm, A, nu, nv, nw):
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        return ux, uy, uz, rx, ry, rz, atms
        
    def exponential_factors(self, Qx, Qy, Qz, ux, uy, uz, nu, nv, nw):
                
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
        
        space_factor = space.factor(nu, nv, nw)
        
        return phase_factor, space_factor

    def neutron_factors(self, Q, atm, ion, occupancy, g, phase_factor):
                    
        scattering_length = scattering.length(atm, Q.size)
                    
        factors = space.prefactors(scattering_length, phase_factor, occupancy)
        
        magnetic_form_factor = magnetic.form(Q, ion, g)
        
        magnetic_factors = space.prefactors(magnetic_form_factor, 
                                            phase_factor, occupancy)
        
        return factors, magnetic_factors
    
    def xray_factors(self, Q, ion, occupancy, phase_factor):
            
        form_factor = scattering.form(ion, Q)
        
        factors = space.prefactors(form_factor, phase_factor, occupancy)

        return factors       

    def initialize_intensity(self, mask, Q):
        
        nh, nk, nl = mask.shape
        
        I_obs = np.full((nh, nk, nl), np.nan)
        I_ref = I_obs[~mask]
        
        I_calc = np.zeros(Q.size, dtype=float)
        
        I_raw = np.zeros(mask.size, dtype=float)
        I_flat = np.zeros(mask.size, dtype=float)
        
        return I_obs, I_ref, I_calc, I_raw, I_flat
        
    def initialize_filter(self, mask):

        a_filt = np.zeros(mask.size, dtype=float)
        b_filt = np.zeros(mask.size, dtype=float)
        c_filt = np.zeros(mask.size, dtype=float)
        d_filt = np.zeros(mask.size, dtype=float)
        e_filt = np.zeros(mask.size, dtype=float)
        f_filt = np.zeros(mask.size, dtype=float)
        g_filt = np.zeros(mask.size, dtype=float)
        h_filt = np.zeros(mask.size, dtype=float)
        i_filt = np.zeros(mask.size, dtype=float)
        
        return a_filt, b_filt, c_filt, \
               d_filt, e_filt, f_filt, \
               g_filt, h_filt, i_filt
               
    def blurring(self, intensity, sigma):
                   
        return space.blurring(intensity, sigma)
    
    def gaussian(self, mask, sigma):
                   
        v_inv = space.gaussian(mask, sigma)
        
        boxes = space.boxblur(sigma, 3)
            
        return v_inv, boxes
        
    def random_moments(self, nu, nv, nw, n_atm, moment):
    
        return magnetic.spin(nu, nv, nw, n_atm, moment)
        
    def random_occupancies(self, nu, nv, nw, n_atm, occupancy):
        
        return occupational.composition(nu, nv, nw, n_atm, value=occupancy)
    
    def random_displacements(self, nu, nv, nw, n_atm, displacement):
        
        return displacive.expansion(nu, nv, nw, n_atm, value=displacement)
            
    def initialize_magnetic(self, Sx, Sy, Sz, H, K, L, 
                            Qx_norm, Qy_norm, Qz_norm, indices, 
                            magnetic_factors, nu, nv, nw, n_atm):
        
        n_uvw = nu*nv*nw
        
        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, Sy, Sz, H, K, L, 
                                                     nu, nv, nw, n_atm)
        
        Fx, Fy, Fz, \
        prod_x, prod_y, prod_z = magnetic.structure(Qx_norm, Qy_norm, Qz_norm, 
                                                    Sx_k, Sy_k, Sz_k, i_dft, 
                                                    magnetic_factors)
        
        Fx_orig = np.zeros(indices.size, dtype=complex)
        Fy_orig = np.zeros(indices.size, dtype=complex)
        Fz_orig = np.zeros(indices.size, dtype=complex)
        
        prod_x_orig = np.zeros(indices.size, dtype=complex)
        prod_y_orig = np.zeros(indices.size, dtype=complex)
        prod_z_orig = np.zeros(indices.size, dtype=complex)
        
        Sx_k_orig = np.zeros(n_uvw, dtype=complex)
        Sy_k_orig = np.zeros(n_uvw, dtype=complex)
        Sz_k_orig = np.zeros(n_uvw, dtype=complex)
        
        Fx_cand = np.zeros(indices.size, dtype=complex)
        Fy_cand = np.zeros(indices.size, dtype=complex)
        Fz_cand = np.zeros(indices.size, dtype=complex)
        
        prod_x_cand = np.zeros(indices.size, dtype=complex)
        prod_y_cand = np.zeros(indices.size, dtype=complex)
        prod_z_cand = np.zeros(indices.size, dtype=complex)
        
        Sx_k_cand = np.zeros(n_uvw, dtype=complex)
        Sy_k_cand = np.zeros(n_uvw, dtype=complex)
        Sz_k_cand = np.zeros(n_uvw, dtype=complex)
        
        return Sx_k, Sy_k, Sz_k, \
               Sx_k_orig, Sy_k_orig, Sz_k_orig, \
               Sx_k_cand, Sy_k_cand, Sz_k_cand, \
               Fx, Fy, Fz, \
               Fx_orig, Fy_orig, Fz_orig, \
               Fx_cand, Fy_cand, Fz_cand, \
               prod_x, prod_y, prod_z, \
               prod_x_orig, prod_y_orig, prod_z_orig, \
               prod_x_cand, prod_y_cand, prod_z_cand, i_dft
               
    def initialize_occupational(self, A_r, H, K, L, indices, 
                                factors, nu, nv, nw, n_atm):
        
        n_uvw = nu*nv*nw
                                                
        A_k, i_dft = occupational.transform(A_r, H, K, L, nu, nv, nw, n_atm)
            
        F, prod = occupational.structure(A_k, i_dft, factors)
                        
        F_orig = np.zeros(indices.size, dtype=complex)
        
        prod_orig = np.zeros(indices.size, dtype=complex)
        
        A_k_orig = np.zeros(n_uvw, dtype=complex)
        
        F_cand = np.zeros(indices.size, dtype=complex)
        
        prod_cand = np.zeros(indices.size, dtype=complex)
        
        A_k_cand = np.zeros(n_uvw, dtype=complex)
        
        return A_k, A_k_orig, A_k_cand, F, F_orig, F_cand, \
               prod, prod_orig, prod_cand, i_dft
            
    def initialize_displacive(self, Ux, Uy, Uz, h, k, l, H, K, L, Qx, Qy, Qz, 
                              indices, factors, nu, nv, nw, n_atm, 
                              p, centering):
        
        n_uvw = nu*nv*nw

        coeffs = displacive.coefficients(p)
        
        U_r = displacive.products(Ux, Uy, Uz, p)
        Q_k = displacive.products(Qx, Qy, Qz, p)
        
        U_k, i_dft = displacive.transform(U_r, H, K, L, nu, nv, nw, n_atm)
        
        H_nuc, K_nuc, L_nuc, \
        cond = crystal.nuclear(H, K, L, h, k, l, nu, nv, nw, centering)    
        
        F, F_nuc, \
        prod, prod_nuc, \
        V_k, V_k_nuc, \
        even, \
        bragg = displacive.structure(U_k, Q_k, coeffs, cond, p, i_dft, factors)
        
        F_orig = np.zeros(indices.size, dtype=complex)
        F_nuc_orig = np.zeros(bragg.size, dtype=complex)
        
        prod_orig = np.zeros(indices.size, dtype=complex)
        prod_nuc_orig = np.zeros(bragg.size, dtype=complex)
        
        V_k_orig = np.zeros(indices.size, dtype=complex)
        V_k_nuc_orig = np.zeros(bragg.size, dtype=complex)
        
        U_k_orig = np.zeros(n_uvw*coeffs.size, dtype=complex)
        
        F_cand = np.zeros(indices.shape, dtype=complex)
        F_nuc_cand = np.zeros(bragg.shape, dtype=complex)
        
        prod_cand = np.zeros(indices.shape, dtype=complex)
        prod_nuc_cand = np.zeros(bragg.shape, dtype=complex)
        
        V_k_cand = np.zeros(indices.size, dtype=complex)
        V_k_nuc_cand = np.zeros(bragg.size, dtype=complex)
        
        U_k_cand = np.zeros(n_uvw*coeffs.size, dtype=complex)
        
        U_r_orig = np.zeros(coeffs.size, dtype=float)
        
        U_r_cand = np.zeros(coeffs.size, dtype=float)
        
        return U_r, U_r_orig, U_r_cand, Q_k, \
               U_k, U_k_orig, U_k_cand, \
               V_k, V_k_orig, V_k_cand, \
               V_k_nuc, V_k_nuc_orig, V_k_nuc_cand, \
               F, F_orig, F_cand, \
               F_nuc, F_nuc_orig, F_nuc_cand, \
               prod, prod_orig, prod_cand, \
               prod_nuc, prod_nuc_orig, prod_nuc_cand, \
               i_dft, coeffs, H_nuc, K_nuc, L_nuc, cond, even, bragg
               
    def reduced_crystal_symmetry(self, h_range, k_range, l_range, nh, nk, nl, 
                                 nu, nv, nw, T, laue):
               
        indices, inverses, operators, \
        Nu, Nv, Nw = crystal.reduced(h_range, k_range, l_range, nh, nk, nl,
                                     nu, nv, nw, T=T, laue=laue)
        
        lauesym = symmetry.operators(invert=True)
        
        symmetries = list(lauesym.keys())
        
        symop = [11,1]
        
        for count, sym in enumerate(symmetries):
            if (np.array([operators[p] in lauesym.get(sym) \
                for p in range(operators.shape[0])]).all() and \
                len(lauesym.get(sym)) == operators.shape[0]):
                
                symop = [count,len(lauesym.get(sym))]
                
        return indices, inverses, operators, Nu, Nv, Nw, symop
    
    def displacive_parameters(self, p, centering):
         
        coeffs = displacive.coefficients(p)
        
        start = (np.cumsum(displacive.number(np.arange(p+1)))
              -  displacive.number(np.arange(p+1)))[::2]
        end = np.cumsum(displacive.number(np.arange(p+1)))[::2]
        
        even = []
        for k in range(len(end)):
            even += range(start[k], end[k])
        even = np.array(even)
                
        nuclear = ['P', 'I', 'F', 'R', 'C', 'A', 'B']
                            
        cntr = np.argwhere([x in centering for x in nuclear])[0][0]
        
        cntr += 1
        
        return coeffs, even, cntr
    
    def save_magnetic(self, fname, run, Sx, Sy, Sz):

        np.save('{}-calculated-spin-x-{}.npy'.format(fname,run), Sx)
        np.save('{}-calculated-spin-y-{}.npy'.format(fname,run), Sy)
        np.save('{}-calculated-spin-z-{}.npy'.format(fname,run), Sz)
        
    def save_occupational(self, fname, run, A_r):
        
        np.save('{}-calculated-composition-{}.npy'.format(fname,run), A_r)
    
    def save_displacive(self, fname, run, Ux, Uy, Uz):

        np.save('{}-calculated-displacement-x-{}.npy'.format(fname,run), Ux)
        np.save('{}-calculated-displacement-y-{}.npy'.format(fname,run), Uy)
        np.save('{}-calculated-displacement-z-{}.npy'.format(fname,run), Uz)
        
    def load_magnetic(self, fname, run):
    
        Sx = np.load('{}-calculated-spin-x-{}.npy'.format(fname,run))
        Sy = np.load('{}-calculated-spin-y-{}.npy'.format(fname,run))
        Sz = np.load('{}-calculated-spin-z-{}.npy'.format(fname,run))
        
        return Sx, Sy, Sz
                                
    def load_occupational(self, fname, run):
                    
        A_r = np.load('{}-calculated-composition-{}.npy'.format(fname,run))
        
        return A_r

    def load_displacive(self, fname, run):
                
        Ux = np.load('{}-calculated-displacement-x-{}.npy'.format(fname,run))
        Uy = np.load('{}-calculated-displacement-y-{}.npy'.format(fname,run))
        Uz = np.load('{}-calculated-displacement-z-{}.npy'.format(fname,run))   
        
        return Ux, Uy, Uz
            
    def save_refinement(self, fname, run, I_obs, chi_sq, energy, temperature, 
                        scale, acc_moves, rej_moves, acc_temps, rej_temps):
        
        np.save('{}-calculated-intensity-{}.npy'.format(fname,run), I_obs)
            
        np.save('{}-goodness-of-fit-{}.npy'.format(fname,run), chi_sq)
        np.save('{}-energy-{}.npy'.format(fname,run), energy)        
        np.save('{}-temperature-{}.npy'.format(fname,run), temperature)
        np.save('{}-scale-factor-{}.npy'.format(fname,run), scale)

        np.save('{}-accepted-moves-{}.npy'.format(fname,run), acc_moves)
        np.save('{}-rejected-moves-{}.npy'.format(fname,run), rej_moves)     
        
        np.save('{}-accepted-temperature-{}.npy'.format(fname,run), acc_temps)
        np.save('{}-rejected-temperature-{}.npy'.format(fname,run), rej_temps)
        
    def load_refinement(self, fname, run):
        
        I_obs = np.load('{}-calculated-intensity-{}.npy'.format(fname,run))
                    
        chi_sq = np.load('{}-goodness-of-fit-{}.npy'.format(fname,run))
        energy = np.load('{}-energy-{}.npy'.format(fname,run))        
        temperature = np.load('{}-temperature-{}.npy'.format(fname,run))
        scale = np.load('{}-scale-factor-{}.npy'.format(fname,run))

        acc_moves = np.load('{}-accepted-moves-{}.npy'.format(fname,run))
        rej_moves = np.load('{}-rejected-moves-{}.npy'.format(fname,run))     
        
        acc_temps = np.load('{}-accepted-temperature-{}.npy'.format(fname,run))
        rej_temps = np.load('{}-rejected-temperature-{}.npy'.format(fname,run))
        
        return I_obs, chi_sq.tolist(), energy.tolist(), \
               temperature.tolist(), scale.tolist(), \
               acc_moves.tolist(), rej_moves.tolist(), \
               acc_temps.tolist(), rej_temps.tolist()
               
    def save_recalculation(self, fname, I_recalc):

        np.save('{}-intensity-recalc.npy'.format(fname), I_recalc)
        
    def load_recalculation(self, fname):
        
        if os.path.isfile('{}-intensity-recalc.npy'.format(fname)):

            I_recalc = np.load('{}-intensity-recalc.npy'.format(fname)) 
            
            return I_recalc
        
        else:
            
            return None
        
    def save_correlations_1d(self, fname, data, header):
        
        np.savetxt(fname, np.column_stack(data), delimiter=',',
                   fmt='%s', header=header)
        
    def save_correlations_3d(self, fname, data, label):
        
        blocks = pv.MultiBlock()
        points = np.column_stack((data[0],data[1],data[2]))
        
        vectors = ['Correlation', 'Collinearity']
        scalars = ['Correlation']
        
        if (label == 'vector-pair'):
            datasets = [data[3], data[4]]
            pairs = data[5]
        elif (label == 'scalar-pair'):
            datasets = [data[3]]
            pairs = data[4]
        elif (label == 'vector'):
            datasets = [data[3], data[4]]
        elif (label == 'scalar'):
            datasets = [data[3]]
            
        if label.endswith('pair'):
            labels = np.unique(pairs)
            for t, array in zip(vectors, datasets):
                for label in labels:
                    mask = pairs == label
                    blocks[t+'-'+label] = pv.PolyData(points[mask])
                    blocks[t+'-'+label].point_arrays[t] = array[mask]
        else:
            for t, array in zip(vectors, datasets):
                blocks[t] = pv.PolyData(points)
                blocks[t].point_arrays[t] = array
                    
        blocks.save(fname, binary=False)
    
    def magnetic_intensity(self, fname, run, ux, uy, uz, atm,
                           h_range, k_range, l_range, indices, symop,
                           T, B, R, twins, variants, nh, nk, nl, nu, nv, nw,
                           Nu, Nv, Nw, g, mask):
            
        Sx, Sy, Sz = self.load_magnetic(fname, run)
        
        n_atm = np.size(Sx) // (nu*nv*nw)
        
        Sx = Sx.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
        Sy = Sy.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
        Sz = Sz.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
                
        I_calc = monocrystal.magnetic(Sx, Sy, Sz, ux, uy, uz, atm,
                                      h_range, k_range, l_range, indices,
                                      symop, T, B, R, twins, variants,
                                      nh, nk, nl, nu, nv, nw, Nu, Nv, Nw, g)
        
        return I_calc
                                
    def occupational_intensity(self, fname, run, occupancy, ux, uy, uz, atm,
                               h_range, k_range, l_range, indices, symop,
                               T, B, R, twins, variants, nh, nk, nl, 
                               nu, nv, nw, Nu, Nv, Nw, mask):
                    
        A_r = self.load_occupational(fname, run)
        
        n_atm = np.size(A_r) // (nu*nv*nw)
        
        A_r = A_r.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
                
        I_calc = monocrystal.occupational(A_r, occupancy, ux, uy, uz, atm,
                                          h_range, k_range, l_range, indices, 
                                          symop, T, B, R, twins, variants,
                                          nh, nk, nl, nu, nv, nw, Nu, Nv, Nw)
        
        return I_calc

    def displacive_intensity(self, fname, run, coeffs, ux, uy, uz, atm,
                             h_range, k_range, l_range, indices, symop,
                             T, B, R, twins, variants, nh, nk, nl, nu, nv, nw,
                             Nu, Nv, Nw, p, even, cntr, mask):
                
        Ux, Uy, Uz = self.load_displacive(fname, run)
        
        n_atm = np.size(Ux) // (nu*nv*nw)
        
        Ux = Ux.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
        Uy = Uy.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
        Uz = Uz.reshape(nu,nv,nw,n_atm).T[mask].T.flatten()
            
        U_r = displacive.products(Ux, Uy, Uz, p)
                
        I_calc = monocrystal.displacive(U_r, coeffs, ux, uy, uz, atm,
                                        h_range, k_range, l_range, indices,
                                        symop, T, B, R, twins, variants,
                                        nh, nk, nl, nu, nv, nw, Nu, Nv, Nw,
                                        p, even, cntr)
                    
        return I_calc
    
    def structural_intensity(self, occupancy, ux, uy, uz, atm,
                             h_range, k_range, l_range, indices, symop,
                             T, B, R, twins, variants, nh, nk, nl, 
                             nu, nv, nw, Nu, Nv, Nw, mask):
                            
        n_atm = np.size(occupancy)
                
        I_calc = monocrystal.structural(occupancy, ux, uy, uz, atm,
                                        h_range, k_range, l_range, indices, 
                                        symop, T, B, R, twins, variants,
                                        nh, nk, nl, nu, nv, nw, Nu, Nv, Nw)
        
        return I_calc
    
    def magnetic_refinement(self, Sx, Sy, Sz, Qx_norm, Qy_norm, Qz_norm, 
                            Sx_k, Sy_k, Sz_k, 
                            Sx_k_orig, Sy_k_orig, Sz_k_orig,
                            Sx_k_cand, Sy_k_cand, Sz_k_cand,
                            Fx, Fy, Fz,
                            Fx_orig, Fy_orig, Fz_orig,
                            Fx_cand, Fy_cand, Fz_cand,
                            prod_x, prod_y, prod_z,
                            prod_x_orig, prod_y_orig, prod_z_orig,  
                            prod_x_cand, prod_y_cand, prod_z_cand,   
                            space_factor, factors, moment, I_calc, I_expt, 
                            inv_sigma_sq, I_raw, I_flat, I_ref, v_inv, 
                            a_filt, b_filt, c_filt, 
                            d_filt, e_filt, f_filt, 
                            g_filt, h_filt, i_filt,
                            boxes, i_dft, inverses, i_mask, i_unmask,
                            acc_moves, acc_temps, rej_moves, rej_temps, chi_sq, 
                            energy, temperature, scale, constant, delta, fixed, 
                            T, nh, nk, nl, nu, nv, nw, n_atm, n, N):

        refinement.magnetic(Sx, Sy, Sz, Qx_norm, Qy_norm, Qz_norm, 
                            Sx_k, Sy_k, Sz_k, 
                            Sx_k_orig, Sy_k_orig, Sz_k_orig,
                            Sx_k_cand, Sy_k_cand, Sz_k_cand,
                            Fx, Fy, Fz,
                            Fx_orig, Fy_orig, Fz_orig,
                            Fx_cand, Fy_cand, Fz_cand,
                            prod_x, prod_y, prod_z,
                            prod_x_orig, prod_y_orig, prod_z_orig,  
                            prod_x_cand, prod_y_cand, prod_z_cand,   
                            space_factor, factors, moment, I_calc, I_expt, 
                            inv_sigma_sq, I_raw, I_flat, I_ref, v_inv, 
                            a_filt, b_filt, c_filt, 
                            d_filt, e_filt, f_filt, 
                            g_filt, h_filt, i_filt,
                            boxes, i_dft, inverses, i_mask, i_unmask,
                            acc_moves, acc_temps, rej_moves, rej_temps, chi_sq, 
                            energy, temperature, scale, constant, delta, fixed, 
                            T, nh, nk, nl, nu, nv, nw, n_atm, n, N)
            
    def occupational_refinement(self, A_r, A_k, A_k_orig, A_k_cand,
                                F, F_orig, F_cand,
                                prod, prod_orig, prod_cand,     
                                space_factor, factors, occupancy,
                                I_calc, I_expt, inv_sigma_sq,
                                I_raw, I_flat, I_ref, v_inv, 
                                a_filt, b_filt, c_filt,
                                d_filt, e_filt, f_filt,
                                g_filt, h_filt, i_filt,
                                boxes, i_dft, inverses, i_mask, i_unmask,
                                acc_moves, acc_temps, rej_moves, rej_temps,
                                chi_sq, energy, temperature, scale, constant, 
                                fixed, nh, nk, nl, nu, nv, nw, n_atm, n, N):

        refinement.occupational(A_r, A_k, A_k_orig, A_k_cand,
                                F, F_orig, F_cand,
                                prod, prod_orig, prod_cand,     
                                space_factor, factors, occupancy,
                                I_calc, I_expt, inv_sigma_sq, 
                                I_raw, I_flat, I_ref, v_inv, 
                                a_filt, b_filt, c_filt,
                                d_filt, e_filt, f_filt,
                                g_filt, h_filt, i_filt,
                                boxes, i_dft, inverses, i_mask, i_unmask,
                                acc_moves, acc_temps, rej_moves, rej_temps,
                                chi_sq, energy, temperature, scale, constant,
                                fixed, nh, nk, nl, nu, nv, nw, n_atm, n, N)
        
    def displacive_refinement(self, Ux, Uy, Uz,
                              U_r, U_r_orig, U_r_cand,
                              U_k, U_k_orig, U_k_cand,
                              V_k, V_k_nuc, V_k_orig,
                              V_k_nuc_orig, V_k_cand, V_k_nuc_cand,
                              F, F_nuc, F_orig, F_nuc_orig, F_cand, F_nuc_cand,
                              prod, prod_nuc, prod_orig, prod_nuc_orig,    
                              prod_cand, prod_nuc_cand, space_factor, factors,
                              coeffs, Q_k, displacement,
                              I_calc, I_expt, inv_sigma_sq, 
                              I_raw, I_flat, I_ref, v_inv, 
                              a_filt, b_filt, c_filt,
                              d_filt, e_filt, f_filt,
                              g_filt, h_filt, i_filt,
                              bragg, even, boxes, i_dft, inverses, i_mask, 
                              i_unmask, acc_moves, acc_temps, rej_moves,
                              rej_temps, chi_sq, energy, temperature, scale,
                              constant, delta, fixed, T, p, nh, nk, nl,
                              nu, nv, nw, n_atm, n, N):
                                            
        refinement.displacive(Ux, Uy, Uz,
                              U_r, U_r_orig, U_r_cand,
                              U_k, U_k_orig, U_k_cand,
                              V_k, V_k_nuc, V_k_orig,
                              V_k_nuc_orig, V_k_cand, V_k_nuc_cand,
                              F, F_nuc, F_orig, F_nuc_orig, F_cand, F_nuc_cand,
                              prod, prod_nuc, prod_orig, prod_nuc_orig,    
                              prod_cand, prod_nuc_cand, space_factor, factors,
                              coeffs, Q_k, displacement,
                              I_calc, I_expt, inv_sigma_sq, 
                              I_raw, I_flat, I_ref, v_inv, 
                              a_filt, b_filt, c_filt,
                              d_filt, e_filt, f_filt,
                              g_filt, h_filt, i_filt,
                              bragg, even, boxes, i_dft, inverses, i_mask, 
                              i_unmask, acc_moves, acc_temps, rej_moves,
                              rej_temps, chi_sq, energy, temperature, scale,
                              constant, delta, fixed, T, p, nh, nk, nl,
                              nu, nv, nw, n_atm, n, N)
    
    def correlation_statistics(self, corr):
    
        runs = np.shape(corr)[0]
        
        return np.mean(corr, axis=0), np.std(corr, axis=0)**2/runs,
               
    def vector_correlations_1d(self, Vx, Vy, Vz, rx, ry, rz, atms, fract, tol,
                               A, nu, nv, nw, n_atm):

        corr, coll, \
        corr_, coll_, \
        d, atm_pair = correlations.radial(Vx, Vy, Vz, rx, ry, rz, atms, 
                                          fract=fract, tol=tol,
                                          period=(A, nu, nv, nw, n_atm))
        
        return corr, coll, d, atm_pair

    def scalar_correlations_1d(self, V_r, rx, ry, rz, atms, fract, tol,
                               A, nu, nv, nw, n_atm):

        corr, corr_, \
        d, atm_pair = correlations.parameter(V_r, rx, ry, rz, atms, 
                                             fract=fract, tol=tol,
                                             period=(A, nu, nv, nw, n_atm))
            
        return corr, d, atm_pair
                
    def vector_average_1d(self, corr, coll, sigma_sq_corr, 
                          sigma_sq_coll, d, tol):
    
        corr, corr_, \
        sigma_sq_corr, sigma_sq_coll, \
        d = crystal.average((corr, coll, sigma_sq_corr, sigma_sq_coll), d, tol)
        
        return corr, coll, sigma_sq_corr, sigma_sq_coll, d
                        
    def scalar_average_1d(self, corr, sigma_sq_corr, d, tol):
            
        corr, sigma_sq_corr, d = crystal.average((corr, sigma_sq_corr), d, tol)
        
        return corr, sigma_sq_corr, d
            
    def vector_correlations_3d(self, Ux, Uy, Uz, rx, ry, rz, atms, fract, tol,
                               A, nu, nv, nw, n_atm):  
        
        corr3d, coll3d, \
        corr3d_, coll3d_, \
        dx, dy, dz, \
        atm_pair3d = correlations.radial3d(Ux, Uy, Uz, rx, ry, rz, atms, 
                                           fract=fract, tol=tol,
                                           period=(A, nu, nv, nw, n_atm))
        
        return corr3d, coll3d, dx, dy, dz, atm_pair3d
               
    def scalar_correlations_3d(self, V_r, rx, ry, rz, atms, fract, tol,
                               A, nu, nv, nw, n_atm):
                            
        corr3d, corr3d_, \
        dx, dy, dz, \
        atm_pair3d = correlations.parameter3d(V_r, rx, ry, rz, atms,
                                              fract=fract, tol=tol,
                                              period=(A, nu, nv, nw, n_atm))    
                                        
        return corr3d, dx, dy, dz, atm_pair3d
       
    def vector_symmetrize_3d(self, corr3d, coll3d,
                             sigma_sq_corr3d, sigma_sq_coll3d, 
                             dx, dy, dz, atm_pair3d, A, laue, tol):
        
        corr3d, coll3d, \
        sigma_sq_corr3d, sigma_sq_coll3d, \
        dx, dy, dz, \
        atm_pair3d = crystal.symmetrize((corr3d, coll3d,
                                         sigma_sq_corr3d, sigma_sq_coll3d),
                                         dx, dy, dz, atm_pair3d, A, laue, tol)
        
        return corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d, \
               dx, dy, dz, atm_pair3d
        
    def scalar_symmetrizes_3d(self, corr3d, sigma_sq_corr3d, dx, dy, dz, 
                              atm_pair3d, A, laue, tol):

        corr3d, sigma_sq_corr3d, \
        dx, dy, dz, \
        atm_pair3d = crystal.symmetrize((corr3d, sigma_sq_corr3d),
                                         dx, dy, dz, atm_pair3d, A, laue, tol)
        
        return corr3d, sigma_sq_corr3d, dx, dy, dz, atm_pair3d
        
    def vector_average_3d(self, corr3d, coll3d, 
                          sigma_sq_corr3d, sigma_sq_coll3d, dx, dy, dz, tol):
        
        corr3d, coll3d, \
        sigma_sq_corr3d, sigma_sq_coll3d, \
        dx, dy, dz = crystal.average3d((corr3d, coll3d, 
                                        sigma_sq_corr3d, sigma_sq_coll3d),
                                        dx, dy, dz, tol=tol)

        return corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d, dx, dy, dz                            
        
    def scalar_average_3d(self, corr3d, sigma_sq_corr3d, dx, dy, dz, tol):

        corr3d, \
        sigma_sq_corr3d, \
        dx, dy, dz = crystal.average3d((corr3d, sigma_sq_corr3d),
                                        dx, dy, dz, tol=tol)
        
        return corr3d, sigma_sq_corr3d, dx, dy, dz
    
    def save_scalar_1d(self, fname, corr, sigma_sq_corr, d, atm_pair):
        
        np.save('{}-correlations-1d.npy'.format(fname), corr)
        np.save('{}-correlations-1d-error.npy'.format(fname), sigma_sq_corr)
        np.save('{}-correlations-1d-d.npy'.format(fname), d)
        np.save('{}-correlations-1d-pair.npy'.format(fname), atm_pair)
        
    def save_vector_1d(self, fname, corr, coll, sigma_sq_corr, sigma_sq_coll, 
                       d, atm_pair):
        
        np.save('{}-correlations-1d.npy'.format(fname), corr)
        np.save('{}-collinearity-1d.npy'.format(fname), coll)
        np.save('{}-correlations-1d-error.npy'.format(fname), sigma_sq_corr)
        np.save('{}-collinearity-1d-error.npy'.format(fname), sigma_sq_coll)
        np.save('{}-correlations-1d-d.npy'.format(fname), d)
        np.save('{}-correlations-1d-pair.npy'.format(fname), atm_pair)
        
    def save_scalar_3d(self, fname, corr, sigma_sq_corr, dx, dy, dz, atm_pair):
        
        np.save('{}-correlations-3d.npy'.format(fname), corr)
        np.save('{}-correlations-3d-error.npy'.format(fname), sigma_sq_corr)
        np.save('{}-correlations-3d-dx.npy'.format(fname), dx)
        np.save('{}-correlations-3d-dy.npy'.format(fname), dy)
        np.save('{}-correlations-3d-dz.npy'.format(fname), dz)
        np.save('{}-correlations-3d-pair.npy'.format(fname), atm_pair)
        
    def save_vector_3d(self, fname, corr, coll, sigma_sq_corr, sigma_sq_coll, 
                       dx, dy, dz, atm_pair):
        
        np.save('{}-correlations-3d.npy'.format(fname), corr)
        np.save('{}-collinearity-3d.npy'.format(fname), coll)
        np.save('{}-correlations-3d-error.npy'.format(fname), sigma_sq_corr)
        np.save('{}-collinearity-3d-error.npy'.format(fname), sigma_sq_coll)
        np.save('{}-correlations-3d-dx.npy'.format(fname), dx)
        np.save('{}-correlations-3d-dy.npy'.format(fname), dy)
        np.save('{}-correlations-3d-dz.npy'.format(fname), dz)
        np.save('{}-correlations-3d-pair.npy'.format(fname), atm_pair)
        
    def load_scalar_1d(self, fname):
        
        corr = np.load('{}-correlations-1d.npy'.format(fname))
        sigma_sq_corr = np.load('{}-correlations-1d-error.npy'.format(fname))
        d = np.load('{}-correlations-1d-d.npy'.format(fname))
        atm_pair = np.load('{}-correlations-1d-pair.npy'.format(fname))
        
        return corr, sigma_sq_corr, d, atm_pair
        
    def load_vector_1d(self, fname):
        
        corr = np.load('{}-correlations-1d.npy'.format(fname))
        coll = np.load('{}-collinearity-1d.npy'.format(fname))
        sigma_sq_corr = np.load('{}-correlations-1d-error.npy'.format(fname))
        sigma_sq_coll = np.load('{}-collinearity-1d-error.npy'.format(fname))
        d = np.load('{}-correlations-1d-d.npy'.format(fname))
        atm_pair = np.load('{}-correlations-1d-pair.npy'.format(fname))
        
        return corr, coll, sigma_sq_corr, sigma_sq_coll, d, atm_pair
        
    def load_scalar_3d(self, fname):
        
        corr = np.load('{}-correlations-3d.npy'.format(fname))
        sigma_sq_corr = np.load('{}-correlations-3d-error.npy'.format(fname))
        dx = np.load('{}-correlations-3d-dx.npy'.format(fname))
        dy = np.load('{}-correlations-3d-dy.npy'.format(fname))
        dz = np.load('{}-correlations-3d-dz.npy'.format(fname))
        atm_pair = np.load('{}-correlations-3d-pair.npy'.format(fname))
        
        return corr, sigma_sq_corr, dx, dy, dz, atm_pair
        
    def load_vector_3d(self, fname):
        
        corr = np.load('{}-correlations-3d.npy'.format(fname))
        coll = np.load('{}-collinearity-3d.npy'.format(fname))
        sigma_sq_corr = np.load('{}-correlations-3d-error.npy'.format(fname))
        sigma_sq_coll = np.load('{}-collinearity-3d-error.npy'.format(fname))
        dx = np.load('{}-correlations-3d-dx.npy'.format(fname))
        dy = np.load('{}-correlations-3d-dy.npy'.format(fname))
        dz = np.load('{}-correlations-3d-dz.npy'.format(fname))
        atm_pair = np.load('{}-correlations-3d-pair.npy'.format(fname))
        
        return corr, coll, sigma_sq_corr, sigma_sq_coll, dx, dy, dz, atm_pair
        