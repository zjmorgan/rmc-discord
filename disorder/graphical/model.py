#!/usr/bin/env/python3

import re

import numpy as np

from disorder.diffuse import experimental, space, scattering, magnetic
from disorder.material import crystal, symmetry, tables

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
        
        nucleus = nucleus[nucleus == '-'] = ''
        return np.array([pre+atm for atm, pre in zip(element, nucleus)])
    
    def get_ion(self, element, charge):
        
        charge = charge[charge == '-'] = ''
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
    
    def data(self, filename):
        
        signal, sigma_sq, \
        h_range, k_range, l_range, \
        nh, nk, nl = experimental.data(filename)        
        
        return signal, sigma_sq, h_range, k_range, l_range, nh, nk, nl
    
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
            
    def matrix_transform(self, T, layer='l'):
    
        M = np.eye(3)
        
        if (layer == 'h'):
            Q = T[1:3,1:3].copy()
        elif (layer == 'k'):
            Q = T[0:3:2,0:3:2].copy()
        elif (layer == 'l'):
            Q = T[0:2,0:2].copy()     
                       
        Q /= Q[1,1]
        
        scale = 1/Q[0,0]
        Q[0,:] *= scale
        
        M[0:2,0:2] = Q
        
        return M, scale
    
    def mask_array(self, array):
        
        return np.ma.masked_less_equal(np.ma.masked_invalid(array, 
                                                            copy=False), 
                                       0, copy=False)
        
    def crop(self, array, h_slice, k_slice, l_slice):
        
        return experimental.crop(array, h_slice, k_slice, l_slice)
    
    def rebin(self, array, binsize):
        
        return experimental.rebin(array, binsize)
    
    def crop_parameters(self, xmin, xmax, minimum, maximum, size):
        
        binning = np.linspace(minimum, maximum, size)
        i_min = np.where(binning <= xmin)[0][-1]
        i_max = np.where(binning >= xmax)[0][-1]
        
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
    
            
        # self.ux, self.uy, self.uz = crystal.transform(self.u, 
        #                                               self.v, 
        #                                               self.w, 
        #                                               self.A)
        
        # self.nh = self.mask.shape[0]
        # self.nk = self.mask.shape[1]
        # self.nl = self.mask.shape[2]
        
        # self.I_obs = np.full((self.nh, self.nk, self.nl), np.nan)
        
        # t = 0
        # T = np.array([[np.cos(t), -np.sin(t), 0],
        #               [np.sin(t),  np.cos(t), 0],
        #               [0,          0,         1]])
    
        # min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        # max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        # min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        # max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        # min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        # max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                
        # h_range = [min_h, max_h]
        # k_range = [min_k, max_k]
        # l_range = [min_l, max_l]
        
    def reciprocal_mapping(self, h_range, k_range, l_range, nu, nv, nw, mask):
        
        nh, nk, nl = mask.shape
        
        h, k, l, \
        H, K, L, \
        indices, inverses, \
        operators = crystal.bragg(h_range, k_range, l_range, 
                                  nh, nk, nl, nu, nv, nw)
        
        i_mask, i_unmask = space.indices(inverses, mask)
            
        return h, k, l, indices, inverses, i_mask, i_unmask 
    
    def reciprocal_space_coordinate_transform(self, h, k, l, B, R):
            
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        return Qx, Qy, Qx, Qx_norm, Qy_norm, Qz_norm, Q 
    
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
        
        magnetic_factors = magnetic_form_factor*phase_factor*occupancy
        
        return factors, magnetic_factors
    
    def xray_factors(self, Q, ion, occupancy, phase_factor):
            
        form_factor = scattering.form(ion, Q)
        
        factors = space.prefactors(form_factor, phase_factor, occupancy)

        return factors
        