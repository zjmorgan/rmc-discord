#!/usr/bin/env/python3

import re

import numpy as np

from disorder.diffuse import experimental
from disorder.material import crystal, symmetry, tables

class Model:

    def __init__(self):
        pass

    def supercell_size(self, n_atm, nu, nv, nw):
        
        return n_atm*nu*nv*nw
    
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
    
        return np.ma.masked_invalid(array, copy=False)
    
    def crop(self, array, h_slice, k_slice, l_slice):
        
        return experimental.crop(array, h_slice, k_slice, l_slice)
    
    def rebin(self, array, binsize):
        
        return experimental.rebin(array, binsize)