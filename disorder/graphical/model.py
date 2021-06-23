#!/usr/bin/env/python3

import re

import numpy as np

from disorder.material import crystal, tables

class Model:

    def __init__(self):
        pass

    def supercell_size(self, n_atm, nu, nv, nw):
        return n_atm*nu*nv*nw
    
    def iso_symbols(self, keys):
        return np.array([re.sub(r'[\d.+-]+$', '', key) for key in keys])

    def ion_symbols(self, keys):
        return np.array([re.sub(r'^\d+\s*', '', key) for key in keys])
    
    def remove_symbols(self, keys):
        return np.array([re.sub(r'[a-zA-Z]', '', key) for key in keys])

    def sort_keys(self, col0, col1, keys):
        keys = np.array([key for key in keys])
        sort = np.lexsort(np.array((col0, col1)))        
        return keys[sort]
    
    def get_neutron_scattering_lengths(self):
        bc_keys = tables.bc.keys()
        bc_atm = self.ion_symbols(bc_keys)
        bc_nuc = self.remove_symbols(bc_keys)
        return self.sort_keys(bc_nuc,bc_atm,bc_keys)
 
    def get_xray_form_factors(self):
        X_keys = tables.X.keys()
        X_atm = self.iso_symbols(X_keys)
        X_ion = self.remove_symbols(X_keys)
        return self.sort_keys(X_ion,X_atm,X_keys)

    def get_magnetic_form_factors(self):
        j0_keys = tables.j0.keys()
        j0_atm = self.iso_symbols(j0_keys)
        j0_ion = self.remove_symbols(j0_keys)
        return self.sort_keys(j0_ion,j0_atm,j0_keys) 
    
    def load_unit_cell(self, folder, filename):
    
        u, \
        v, \
        w, \
        occupancy, \
        displacement, \
        moment, \
        site, \
        atms, \
        n_atm = crystal.unitcell(folder=folder, 
                                 filename=filename,
                                 occupancy=True,
                                 displacement=True,
                                 moment=True,
                                 site=True)
        
        return u, v, w, occupancy, displacement, moment, site, atms, n_atm
    
    def load_space_group(self, folder, filename):
        
        group, hm = crystal.group(folder=folder, filename=filename)
        
        return group, hm
    
    def load_lattice_parameters(self, folder, filename):
        
        a, \
        b, \
        c, \
        alpha, \
        beta, \
        gamma = crystal.parameters(folder=folder, filename=filename)
        
        return a, b, c, alpha, beta, gamma
    
    def find_lattice(self, a, b, c, alpha, beta, gamma):
    
        return crystal.lattice(a, b, c, alpha, beta, gamma)
    
    def crystal_matrices(self, a, b, c, alpha, beta, gamma):
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
    
        D = crystal.orthogonalized(A, a, b, c, alpha, beta, gamma)
        
        return A, B, R, D
    
    def atomic_displacement_parameters(self, U11, U22, U33, U23, U13, U12, D):

        U = np.array([[U11,U12,U13], [U12,U22,U13], [U13,U23,U33]])
        n = np.size(U11)
        
        U = U.reshape(3,3,n)
                
        D_inv = np.linalg.inv(D)
        
        Uiso = []
        for i in range(n):
            Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D_inv))
            Uiso.append(np.mean(Up).real)
        
        return np.array(Uiso)
    
    def magnetic_moments(self, mu1, mu2, mu3, D):
        
        M = np.array([mu1,mu2,mu3])
        n = np.size(mu1)
        
        M = M.reshape(3,n)

        mu = []
        for i in range(n):
            mu.append(np.linalg.norm(np.dot(D, M[:,i])))
        
        return np.array(mu)
    
    