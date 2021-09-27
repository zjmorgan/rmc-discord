#!/usr/bin/env/python3

import os
import sys

import numpy as np

from disorder.material import crystal

class UnitCell:

    def __init__(self, filename, tol=1e-2):
        
        filename = os.path.abspath(filename)
                           
        folder = os.path.dirname(filename)
        filename = os.path.basename(filename)
        
        self.__folder = folder
        self.__filename = filename
        
        self. __load_unit_cell(tol)
        
    def __load_unit_cell(self, tol):
        
        folder = self.get_filepath()
        filename = self.get_filename()
        
        constants = crystal.parameters(folder=folder, filename=filename)
               
        a, b, c, alpha, beta, gamma = constants
        
        self.__a, self.__b, self.__c = a, b, c
        self.__alpha, self.__beta, self.__gamma = alpha, beta, gamma
        
        uc_dict = crystal.unitcell(folder=folder, filename=filename, tol=tol)  
        
        n_atm = uc_dict['n_atom']
       
        self.__atm = uc_dict['atom']
        self.__site = uc_dict['site']
        
        uni = np.unique(self.__site)
        self.__act = np.full(uni.size, True)
        
        self.__op = uc_dict['operator']
        self.__mag_op = uc_dict['magnetic_operator']
                
        self.__u = uc_dict['u']
        self.__v = uc_dict['v']
        self.__w = uc_dict['w']
                
        self.__occ = uc_dict['occupancy']
                
        displacement = uc_dict['displacement']
        
        self.__U11 = np.zeros(n_atm)
        self.__U22 = np.zeros(n_atm)
        self.__U33 = np.zeros(n_atm)
        self.__U23 = np.zeros(n_atm)
        self.__U13 = np.zeros(n_atm)
        self.__U12 = np.zeros(n_atm)
        
        if (len(displacement.shape) != 2):
            displacement = displacement.reshape(displacement.size, 1)
        
        if (displacement.shape[1] == 6):
            self.set_anisotropic_displacement_parameters(*displacement.T)
        else:
            self.set_isotropic_displacement_parameter(displacement.flatten())
            
        self.__mu1, self.__mu2, self.__mu3 = uc_dict['moment'].T
                
        self.__g = np.full(n_atm, 2.0)
        
        hm, sg = crystal.group(folder=folder, filename=filename)
        
        self.__hm = hm
        self.__sg = sg
                
        lat = crystal.lattice(*constants)
        
        self.__lat = lat
        
        laue =  crystal.laue(folder=folder, filename=filename)
        
        self.__laue = laue
        
    def __get_mask(self):
        
        active = self.get_active_sites()
        site = self.get_sites()
        
        mask = active[site]
        
        return mask
    
    def get_sites(self):
                
        return self.__site
        
    def get_active_sites(self):
        
        return self.__act
    
    def set_active_sites(self, act):
                
        self.__act = act
    
    def get_number_atoms_per_unit_cell(self):
                
        return self.__get_mask().sum()
            
    def get_filepath(self):
        
        return self.__folder
    
    def get_filename(self):
        
        return self.__filename
    
    def get_fractional_coordinates(self):
        
        mask = self.__get_mask()
        
        return self.__u[mask], self.__v[mask], self.__w[mask]
    
    def set_fractional_coordinates(self, u, v, w):
        
        mask = self.__get_mask()
        
        self.__u[mask], self.__v[mask], self.__w[mask] = u, v, w
        
    def get_unit_cell_cartesian_atomic_coordinates(self):
        
        A = self.get_fractional_cartesian_transform()
        u, v, w = self.get_fractional_coordinates()
                
        return crystal.transform(u, v, w, A)
    
    def get_unit_cell_atoms(self):
        
        mask = self.__get_mask()
        
        return self.__atm[mask]
    
    def set_unit_cell_atoms(self, atm):
        
        mask = self.__get_mask()
        
        self.__atm[mask] = atm

    def get_occupancies(self):
        
        mask = self.__get_mask()
        
        return self.__occ[mask]
    
    def set_occupancies(self, occ):
        
        mask = self.__get_mask()
        
        self.__occ[mask] = occ
        
    def get_anisotropic_displacement_parameters(self):
        
        mask = self.__get_mask()
        
        U11 = self.__U11[mask]
        U22 = self.__U22[mask]
        U33 = self.__U33[mask]
        U23 = self.__U23[mask]
        U13 = self.__U13[mask]
        U12 = self.__U12[mask]
        
        return U11, U22, U33, U23, U13, U12
    
    def set_anisotropic_displacement_parameters(self, U11, U22, U33, 
                                                      U23, U13, U12):
        
        mask = self.__get_mask()
        
        self.__U11[mask] = U11
        self.__U22[mask] = U22
        self.__U33[mask] = U33
        self.__U23[mask] = U23
        self.__U13[mask] = U13
        self.__U12[mask] = U12
        
    def set_isotropic_displacement_parameter(self, Uiso):
        
        mask = self.__get_mask()
        
        D = self.get_atomic_displacement_cartesian_transform()
    
        uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
        
        U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
        U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]
        
        self.__U11[mask] = U11
        self.__U22[mask] = U22
        self.__U33[mask] = U33
        self.__U23[mask] = U23
        self.__U13[mask] = U13
        self.__U12[mask] = U12
        
    def get_isotropic_displacement_parameter(self):
        
        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()
        
        U11, U22, U33, U23, U13, U12 = adps
        
        U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
        n = np.size(U11)
        
        U = U.reshape(3,3,n)
        
        Uiso = []
        for i in range(n):
            Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D.T))
            Uiso.append(np.mean(Up).real)
        
        return np.array(Uiso)
    
    def get_principal_displacement_parameters(self):
        
        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()
        
        U11, U22, U33, U23, U13, U12 = adps
        
        U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
        n = np.size(U11)
        
        U = U.reshape(3,3,n)
        
        U1, U2, U3 = [], [], []
        for i in range(n):
            Up, _ = np.linalg.eig(np.dot(np.dot(D, U[...,i]), D.T))
            Up.sort()
            U1.append(Up[0].real)
            U2.append(Up[1].real)
            U3.append(Up[2].real)
        
        return np.array(U1), np.array(U2), np.array(U3)
    
    def get_cartesian_anistropic_displacement_parameters(self):
        
        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()
        
        U11, U22, U33, U23, U13, U12 = adps
        
        U = np.array([[U11,U12,U13], [U12,U22,U23], [U13,U23,U33]])
        n = np.size(U11)
        
        U = U.reshape(3,3,n)
        
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []
        for i in range(n):
            Up = np.dot(np.dot(D, U[...,i]), D.T)
            Uxx.append(Up[0,0])
            Uyy.append(Up[1,1])
            Uzz.append(Up[2,2])
            Uyz.append(Up[1,2])
            Uxz.append(Up[0,2])
            Uxy.append(Up[0,1])
        
        return np.array(Uxx), np.array(Uyy), np.array(Uzz), \
               np.array(Uyz), np.array(Uxz), np.array(Uxy)
               
    def get_crystal_axis_magnetic_moments(self):
        
        mask = self.__get_mask()
        
        mu1 = self.__mu1[mask]
        mu2 = self.__mu2[mask]
        mu3 = self.__mu3[mask]
        
        return mu1, mu2, mu3
    
    def set_crystal_axis_magnetic_moments(self, mu1, mu2, mu3):
        
        mask = self.__get_mask()
        
        self.__mu1[mask] = mu1
        self.__mu2[mask] = mu2
        self.__mu2[mask] = mu3
               
    def get_magnetic_moment_magnitude(self):
        
        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()
        
        M = np.array([mu1,mu2,mu3])
        n = np.size(mu1)
        
        M = M.reshape(3,n)

        mu = []
        for i in range(n):
            mu.append(np.linalg.norm(np.dot(C, M[:,i])))
        
        return np.array(mu)
    
    def get_cartesian_magnetic_moments(self):
        
        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()
        
        M = np.array([mu1,mu2,mu3])
        n = np.size(mu1)
        
        M = M.reshape(3,n)

        mux, muy, muz = [], [], []
        for i in range(n):
            Mp = np.dot(C, M[:,i])
            mux.append(Mp[0])
            muy.append(Mp[1])
            muz.append(Mp[2])
        
        return np.array(mux), np.array(muy), np.array(muz)
    
    def get_g_factors(self):
        
        mask = self.__get_mask()
        
        return self.__g[mask]
    
    def set_g_factors(self, g):
        
        mask = self.__get_mask()
        
        self.__g = g[mask]

    def get_lattice_constants(self):
        
        constants = self.__a, self.__b, self.__c, \
                    self.__alpha, self.__beta, self.__gamma
        
        return constants
    
    def set_lattice_constants(self, *constants):
        
        lat = self.get_lattice_system()
        a, b, c, alpha, beta, gamma = self.get_lattice_constants()
        
        if (lat == 'Cubic'):
            a, = constants
            b = c = a
        elif (lat == 'Hexagonal' or lat == 'Tetragonal'):
            a, c = constants
            b = a
        elif (lat == 'Rhobmohedral'):
            a, alpha = constants
            b = c = a
            beta = gamma = alpha
        elif (lat == 'Orthorhombic'):
            a, b, c = constants
        elif (lat == 'Monoclinic'):
            if (not np.isclose(beta, np.pi/2)):
                a, b, c, alpha, gamma = constants
            else:
                a, b, c, alpha, beta = constants
        else:
            a, b, c, alpha, beta, gamma = constants
        
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
    
    def get_reciprocal_lattice_constants(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.reciprocal(*constants)
    
    def get_symmetry_operators(self):
        
        mask = self.__get_mask()
        
        return self.__op[mask]
    
    def get_magnetic_symmetry_operators(self):
        
        mask = self.__get_mask()
        
        return self.__mag_op[mask]
    
    def get_lattice_system(self):
        
        return self.__lat
    
    def get_lattice_volume(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.volume(*constants)
    
    def get_reciprocal_lattice_volume(self):
        
        constants = self.get_reciprocal_lattice_constants()
        
        return crystal.volume(*constants)
    
    def get_metric_tensor(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.metric(*constants)
    
    def get_reciprocal_metric_tensor(self):
        
        constants = self.get_reciprocal_lattice_constants()
        
        return crystal.metric(*constants)
    
    def get_fractional_cartesian_transform(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.cartesian(*constants)
    
    def get_miller_cartesian_transform(self):
        
        constants = self.get_reciprocal_lattice_constants()
        
        return crystal.cartesian(*constants)
    
    def get_cartesian_rotation(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.cartesian_rotation(*constants)
    
    def get_moment_cartesian_transform(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.cartesian_moment(*constants)
    
    def get_atomic_displacement_cartesian_transform(self):
        
        constants = self.get_lattice_constants()
        
        return crystal.cartesian_displacement(*constants)
    
    def get_space_group_symbol(self):
        
        return self.__hm
    
    def get_space_group_number(self):
        
        return self.__sg
    
    def get_laue(self):
        
        return self.__laue
    
class SuperCell(UnitCell):
    
    def __init__(self, filename, nu=1, nv=1, nw=1, tol=1e-2):
        
        super(SuperCell, self).__init__(filename, tol)
        
        self.set_super_cell_dimensions(nu, nv, nw)

    def get_super_cell_dimensions(self):
        
        return self.__nu, self.__nv, self.__nw
        
    def set_super_cell_dimensions(self, nu, nv, nw):
        
        self.__nu = nu
        self.__nv = nv
        self.__nw = nw
        
    def get_super_cell_size(self):
        
        nu, nv, nw = self.get_super_cell_dimensions()
        
        return nu*nv*nw
        
    def get_number_atoms_per_super_cell(self):
        
        n_uvw = self.get_super_cell_size()
        n_atm = self.get_number_atoms_per_unit_cell()
        
        return n_uvw*n_atm
    
    def get_cartesian_lattice_points(self):
        
        A = self.get_fractional_cartesian_transform()
        nu, nv, nw = self.get_super_cell_dimensions()
        
        return space.cell(nu, nv, nw, A)
    
    def get_super_cell_cartesian_atomic_coordinates(self):
        
        ux, uy, uz = self.get_unit_cell_cartesian_atomic_coordinates()
        ix, iy, iz = self.get_cartesian_lattice_points()
        atm = self.get_unit_cell_atoms()
                
        return space.real(ux, uy, uz, ix, iy, iz, atm)