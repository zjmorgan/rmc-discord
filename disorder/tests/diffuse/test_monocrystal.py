#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import tables, crystal, symmetry
from disorder.diffuse import space, scattering, monocrystal
from disorder.diffuse import magnetic, occupational

class test_monocrystal(unittest.TestCase):
    
    def test_magnetic(self):
                
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-1,1], 5
        k_range, nk = [0,2], 11
        l_range, nl = [-1,0], 5
        
        nu, nv, nw, n_atm = 2, 5, 4, 2
        
        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])
        
        atm = np.array(['Fe3+','Mn3+'])
        occupancy = np.array([0.75,0.5])
        g = np.array([2.,2.])
        
        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])
        
        twins = np.eye(3).reshape(1,3,3)
        variants = np.array([1.0])
        W = np.eye(3)
                            
        T = space.debye_waller(h_range, k_range, l_range, nh, nk, nl, 
                               U11, U22, U33, U23, U13, U12, a_, b_, c_)
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)
        
        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)
        
        index_parameters = space.mapping(h_range, k_range, l_range,
                                         nh, nk, nl, nu, nv, nw)
         
        h, k, l, H, K, L, indices, inverses, operators = index_parameters
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
        
        form_factor = magnetic.form(Q, atm, g=g)
        
        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, Sy, Sz, H, K, L, 
                                                     nu, nv, nw, n_atm)
        
        factors = space.prefactors(form_factor, phase_factor, occupancy)
        
        factors *= T
        
        I_ref = magnetic.intensity(Qx_norm, Qy_norm, Qz_norm,
                                   Sx_k, Sy_k, Sz_k, i_dft, factors)
        
        reduced_params = space.reduced(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw)
        
        indices, reverses, symops, Nu, Nv, Nw = reduced_params
        
        symop = symmetry.symmetry_id(symops)
                
        I = monocrystal.magnetic(Sx, Sy, Sz, occupancy, 
                                 U11, U22, U33, U23, U13, U12, ux, uy, uz, atm,
                                 h_range, k_range, l_range, indices, symop, 
                                 W, B, R, D, twins, variants, nh, nk, nl, 
                                 nu, nv, nw, Nu, Nv, Nw, g)
           
        np.testing.assert_array_almost_equal(I, I_ref)
        
    def test_occupational(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-1,1], 5
        k_range, nk = [0,2], 11
        l_range, nl = [-1,0], 5
        
        nu, nv, nw, n_atm = 2, 5, 4, 2
        
        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])
        
        atm = np.array(['Fe','Mn'])
        occupancy = np.array([0.75,0.5])
        
        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])
        
        twins = np.eye(3).reshape(1,3,3)
        variants = np.array([1.0])
        W = np.eye(3)
                            
        T = space.debye_waller(h_range, k_range, l_range, nh, nk, nl, 
                               U11, U22, U33, U23, U13, U12, a_, b_, c_)
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)
        
        A_r = occupational.composition(nu, nv, nw, n_atm, value=occupancy)
        
        index_parameters = space.mapping(h_range, k_range, l_range,
                                         nh, nk, nl, nu, nv, nw)
         
        h, k, l, H, K, L, indices, inverses, operators = index_parameters
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
        
        scattering_length = scattering.length(atm, Q.size)
        
        A_k, i_dft = occupational.transform(A_r, H, K, L, nu, nv, nw, n_atm)
        
        factors = space.prefactors(scattering_length, phase_factor, occupancy)
        
        factors *= T
        
        I_ref = occupational.intensity(A_k, i_dft, factors)
        
        reduced_params = space.reduced(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw)
        
        indices, reverses, symops, Nu, Nv, Nw = reduced_params
        
        symop = symmetry.symmetry_id(symops)
                
        I = monocrystal.occupational(A_r, occupancy,
                                     U11, U22, U33, U23, U13, U12, ux, uy, uz, 
                                     atm, h_range, k_range, l_range, indices, 
                                     symop, W, B, R, D, twins, variants, 
                                     nh, nk, nl, nu, nv, nw, Nu, Nv, Nw)
           
        np.testing.assert_array_almost_equal(I, I_ref)

if __name__ == '__main__':
    unittest.main()