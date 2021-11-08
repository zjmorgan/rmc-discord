#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import structure, crystal, symmetry

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_structure(unittest.TestCase):
    
    def test_factor(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
        
        names = ('h', 'k', 'l', 'd(angstrom)', 'F(real)', 'F(imag)', 'mult')
        formats = (int, int, int, float, float, float, int)
                                        
        uc_dict = crystal.unitcell(folder=folder, 
                                    filename='Cu3Au.cif', 
                                    tol=1e-4)
                
        u = uc_dict['u']
        v = uc_dict['v']
        w = uc_dict['w']
        occ = uc_dict['occupancy']
        disp = uc_dict['displacement']
        atm = uc_dict['atom']
        n_atm = uc_dict['n_atom']
        
        constants = crystal.parameters(folder=folder, filename='Cu3Au.cif')
                
        symops = crystal.operators(folder=folder, filename='Cu3Au.cif')
                
        a, b, c, alpha, beta, gamma = constants
                        
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)
        
        Uiso = 1/(8*np.pi**2)
        uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
        
        U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
        U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]
        
        h, k, l, d, F, mult = structure.factor(u, v, w, atm, occ, 
                                               U11, U22, U33, U23, U13, U12,
                                               a, b, c, alpha, beta, gamma, 
                                               symops, dmin=0.7, 
                                               source='Neutron')
        
        data = np.loadtxt(os.path.join(folder, 'Cu3Au.csv'),
                          dtype={'names': names, 'formats': formats},
                          delimiter=',', skiprows=1, unpack=True)
                        
        np.testing.assert_array_equal(h, data[0])
        np.testing.assert_array_equal(k, data[1])
        np.testing.assert_array_equal(l, data[2])
        np.testing.assert_array_equal(mult, data[6])
        
        np.testing.assert_array_almost_equal(d, data[3])
        np.testing.assert_array_almost_equal(F.real, data[4], decimal=4)
        np.testing.assert_array_almost_equal(F.imag, data[5], decimal=4)
        
        uc_dict = crystal.unitcell(folder=folder, 
                                   filename='CaTiOSiO4.cif', 
                                   tol=1e-4)
                
        u = uc_dict['u']
        v = uc_dict['v']
        w = uc_dict['w']
        occ = uc_dict['occupancy']
        disp = uc_dict['displacement']
        atm = uc_dict['atom']
        n_atm = uc_dict['n_atom']
                                
        constants = crystal.parameters(folder=folder, filename='CaTiOSiO4.cif')
                
        symops = crystal.operators(folder=folder, filename='CaTiOSiO4.cif')
                
        a, b, c, alpha, beta, gamma = constants
                        
        U11, U22, U33, U23, U13, U12 = disp.T
                                
        h, k, l, d, F, mult = structure.factor(u, v, w, atm, occ, 
                                               U11, U22, U33, U23, U13, U12,
                                               a, b, c, alpha, beta, gamma, 
                                               symops, dmin=0.7, 
                                               source='Neutron')
        
        data = np.loadtxt(os.path.join(folder, 'CaTiOSiO4.csv'),
                          dtype={'names': names, 'formats': formats},
                          delimiter=',', skiprows=1, unpack=True)
                        
        np.testing.assert_array_equal(h, data[0])
        np.testing.assert_array_equal(k, data[1])
        np.testing.assert_array_equal(l, data[2])
        np.testing.assert_array_equal(mult, data[6])
        
        np.testing.assert_array_almost_equal(d, data[3])
        np.testing.assert_array_almost_equal(F.real, data[4], decimal=4)
        np.testing.assert_array_almost_equal(F.imag, data[5], decimal=4)
                
if __name__ == '__main__':
    unittest.main()