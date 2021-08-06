#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.graphical.model import Model

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_model(unittest.TestCase):      
    
    def setUp(self):
        self.model = Model()   
        
    def test_supercell_size(self):
        
        self.assertEqual(self.model.supercell_size(2,3,5,7), 2*3*5*7)
        
    def test_ion_symbols(self):

        atms = ['Fe3+', '54Fe', 'H', 'Mn3+', '55Mn3+']
        np.testing.assert_array_equal(self.model.ion_symbols(atms), 
                                      ['Fe', '54Fe', 'H', 'Mn', '55Mn'])
        
    def test_iso_symbols(self):

        atms = ['Fe3+', '54Fe', 'H', 'Mn3+', '55Mn3+']
        np.testing.assert_array_equal(self.model.iso_symbols(atms), 
                                      ['Fe3+', 'Fe', 'H', 'Mn3+', 'Mn3+'])

    def test_remove_symbols(self):

        atms = ['Fe3+', '54Fe', 'H', 'Mn3+', '55Mn3+']
        np.testing.assert_array_equal(self.model.remove_symbols(atms), 
                                      ['3+', '54', '', '3+', '553+'])
        
    def test_sort_keys(self):

        atms = ['Mn3+', '55Mn3+', 'H', 'Fe3+', '54Fe']
        
        ions = self.model.iso_symbols(atms)
        ion_type = self.model.remove_symbols(ions)
        ion_symbols = self.model.ion_symbols(ions)
                
        isotopes = self.model.ion_symbols(atms)
        iso_type = self.model.remove_symbols(isotopes)
        iso_symbols = self.model.iso_symbols(isotopes)
        
        ion_keys = self.model.sort_keys(ion_type, ion_symbols, atms)
        np.testing.assert_array_equal(ion_keys, 
                                      ['54Fe', 'Fe3+', 'H', 'Mn3+', '55Mn3+'])
                
        iso_keys = self.model.sort_keys(iso_type, iso_symbols, atms)
        np.testing.assert_array_equal(iso_keys, 
                                      ['Fe3+', '54Fe', 'H', 'Mn3+', '55Mn3+'])
        
    def test_get_neutron_scattering_length_keys(self):
        
        bc_keys = self.model.get_neutron_scattering_length_keys()
        
        self.assertEqual(bc_keys[0], 'Ac')
        self.assertEqual(bc_keys[-1], '96Zr')
        
    def test_get_xray_form_factor_keys(self):
        
        X_keys = self.model.get_xray_form_factor_keys()
        
        self.assertEqual(X_keys[0], 'Ac')
        self.assertEqual(X_keys[-1], 'Zr4+')
        
    def test_get_magnetic_form_factor_keys(self):
        
        j_keys = self.model.get_magnetic_form_factor_keys()
        
        self.assertEqual(j_keys[0], 'Am2+')
        self.assertEqual(j_keys[-1], 'Zr1+')
        
    def test_load_unit_cell(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
        
        data = self.model.load_unit_cell(folder, 'chlorastrolite.cif')
        self.assertEqual(data[-1], 104)
        
    def test_load_space_group(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
        
        sg, hm = self.model.load_space_group(folder, 'chlorastrolite.cif')
        
        self.assertEqual(sg, 12)
        self.assertEqual(hm, 'A12/m1')
        
    def test_load_lattice_parameters(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        constants = self.model.load_lattice_parameters(folder, 
                                                       'chlorastrolite.cif')
        
        a, b, c, alpha, beta, gamma = constants
        
        self.assertAlmostEqual(a, 8.8192)
        self.assertAlmostEqual(b, 5.9192)
        self.assertAlmostEqual(c, 19.1274)
        self.assertAlmostEqual(alpha, np.deg2rad(90))
        self.assertAlmostEqual(beta, np.deg2rad(97.446))
        self.assertAlmostEqual(gamma, np.deg2rad(90))
        
    def test_find_lattice(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        constants = self.model.load_lattice_parameters(folder, 
                                                       'chlorastrolite.cif')
        
        self.assertEqual(self.model.find_lattice(*constants), 'Monoclinic')
        
    def test_crystal_matrices(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        constants = self.model.load_lattice_parameters(folder, 'H2O.cif')
        
        a, b, c, alpha, beta, gamma = constants
        
        A, B, R, C, D = self.model.crystal_matrices(*constants)
        
        u, v, w = np.dot(A, [1,0,0]), np.dot(A, [0,1,0]), np.dot(A, [0,0,1])
        
        V = np.dot(u, np.cross(v, w))
        
        a_ = np.linalg.norm(np.cross(v, w)/V)
        b_ = np.linalg.norm(np.cross(w, u)/V)
        c_ = np.linalg.norm(np.cross(u, v)/V)
        
        np.testing.assert_array_almost_equal(np.dot(A.T,A), 
                                             np.linalg.inv(np.dot(B.T,B)))
          
        np.testing.assert_array_almost_equal(np.dot(R.T,R), np.eye(3))
        
        np.testing.assert_array_almost_equal(np.dot(np.linalg.inv(C),D), 
                                             np.diag((a*a_,b*b_,c*c_)))
        
    def test_atomic_displacement_parameters(self):
        
        a, b, c, alpha, beta, gamma = 5, 5, 7, np.pi/2, np.pi/2, 2*np.pi/3
        
        A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                    alpha, beta, gamma)
        
        U11 = np.array([0.0,2.3])
        U22 = np.array([0.0,2.3])
        U33 = np.array([0.0,2.3])
        U23 = np.array([1.0,0.0])
        U13 = np.array([-1.0,0.0])
        U12 = np.array([0.0,0.0])
        
        Uiso, U1, U2, U3 = self.model.atomic_displacement_parameters(U11, 
                                                                     U22, 
                                                                     U33, 
                                                                     U23, 
                                                                     U13, 
                                                                     U12, 
                                                                     D)
        
        u = np.dot(D, D.T)
        
        U = np.trace(u)/3
        
        np.testing.assert_array_almost_equal(Uiso, np.array([0.0, 2.3*U]))

        V, _ = np.linalg.eig(u)
        V.sort()
        
        np.testing.assert_array_almost_equal(U1, np.array([-2.0, 2.3*V[0]]))
        np.testing.assert_array_almost_equal(U2, np.array([0.0, 2.3*V[1]]))
        np.testing.assert_array_almost_equal(U3, np.array([2.0, 2.3*V[2]]))
        
    def test_magnetic_moments(self):
        
        a, b, c, alpha, beta, gamma = 5, 5, 7, np.pi/2, np.pi/2, 2*np.pi/3
        
        A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                    alpha, beta, gamma)
        
        mu1 = np.array([0.0,0.4,1.6,0.0])
        mu2 = np.array([0.7,0.0,1.6,0.0])
        mu3 = np.array([0.0,0.0,0.0,2.3])
        
        mu = self.model.magnetic_moments(mu1, mu2, mu3, C)
        
        np.testing.assert_array_almost_equal(mu, np.array([0.7,0.4,1.6,2.3]))

    def test_magnetic_symmetry(self):
        
        operator = '-mx,my,-mz'
        moment = [0.27,0.47,-0.31]
        
        transformed_moment = self.model.magnetic_symmetry(operator, moment)
        
        np.testing.assert_array_almost_equal(transformed_moment, 
                                             np.array([-0.27,0.47,0.31]))
        
    def test_symmetry(self):
        
        operator = 'x-y,x-1/4,z+3/4'
        coordinate = [0.24,0.24,0.26]
        
        transformed = self.model.symmetry(operator, coordinate)
        
        np.testing.assert_array_almost_equal(transformed, 
                                             np.array([0.0,0.99,0.01]))
        
    def test_reverse_symmetry(self):
        
        operator = 'x-y,x-1/4,z+3/4'
        coordinate = [0.24,0.24,0.26]
        
        transformed = self.model.symmetry(operator, coordinate)
        transformed = self.model.reverse_symmetry(operator, transformed)
        
        np.testing.assert_array_almost_equal(transformed, coordinate)
        
    def test_slice_value(self):
        
        self.assertAlmostEqual(self.model.slice_value(-2, 1, 31, 15), -0.5)
        self.assertAlmostEqual(self.model.slice_value(-2, 1, 31, -1), -2)
        self.assertAlmostEqual(self.model.slice_value(-2, 1, 31, 32), 1)

    def test_slice_index(self):
        
        self.assertAlmostEqual(self.model.slice_index(-2, 1, 31, -0.5), 15)
        self.assertAlmostEqual(self.model.slice_index(-2, 1, 31, -3), 0)
        self.assertAlmostEqual(self.model.slice_index(-2, 1, 31, 2), 30)

if __name__ == '__main__':
    unittest.main()
