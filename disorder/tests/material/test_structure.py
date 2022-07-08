#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import structure, crystal

import os
directory = os.path.dirname(os.path.abspath(__file__))

# import pstats, cProfile

class test_structure(unittest.TestCase):

    # def setUp(self):

    #     self.pr = cProfile.Profile()
    #     self.pr.enable()

    # def tearDown(self):

    #     p = pstats.Stats(self.pr)
    #     p.strip_dirs()
    #     p.sort_stats('cumtime')
    #     p.print_stats()

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
                                                source='neutron')

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

        constants = crystal.parameters(folder=folder, filename='CaTiOSiO4.cif')

        symops = crystal.operators(folder=folder, filename='CaTiOSiO4.cif')

        a, b, c, alpha, beta, gamma = constants

        U11, U22, U33, U23, U13, U12 = disp.T

        h, k, l, d, F, mult = structure.factor(u, v, w, atm, occ,
                                               U11, U22, U33, U23, U13, U12,
                                               a, b, c, alpha, beta, gamma,
                                               symops, dmin=0.7,
                                               source='neutron')

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

    def test_UnitCell(self):

        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        cif_file = 'CaTiOSiO4.cif'

        filename = os.path.join(folder, cif_file)

        uc = structure.UnitCell(filename, tol=1e-4)

        self.assertEqual(uc.get_filepath(), folder)
        self.assertEqual(uc.get_filename(), cif_file)

        self.assertTrue(uc.get_active_sites().all())
        self.assertEqual(uc.get_number_atoms_per_unit_cell(), 32)

        active_sites = uc.get_active_sites()
        atms = uc.get_unit_cell_atoms()

        active_sites[atms == 'O'] = False
        uc.set_active_sites(active_sites)

        u, v, w = uc.get_fractional_coordinates()
        uc.set_fractional_coordinates(u, v, w)

        u_ref, v_ref, w_ref = uc.get_fractional_coordinates()
        np.testing.assert_array_almost_equal(u, u_ref)
        np.testing.assert_array_almost_equal(v, v_ref)
        np.testing.assert_array_almost_equal(w, w_ref)

        occ = uc.get_occupancies()
        uc.set_occupancies(occ)

        occ_ref = uc.get_occupancies()
        np.testing.assert_array_almost_equal(occ, occ_ref)

        disp_params = uc.get_anisotropic_displacement_parameters()
        uc.set_anisotropic_displacement_parameters(*disp_params)

        disp_params_ref = uc.get_anisotropic_displacement_parameters()
        np.testing.assert_array_almost_equal(disp_params, disp_params_ref)

        constants = uc.get_lattice_constants()
        uc.set_lattice_constants(*constants)

        constants_ref = uc.get_lattice_constants()
        np.testing.assert_array_almost_equal(constants, constants_ref)

        # ---

        cif_file = 'Li2Co(SO4)2.mcif'

        filename = os.path.join(folder, cif_file)

        uc = structure.UnitCell(filename, tol=1e-4)

        self.assertEqual(uc.get_filepath(), folder)
        self.assertEqual(uc.get_filename(), cif_file)

        self.assertTrue(uc.get_active_sites().all())
        self.assertEqual(uc.get_number_atoms_per_unit_cell(), 104)

        active_sites = uc.get_active_sites()
        atms = uc.get_unit_cell_atoms()

        active_sites[atms == 'O'] = False
        uc.set_active_sites(active_sites)

        mu1, mu2, mu3 = uc.get_crystal_axis_magnetic_moments()
        uc.set_crystal_axis_magnetic_moments(mu1, mu2, mu3)

        mu1_ref, mu2_ref, mu3_ref = uc.get_crystal_axis_magnetic_moments()
        np.testing.assert_array_almost_equal(mu1, mu1_ref)
        np.testing.assert_array_almost_equal(mu2, mu2_ref)
        np.testing.assert_array_almost_equal(mu3, mu3_ref)

        g = uc.get_g_factors()
        uc.set_g_factors(g)

        g_ref = uc.get_g_factors()
        np.testing.assert_array_almost_equal(g, g_ref)

if __name__ == '__main__':
    unittest.main()