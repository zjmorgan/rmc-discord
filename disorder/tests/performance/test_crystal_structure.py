#!/usr/bin/env python

import pytest
import unittest

from disorder.material.structure import SuperCell
from disorder.graphical.visualization import CrystalStructure

import os
directory = os.path.dirname(os.path.abspath(__file__))

import pstats, cProfile

import numpy as np

class test_crystal_structure(unittest.TestCase):

    def setUp(self):

        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):

        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('time')
        #p.print_stats()

    @pytest.mark.skip(reason='Should be done offscreen')
    def test_standard_sample(self):

        folder = '../data/'
        sample = 'bixbyite'

        n1, n2, n3 = 2, 2, 2

        sc = SuperCell(os.path.join(folder, sample+'.cif'), n1, n2, n3)

        A = sc.get_fractional_cartesian_transform()

        sites = sc.get_atom_sites()

        sc.set_active_sites(sites == 'Mn')

        atms = sc.get_unit_cell_atoms()
        colors = sc.get_atom_colors()
        radii = sc.get_atom_radii()

        mux, muy, muz = np.zeros(atms.size), np.zeros(atms.size), np.zeros(atms.size)

        muz[atms != 'O'] = 1

        sc.set_crystal_axis_magnetic_moments(mux,muy,muz)

        sc.randomize_magnetic_moments()

        *xyz, _ = sc.get_super_cell_cartesian_atomic_coordinates()
        occ = sc.get_super_cell_occupancies()
        sx, sy, sz = sc.get_super_cell_cartesian_magnetic_moments()
        #U = sc.get_anisotropic_displacement_parameters()

        cs = CrystalStructure(A, *xyz, atms, colors, n1, n2, n3)
        cs.draw_cell_edges()
        cs.draw_basis_vectors()
        cs.atomic_radii(radii, occ[0])
        #cs.atomic_displacement_ellipsoids(*U)
        cs.magnetic_vectors(sx[0], sy[0], sz[0])
        cs.view_direction(3, 2, 1)

        cs.save_figure(os.path.join(directory, sample+'.pdf'))

if __name__ == '__main__':
    unittest.main()