#!/usr/bin/env python

import pytest
import unittest

from disorder.material.structure import SuperCell
from disorder.graphical.visualization import CrystalStructure

import os
directory = os.path.dirname(os.path.abspath(__file__))

import pstats, cProfile

class test_crystal_structure(unittest.TestCase):

    def setUp(self):

        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):

        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()

    @pytest.mark.skip(reason='Should be done offscreen')
    def test_standard_sample(self):

        folder = '../data/'
        sample = 'Yb3Al5O12'

        n1, n2, n3 = 1, 1, 1

        sc = SuperCell(os.path.join(folder, sample+'.cif'), n1, n2, n3)

        xyz = sc.get_unit_cell_cartesian_atomic_coordinates()
        occ = sc.get_occupancies()

        A = sc.get_fractional_cartesian_transform()

        atms = sc.get_unit_cell_atoms()
        colors = sc.get_atom_colors()
        radii = sc.get_atom_radii()

        cs = CrystalStructure(A, *xyz, atms, colors, 1, 1, 1)
        cs.draw_cell_edges()
        cs.draw_basis_vectors()
        cs.atomic_radii(radii, occ)
        cs.view_direction(3, 2, 1)

        cs.save_figure(os.path.join(directory, sample+'.pdf'))

if __name__ == '__main__':
    unittest.main()