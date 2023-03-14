#!/usr/bin/env python3

import unittest

import numpy as np
np.random.seed(13)

from scipy.spatial.transform import Rotation

from disorder.graphical.visualization import CrystalStructure

from disorder.material import crystal
from disorder.material import tables

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_visualization(unittest.TestCase):

    def test_CrystalStructure(self):

        a, b, c = 5, 5, 12
        alpha, beta, gamma = np.deg2rad(90), np.deg2rad(90), np.deg2rad(120)

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        atms = np.array(['Co', 'Fe', 'Mn'])

        n_atm = atms.shape[0]

        occ = np.ones(n_atm)

        u = np.array([0.5,0.0,0.0])
        v = np.array([0.0,0.5,0.0])
        w = np.array([0.5,0.5,0.0])

        ux, uy, uz = np.dot(A, [u,v,w])

        radii = np.array([tables.r.get(atm)[0] for atm in atms])

        Sx, Sy, Sz = [], [], []
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []

        rots = Rotation.random(n_atm).as_matrix()

        ae = 2*np.random.random(n_atm)+1
        be = 2*np.random.random(n_atm)+1
        ce = 2*np.random.random(n_atm)+1

        for i in range(n_atm):

            S = np.dot(rots[i], np.array([2,2,2]))
            Sx.append(S[0])
            Sy.append(S[1])
            Sz.append(S[2])

            U = np.dot(np.dot(rots[i],np.diag([ae[i],be[i],ce[i]])),rots[i].T)
            Uxx.append(U[0,0])
            Uyy.append(U[1,1])
            Uzz.append(U[2,2])
            Uyz.append(U[1,2])
            Uxz.append(U[0,2])
            Uxy.append(U[0,1])

        Sx = np.array(Sx)
        Sy = np.array(Sy)
        Sz = np.array(Sz)

        Uxx = np.array(Uxx)
        Uyy = np.array(Uyy)
        Uzz = np.array(Uzz)
        Uyz = np.array(Uyz)
        Uxz = np.array(Uxz)
        Uxy = np.array(Uxy)

        colors = [tables.rgb.get(atm) for atm in atms]

        filename = 'test_figure.svg'

        cs = CrystalStructure(A, ux, uy, uz, atms, colors)
        cs.draw_cell_edges()
        cs.view_direction(1, 1, 1)
        cs.draw_basis_vectors()
        cs.atomic_radii(radii, occ)
        cs.atomic_displacement_ellipsoids(Uxx, Uyy, Uzz,
                                          Uyz, Uxz, Uxy)
        cs.magnetic_vectors(Sx, Sy, Sz)
        cs.save_figure(filename)
        cs.show_figure()

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
