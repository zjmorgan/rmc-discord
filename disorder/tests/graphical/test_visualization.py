#!/usr/bin/env python3

import pytest
import unittest

import numpy as np
np.random.seed(13)

from scipy.spatial.transform import Rotation

from disorder.graphical.visualization import CrystalStructure

from disorder.material import crystal
from disorder.material import tables

import os
directory = os.path.dirname(os.path.abspath(__file__))

os.environ['ETS_TOOLKIT'] = 'null'
os.environ['QT_API'] = 'pyqt5'

from mayavi import mlab

class test_visualization(unittest.TestCase):

    @pytest.mark.skip(reason='Issue with build server')
    def test_CrystalStructure(self):

        a, b, c = 5, 5, 12
        alpha, beta, gamma = np.deg2rad(90), np.deg2rad(90), np.deg2rad(120)

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        atms = np.array(['Co', 'Fe', 'Mn'])

        n_atm = atms.shape[0]

        u = np.array([0.5,0.0,0.0])
        v = np.array([0.0,0.5,0.0])
        w = np.array([0.5,0.5,0.0])

        ux, uy, uz = np.dot(A, [u,v,w])

        radii = np.array([tables.r.get(atm)[0] for atm in atms])

        Sx, Sy, Sz = [], [], []
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []

        rots = Rotation.random(n_atm).as_matrix()

        ae = np.random.random(n_atm)
        be = np.random.random(n_atm)
        ce = np.random.random(n_atm)

        for i in range(n_atm):

            S = np.dot(rots[i], np.array([1,1,1]))
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

        uc = np.array([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0])
        vc = np.array([0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
        wc = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0])

        ucx, ucy, ucz = np.dot(A, [uc,vc,wc])

        filename = 'test_figure.png'

        mlab.options.offscreen = True

        cs = CrystalStructure()
        cs.draw_cell_edges(ucx, ucy, ucz)
        cs.view_direction(A, 1,1,1)
        cs.draw_basis_vectors(A,a,b,c)
        cs.atomic_radii(ux, uy, uz, radii, colors)
        cs.atomic_displacement_ellipsoids(ux, uy, uz, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy, colors)
        cs.magnetic_vectors(ux, uy, uz, Sx, Sy, Sz)
        cs.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
