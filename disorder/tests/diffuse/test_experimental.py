#!/usr/bin/env python3

import unittest
import numpy as np

import pyvista as pv

from disorder.diffuse import experimental

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_experimental(unittest.TestCase):

    def test_data(self):

        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        np.random.seed(13)

        signal, sigma_sq, \
        h_range, k_range, l_range, \
        nh, nk, nl = experimental.data(os.path.join(folder, 'test.nxs'))

        self.assertEqual(nh, 13)
        self.assertEqual(nk, 7)
        self.assertEqual(nl, 26)

        self.assertEqual(h_range, [-4,2])
        self.assertEqual(k_range, [-2,4])
        self.assertEqual(l_range, [-3,3])

        shape = (13,7,26)
        np.testing.assert_array_almost_equal(signal, np.random.random(shape))
        np.testing.assert_array_almost_equal(sigma_sq, np.random.random(shape))

    def test_mask(self):

        np.random.seed(13)

        signal = 1000*np.random.random((10,11,12))
        error_sq = np.sqrt(signal)

        signal[1,2,3] = np.nan
        signal[2,3,4] = np.inf
        signal[4,5,6] = -1

        error_sq[2,3,4] = np.nan
        error_sq[3,4,5] = -1

        mask = experimental.mask(signal, error_sq)

        self.assertTrue((signal[~mask] > 0).all())
        self.assertTrue((error_sq[~mask] > 0).all())

    def test_rebin(self):

        x, y, z = np.meshgrid(np.arange(6),
                              np.arange(14),
                              np.arange(10), indexing='ij')

        data = 0.5*x+2.5*y-z

        tmp_data = experimental.rebin(data, [6,14,10])
        np.testing.assert_array_almost_equal(tmp_data, data)

        tmp_data = experimental.rebin(data, [3,14,10])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=0)*2,
                                             np.sum(data, axis=0))

        tmp_data = experimental.rebin(data, [6,7,10])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=1)*2,
                                             np.sum(data, axis=1))

        tmp_data = experimental.rebin(data, [6,14,2])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=2)*5,
                                             np.sum(data, axis=2))

        tmp_data = experimental.rebin(data, [3,7,10])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=(0,1))*4,
                                             np.sum(data, axis=(0,1)))

        tmp_data = experimental.rebin(data, [6,7,2])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=(1,2))*10,
                                             np.sum(data, axis=(1,2)))

        tmp_data = experimental.rebin(data, [3,14,2])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=(0,2))*10,
                                             np.sum(data, axis=(0,2)))

        tmp_data = experimental.rebin(data, [3,7,2])
        np.testing.assert_array_almost_equal(np.sum(tmp_data, axis=(0,1,2))*20,
                                             np.sum(data, axis=(0,1,2)))

    def test_weights(self):

        weight = experimental.weights(4, 2)
        np.testing.assert_array_almost_equal(weight.sum(axis=0), 0.5)
        np.testing.assert_array_almost_equal(weight.sum(axis=1), 1.0)

        weight = experimental.weights(10, 2)
        np.testing.assert_array_almost_equal(weight.sum(axis=0), 0.2)
        np.testing.assert_array_almost_equal(weight.sum(axis=1), 1.0)

        weight = experimental.weights(5, 3)
        np.testing.assert_array_almost_equal(weight.sum(axis=0), 0.6)
        np.testing.assert_array_almost_equal(weight.sum(axis=1), 1.0)

    def test_crop(self):

        data  = np.random.random((23,24,25))

        tmp_data = experimental.crop(data, [0,23],[0,24],[0,25])

        np.testing.assert_array_almost_equal(tmp_data, data)

        tmp_data = experimental.crop(data, [3,6],[4,8],[5,10])

        self.assertEqual(tmp_data.shape[0], 3)
        self.assertEqual(tmp_data.shape[1], 4)
        self.assertEqual(tmp_data.shape[2], 5)

        self.assertAlmostEqual(tmp_data[0,0,0], data[3,4,5])

    def test_factors(self):

        fact = np.array([1, 2, 5, 10])
        np.testing.assert_array_equal(experimental.factors(10), fact)

        fact = np.array([1, 11])
        np.testing.assert_array_equal(experimental.factors(11), fact)

        fact = np.array([1, 5, 25])
        np.testing.assert_array_equal(experimental.factors(25), fact)

    def test_punch(self):

        h_range, nh = [-3,3], 25
        k_range, nk = [-4,4], 33
        l_range, nl = [-5,5], 41

        radius_h, radius_k, radius_l = 2, 3, 4

        h, k, l = np.meshgrid(np.linspace(h_range[0],h_range[1],nh),
                              np.linspace(k_range[0],k_range[1],nk),
                              np.linspace(l_range[0],l_range[1],nl),
                              indexing='ij')

        signal = np.random.random((nh,nk,nl))

        mask = np.isclose(np.mod(h,1),0)\
             & np.isclose(np.mod(k,1),0)\
             & np.isclose(np.mod(l,1),0)

        signal[mask] = 10

        data = experimental.punch(signal, radius_h, radius_k, radius_l,
                                  h_range, k_range, l_range, punch='Box')

        np.testing.assert_array_equal(np.isnan(data[mask]), True)

        data = experimental.punch(signal, radius_h, radius_k, radius_l,
                                  h_range, k_range, l_range, punch='Ellipsoid')

        np.testing.assert_array_equal(np.isnan(data[mask]), True)

    def test_outlier(self):

        signal = np.random.random((25,26,27))
        signal[3,4,5] = 2
        signal[13,14,15] = -1

        size = 3
        data = experimental.outlier(signal, size)

        self.assertTrue(np.isnan(data[3,4,5]))
        self.assertTrue(np.isnan(data[13,14,15]))

    def test_reflections(self):

        cntr = 'P'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 5, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(4, 2, 6, centering=cntr), 1)

        cntr = 'I'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 5, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(4, 2, 6, centering=cntr), 1)

        cntr = 'F'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(1, 5, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(4, 2, 6, centering=cntr), 1)

        cntr = 'A'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(2, 5, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(3, 2, 6, centering=cntr), 1)

        cntr = 'B'
        self.assertEqual(experimental.reflections(3, 1, 2, centering=cntr), 0)
        self.assertEqual(experimental.reflections(3, 2, 5, centering=cntr), 1)
        self.assertEqual(experimental.reflections(6, 3, 2, centering=cntr), 1)

        cntr = 'C'
        self.assertEqual(experimental.reflections(2, 3, 1, centering=cntr), 0)
        self.assertEqual(experimental.reflections(5, 3, 2, centering=cntr), 1)
        self.assertEqual(experimental.reflections(2, 6, 3, centering=cntr), 1)

        cntr = 'R(obv)'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(1, 2, 2, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 2, 5, centering=cntr), 1)

        cntr = 'R(rev)'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(1, 2, 4, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 2, 5, centering=cntr), 0)

        cntr = 'H'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 0)
        self.assertEqual(experimental.reflections(1, 4, 4, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 4, 5, centering=cntr), 1)

        cntr = 'D'
        self.assertEqual(experimental.reflections(1, 2, 3, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 4, 4, centering=cntr), 1)
        self.assertEqual(experimental.reflections(1, 4, 5, centering=cntr), 0)

    def test_intensity(self):

        folder = os.path.abspath(os.path.join(directory, '..', 'data'))

        x, y, z = np.meshgrid(np.linspace(-3,3,6),
                              np.linspace(-7,7,14),
                              np.linspace(-1,1,10), indexing='ij')

        data = 0.5*x+2.5*y-z

        A = np.array([[2,1,0],[-0.5,1,0],[0,0,1]])

        experimental.intensity(folder+'/test.vts', x, y, z, data, B=A)

        grid = pv.read(folder+'/test.vts')

        values = grid.point_arrays['intensity'].reshape(6,14,10,order='F')

        np.testing.assert_array_almost_equal(data, values)

        A_inv = np.linalg.inv(A)

        T = np.eye(4)
        T[:3,:3] = A_inv

        grid.transform(T)

        xs, ys, zs = grid.x, grid.y, grid.z

        np.testing.assert_array_almost_equal(x, xs)
        np.testing.assert_array_almost_equal(y, ys)
        np.testing.assert_array_almost_equal(z, zs)

        os.remove(folder+'/test.vts')

if __name__ == '__main__':
    unittest.main()