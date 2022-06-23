#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import unittest
import numpy as np

from disorder.diffuse cimport refinement

class test_c_refinement(unittest.TestCase):

    def test_random_uniform_nonzero(self):

        N = 100000

        u = np.zeros(N)

        for i in range(N):
            u[i] = refinement.random_uniform_nonzero()

        self.assertAlmostEqual(u.mean(), 0.5, 2)
        self.assertGreater(u.min(), 0)
        self.assertLessEqual(u.mean(), 1)

    def test_random_uniform(self):

        N = 100000

        u = np.zeros(N)

        for i in range(N):
            u[i] = refinement.random_uniform()

        self.assertAlmostEqual(u.mean(), 0.5, 2)
        self.assertGreaterEqual(u.min(), 0)
        self.assertLessEqual(u.mean(), 1)

    def test_random_gaussian(self):

        N = 100000

        u = np.zeros(N)

        for i in range(N):
            u[i] = refinement.random_gaussian()

        self.assertAlmostEqual(u.mean(), 0.0, 1)
        self.assertAlmostEqual(u.std(), 1.0, 1)

    def test_random_gaussian_3d(self):

        N = 100000

        u, v, w = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            u[i], v[i], w[i] = refinement.random_gaussian_3d()

        self.assertAlmostEqual(u.mean(), 0.0, 1)
        self.assertAlmostEqual(v.mean(), 0.0, 1)
        self.assertAlmostEqual(w.mean(), 0.0, 1)
        self.assertAlmostEqual(u.std(), 1.0, 1)
        self.assertAlmostEqual(v.std(), 1.0, 1)
        self.assertAlmostEqual(w.std(), 1.0, 1)

    def test_iszero(self):

        self.assertTrue(refinement.iszero(1e-31))
        self.assertFalse(refinement.iszero(1e-15))

    def test_cexp(self):

        z = np.random.random()+1j*np.random.random()

        self.assertAlmostEqual(refinement.cexp(z), np.exp(z))

    def test_iexp(self):

        z = np.random.random()

        self.assertAlmostEqual(refinement.iexp(z), np.exp(1j*z))