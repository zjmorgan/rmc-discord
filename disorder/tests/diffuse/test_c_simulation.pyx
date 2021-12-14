#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import unittest
import numpy as np

from disorder.diffuse import space
from disorder.material import crystal

from disorder.diffuse cimport simulation

class test_c_simulation(unittest.TestCase):

    def test_iszero(self):

        self.assertTrue(simulation.iszero(1e-15))
        self.assertFalse(simulation.iszero(1e-7))
        
    def test_alpha(self):
        
        E = 0.01
        T = 5
        
        kB = 0.08617
        beta = 1/(kB*T)
        
        self.assertAlmostEqual(simulation.alpha(E,beta), np.exp(-E*beta))
        
    def test_random_uniform(self):
        
        N = 100000
        
        u = np.zeros(N)
        
        for i in range(N):
            u[i] = simulation.random_uniform()
            
        self.assertAlmostEqual(u.mean(), 0.5, 2)
        self.assertGreaterEqual(u.min(), 0)
        self.assertLessEqual(u.mean(), 1)

    def test_random_vector_candidate(self):
        
        N = 100000
        
        ux, uy, uz = np.zeros(N), np.zeros(N), np.zeros(N)
        
        for i in range(N):
            ux[i], uy[i], uz[i] = simulation.random_vector_candidate()
        
        u = np.sqrt(ux**2+uy**2+uz**2)
        t = np.arccos(uz/u)
        p = np.arctan2(ux,uy)
        
        np.testing.assert_array_almost_equal(u, 1)
        
        self.assertAlmostEqual(np.cos(t).mean(), 0.0, 1)
        self.assertAlmostEqual(p.mean(), 0.0, 1)
        
    def test_random_gaussian(self):
        
        N = 100000
        
        u = np.zeros(N)
        
        for i in range(N):
            u[i] = simulation.random_gaussian()
            
        self.assertAlmostEqual(u.mean(), 0.0, 2)
        self.assertAlmostEqual(u.std(), 1.0, 2)
        
    def test_random_vector_length_candidate(self):

        N = 100000
        
        ux, uy, uz = np.zeros(N), np.zeros(N), np.zeros(N)
        
        for i in range(N):
            ux[i], uy[i], uz[i] = simulation.random_vector_length_candidate()
        
        u = np.sqrt(ux**2+uy**2+uz**2)
        t = np.arccos(uz/u)
        p = np.arctan2(ux,uy)
        
        np.testing.assert_array_equal(u <= 1, True)
        
        self.assertAlmostEqual(np.cos(t).mean(), 0.0, 1)
        self.assertAlmostEqual(p.mean(), 0.0, 1)
        
    def test_random_gaussian_3d(self):
        
        N = 100000
        
        u, v, w = np.zeros(N), np.zeros(N), np.zeros(N)
        
        for i in range(N):
            u[i], v[i], w[i] = simulation.random_gaussian_3d()
            
        self.assertAlmostEqual(u.mean(), 0.0, 2)
        self.assertAlmostEqual(v.mean(), 0.0, 2)
        self.assertAlmostEqual(w.mean(), 0.0, 2)
        self.assertAlmostEqual(u.std(), 1.0, 2)
        self.assertAlmostEqual(v.std(), 1.0, 2)
        self.assertAlmostEqual(w.std(), 1.0, 2)
        
    def test_gaussian_vector_candidate(self):
        
        ux, uy, uz = simulation.random_vector_candidate()
        
        vx, vy, vz = simulation.gaussian_vector_candidate(ux, uy, uz, 0)
        
        self.assertAlmostEqual(ux, vx)
        self.assertAlmostEqual(uy, vy)
        self.assertAlmostEqual(uz, vz)
        
        for i in range(10):
            vx, vy, vz = simulation.gaussian_vector_candidate(ux, uy, uz, 10)
            t = np.arccos(vz)
            p = np.arctan2(vx,vy)
            print(t,p)
