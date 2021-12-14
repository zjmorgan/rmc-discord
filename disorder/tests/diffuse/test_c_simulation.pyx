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
        
        ux, uy, uz = simulation.random_vector_candidate()
        
        self.assertAlmostEqual(ux**2+uy**2+uz**2, 1)
        
    def test_random_gaussian(self):
        
        N = 100000
        
        u = np.zeros(N)
        
        for i in range(N):
            u[i] = simulation.random_gaussian()
            
        self.assertAlmostEqual(u.mean(), 0.0, 2)
        self.assertAlmostEqual(u.std(), 1.0, 2)