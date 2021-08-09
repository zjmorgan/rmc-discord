#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.diffuse import annealing

class test_annealing(unittest.TestCase):
    
    def test_original(self):
        
        np.random.seed(13)
        
        nh, nk, nl, n_atm = 16, 27, 36, 3

        A = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        B = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        C = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        
        A_orig, i = annealing.original(A)
        
        self.assertAlmostEqual(A[i], A_orig)
        
        A_orig, B_orig, C_orig, i = annealing.original(A, B, C)
        
        self.assertAlmostEqual(A[i], A_orig)
        self.assertAlmostEqual(B[i], B_orig)
        self.assertAlmostEqual(C[i], C_orig)
    
    def test_candidate(self):
        
        np.random.seed(13)
        
        nh, nk, nl, n_atm = 16, 27, 36, 3

        A = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        B = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        C = np.random.random(size=(nh,nk,nl,n_atm)).flatten()
        
        A_orig, i = annealing.original(A)
        
        A_cand = annealing.candidate(A_orig)
        
        self.assertAlmostEqual(1/1-2-A[i], A_cand)
        
        value = 0.6
        A_cand = annealing.candidate(A_orig, value=value)

        self.assertAlmostEqual(1/value-2-A[i], A_cand)
        
        value = 0.4
        A_cand = annealing.candidate(A_orig, value=value, fixed=False)

        self.assertLess(1/value-2-A[i], A_cand)
       
        A_orig, B_orig, C_orig, i = annealing.original(A, B, C)
        
        A_cand, B_cand, C_cand = annealing.candidate(A_orig, B_orig, C_orig)
        
        self.assertAlmostEqual(1, A_cand**2+B_cand**2+C_cand**2)
        
        value = 0.7
        A_cand, B_cand, C_cand = annealing.candidate(A_orig, 
                                                     B_orig, 
                                                     C_orig, 
                                                     value=value)
        
        self.assertAlmostEqual(value**2, A_cand**2+B_cand**2+C_cand**2)
        
        value = 0.3
        A_cand, B_cand, C_cand = annealing.candidate(A_orig, 
                                                     B_orig, 
                                                     C_orig, 
                                                     value=value, 
                                                     fixed=False)
        
        self.assertGreater(value**2, A_cand**2+B_cand**2+C_cand**2)
        
    def test_test(self):
                
        beta, E = 13, 17
        
        self.assertGreater(annealing.test(beta, E), 0)
        
        beta, E = 8, 0
        
        self.assertAlmostEqual(annealing.test(beta, E), 1)
        
        beta, E = 0, 29
        
        self.assertAlmostEqual(annealing.test(beta, E), 1)
        
        beta, E = 13, -17
        
        self.assertGreater(annealing.test(beta, E), 1)

if __name__ == '__main__':
    unittest.main()