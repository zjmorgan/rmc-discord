#!/usr/bin/env python3U

import io
import os
import sys

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, scattering

class test_scattering(unittest.TestCase):
    
    def test_parallelism(self):
                    
        out = io.StringIO()
        sys.stdout = out
        
        scattering.parallelism(app=False)
        
        sys.stdout = sys.__stdout__
        
        num_threads = os.environ.get('OMP_NUM_THREADS')

        self.assertEqual(out.getvalue(), 'threads: {}\n'.format(num_threads))
                        
    def test_threads(self):
                    
        out = io.StringIO()
        sys.stdout = out
        
        scattering.threads()
        
        sys.stdout = sys.__stdout__
        
        # num_threads = os.environ.get('OMP_NUM_THREADS')
        
        print(out.getvalue())
        
        # self.assertEqual(out.getvalue(), 
        #                   ''.join(['id: {}\n'.format(i_thread) \
        #                           for i_thread in range(int(num_threads))]))
            
    def test_extract(self):
        
        n_hkl, n_atm = 101, 3
        
        n = n_hkl*n_atm
        
        data = np.random.random(n)+1j*np.random.random(n)
        values = np.zeros(n_hkl, dtype=complex)
                
        j = 1
        scattering.extract(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])
        
        j = 2
        scattering.extract(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])
        
    def test_insert(self):
        
        n_hkl, n_atm = 101, 3
        
        n = n_hkl*n_atm
        
        data = np.random.random(n)+1j*np.random.random(n)
        values = np.random.random(n_hkl)+1j*np.random.random(n_hkl)
                
        j = 1
        scattering.insert(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])
        
        j = 2
        scattering.insert(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

    def test_phase(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        h, k, l = np.array([-7,-3]), np.array([2,2]), np.array([2,5])
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
                
        phase_factor = scattering.phase(Qx, Qy, Qz, rx, ry, rz)
        
        np.testing.assert_array_almost_equal(phase_factor, 1+0j)
        
if __name__ == '__main__':
    unittest.main()