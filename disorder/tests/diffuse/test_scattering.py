#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, scattering

class test_scattering(unittest.TestCase):
    
    def test_length(self):
        
        atms = ['H']
                
        b = scattering.length(atms, 10)
        
        np.testing.assert_array_almost_equal(b, -3.7390, 3)
 
        atms = ['1H','2H','3H']
                
        b = scattering.length(atms, 13)
        
        np.testing.assert_array_almost_equal(b.reshape(13,3)[:,0], -3.7406, 3)
        np.testing.assert_array_almost_equal(b.reshape(13,3)[:,1], 6.671, 3)
        np.testing.assert_array_almost_equal(b.reshape(13,3)[:,2], 4.792, 3)
    
    def test_form(self):
        
        ions = ['Mn3+']
        
        Q = np.array([0.,5.,100.])
        
        f = scattering.form(ions, Q, source='x-ray')
        
        self.assertAlmostEqual(f[0], 22.000047, 3)
        self.assertAlmostEqual(f[1], 13.077269, 3)
        self.assertAlmostEqual(f[2], 0.3939741, 3)
        
        ions = ['La3+']
        
        Q = np.array([0.,5.,100.])
        
        f = scattering.form(ions, Q, source='x-ray')
        
        self.assertAlmostEqual(f[0], 54.002148, 3)
        self.assertAlmostEqual(f[1], 34.475719, 3)
        self.assertAlmostEqual(f[2], 2.4086025, 3)

        ions = ['O','Mg']
        
        Q = 4*np.pi*np.array([0.01,0.05,0.1])
        
        f = scattering.form(ions, Q, source='electron')
        f = f.reshape(Q.size,len(ions))
        
        self.assertAlmostEqual(f[0,0], 1.981, 2)
        self.assertAlmostEqual(f[1,0], 1.937, 2)
        self.assertAlmostEqual(f[2,0], 1.808, 2)
        
        self.assertAlmostEqual(f[0,1], 5.187, 2)
        self.assertAlmostEqual(f[1,1], 4.717, 2)
        self.assertAlmostEqual(f[2,1], 3.656, 2)
        
        ions = ['O1-','Mg2+']
                
        Q = 4*np.pi*np.array([0.01,0.05,0.1])
        
        f = scattering.form(ions, Q, source='electron')
        f = f.reshape(Q.size,len(ions))
        
        self.assertAlmostEqual(f[0,0], -236.025, 1)
        self.assertAlmostEqual(f[1,0], -6.41, 1)
        self.assertAlmostEqual(f[2,0], 0.39, 1)
        
        self.assertAlmostEqual(f[0,1], 479.502, 1)
        self.assertAlmostEqual(f[1,1], 19.970, 1)
        self.assertAlmostEqual(f[2,1], 5.593, 1)
        
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