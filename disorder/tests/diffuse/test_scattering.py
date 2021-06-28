#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, scattering

class test_scattering(unittest.TestCase):

    def test_phase(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        h, k, l = np.array([-7,-3]), np.array([2,2]), np.array([2,5])
        
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
                
        phase_factor = scattering.phase(Qx, Qy, Qz, rx, ry, rz)
        
        np.testing.assert_array_almost_equal(phase_factor, 1+0j)
        
if __name__ == '__main__':
    unittest.main()