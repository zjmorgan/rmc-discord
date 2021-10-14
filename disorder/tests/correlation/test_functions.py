#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space
from disorder.correlation import functions

class test_functions(unittest.TestCase):

    def test_pairs1d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 7
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, fract=1.0)
        distance, ion_pair, counts, search, coordinate, N = data
       
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        m_uvw = mu*mv*mw
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        self.assertEqual(counts.sum(), nc+nl)
        
    def test_pairs3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 1, 5, 9
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co', 'Ni'])
        u = np.array([0,0.2,0.25])
        v = np.array([0.01,0.31,0.1])
        w = np.array([0.1,0.4,0.62])
        
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, fract=1.0)
        distance, ion_pair, counts, search, coordinate, N = data
        
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        m_uvw = mu*mv*mw
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        self.assertEqual(counts.sum(), nc+nl)
        
if __name__ == '__main__':
    unittest.main()