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
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co', 'Ni'])
        u, v, w = np.array([0,0.2,0.25]), np.array([0,0.3,0.1]), np.array([0,0.4,0.6])
        
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, fract=1.0, tol=1e-2)
        distance, ion_pair, counts, search, coordinate, N = data
       
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        m_uvw = mu*mv*mw
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*(n_uvw*(m_uvw*2+1))
        
        #self.assertEqual(counts.sum(), nc+nl)
        
    def test_pairs3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 6, 4, 5
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        atm = np.array(['Fe', 'Co', 'Ni'])
        u, v, w = np.array([0,0.2,0.25]), np.array([0,0.3,0.1]), np.array([0,0.4,0.6])
        
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
        nl = n_atm**2*(n_uvw*(m_uvw*2+1))
        
        #self.assertEqual(counts.sum(), nc+nl)
        
if __name__ == '__main__':
    unittest.main()