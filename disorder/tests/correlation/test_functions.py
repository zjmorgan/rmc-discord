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
        
        nu, nv, nw = 3, 4, 8
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co', 'Ni'])
        u = np.array([0,0.2,0.25])
        v = np.array([0.01,0.31,0.1])
        w = np.array([0.1,0.4,0.62])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, fract=1.0)
        distance, atm_pair, counts, search, coordinate, N = data
       
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        ns = (1+nu-2*mu)*nv*nw+(1+nv-2*mv)*nw*nu+(1+nw-2*mw)*nu*nv
        
        nd = (1+nu-2*mu)*(1+nv-2*mv)*nw\
           + (1+nv-2*mv)*(1+nw-2*mw)*nu\
           + (1+nw-2*mw)*(1+nu-2*mu)*nv
          
        nt = (1+nu-2*mu)*(1+nv-2*mv)*(1+nw-2*mw)
        
        nr = n_atm**2*n_uvw*(ns-nd+nt)//2 # even removal
        
        self.assertEqual(counts.sum(), nc+nl-nr)
        
        self.assertEqual(search.size, N)
        
        self.assertEqual(atm_pair.shape[0], search.shape[0])
        
        np.testing.assert_array_equal(np.diff(search), counts)
        
        i, j = coordinate.T
                
        Dx = (rx[j]-rx[i])[search[:-1]]
        Dy = (ry[j]-ry[i])[search[:-1]]
        Dz = (rz[j]-rz[i])[search[:-1]]

        Distance = np.sqrt(Dx**2+Dy**2+Dz**2)

        np.testing.assert_array_almost_equal(Distance, distance[:-1])
        
        atm_pair_ij = np.stack((atms[i],atms[j])).T[search[:-1]]
        atm_pair_ij = np.sort(atm_pair_ij, axis=1)
        
        unique_pairs = np.stack([ap.split('_') for ap in atm_pair[:-1]])
        
        np.testing.assert_array_equal(atm_pair_ij, unique_pairs)  

    def test_pairs3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 4, 2, 8
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co', 'Ni', 'Mn'])
        u = np.array([0,0.2,0.25,0.25])
        v = np.array([0.01,0.31,0.1,0.1])
        w = np.array([0.1,0.4,0.62,0.62])
        
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs3d(rx, ry, rz, atms, nu, nv, nw, fract=1.0)
        dx, dy, dz, atm_pair, counts, search, coordinate, N = data
        
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        ns = (1+nu-2*mu)*nv*nw+(1+nv-2*mv)*nw*nu+(1+nw-2*mw)*nu*nv
        
        nd = (1+nu-2*mu)*(1+nv-2*mv)*nw\
           + (1+nv-2*mv)*(1+nw-2*mw)*nu\
           + (1+nw-2*mw)*(1+nu-2*mu)*nv
          
        nt = (1+nu-2*mu)*(1+nv-2*mv)*(1+nw-2*mw)
        
        nr = n_atm**2*n_uvw*(ns-nd+nt)//2 # even removal
        
        self.assertEqual(counts.sum(), nc+nl-nr)
        
        self.assertEqual(search.size, N)
        
        self.assertEqual(atm_pair.shape[0], search.shape[0])
        
        np.testing.assert_array_equal(np.diff(search), counts)
        
        i, j = coordinate.T
                
        Dx = (rx[j]-rx[i])[search[:-1]]
        Dy = (ry[j]-ry[i])[search[:-1]]
        Dz = (rz[j]-rz[i])[search[:-1]]

        np.testing.assert_array_almost_equal(Dx, dx[:-1])
        np.testing.assert_array_almost_equal(Dy, dy[:-1])
        np.testing.assert_array_almost_equal(Dz, dz[:-1])
        
        atm_pair_ij = np.stack((atms[i],atms[j])).T[search[:-1]]
        atm_pair_ij = np.sort(atm_pair_ij, axis=1)
        
        unique_pairs = np.stack([ap.split('_') for ap in atm_pair[:-1]])
        
        np.testing.assert_array_equal(atm_pair_ij, unique_pairs)        
        
if __name__ == '__main__':
    unittest.main()