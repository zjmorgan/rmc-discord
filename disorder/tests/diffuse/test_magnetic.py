#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import tables
from disorder.diffuse import magnetic

class test_magnetic(unittest.TestCase):
    
    def test_j0(self):
                
        A, a, B, b, C, c, D = tables.j0.get('Fe3+')
        
        Q = np.array([0,5,100])
        
        j0 = magnetic.j0(Q, A, a, B, b, C, c, D)
        
        self.assertAlmostEqual(j0[0], 1.0, 3)
        self.assertAlmostEqual(j0[1], 0.31313, 3)
        self.assertAlmostEqual(j0[2], 0.00440, 3)
            
    def test_j2(self):
                
        A, a, B, b, C, c, D = tables.j2.get('Fe3+')
        
        Q = np.array([0,5,100])
        
        j2 = magnetic.j2(Q, A, a, B, b, C, c, D)
        
        self.assertAlmostEqual(j2[0], 0.0, 3)
        self.assertAlmostEqual(j2[1], 0.19260, 3)
        self.assertAlmostEqual(j2[2], 0.22164, 3)
        
    def test_f(self):

        A0, a0, B0, b0, C0, c0, D0 = tables.j0.get('Fe3+')
        A2, a2, B2, b2, C2, c2, D2 = tables.j2.get('Fe3+')
        
        Q = np.array([0,5,100])

        j0 = magnetic.j0(Q, A0, a0, B0, b0, C0, c0, D0)        
        j2 = magnetic.j2(Q, A2, a2, B2, b2, C2, c2, D2)
        
        f = magnetic.f(Q, j0, j2, K2=0)
        
        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.31313, 3)
        self.assertAlmostEqual(f[2], 0.00440, 3)
        
        f = magnetic.f(Q, j0, j2, K2=1)
        
        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.50573, 3)
        self.assertAlmostEqual(f[2], 0.22604, 3)
        
        f = magnetic.f(Q, j0, j2, K2=-1)
        
        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.12052, 3)
        self.assertAlmostEqual(f[2], -0.21724, 3)

    def test_form(self):
        
        ions = ['Fe3+']
        
        Q = np.array([0,5,100])
    
        f = magnetic.form(Q, ions, g=2)
        
        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.31313, 3)
        self.assertAlmostEqual(f[2], 0.00440, 3)
        
        f = magnetic.form(Q, ions, g=1)

        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.50573, 3)
        self.assertAlmostEqual(f[2], 0.22604, 3)
        
        f = magnetic.form(Q, ions, g=1e+6)
        
        self.assertAlmostEqual(f[0], 1.0, 3)
        self.assertAlmostEqual(f[1], 0.12052, 3)
        self.assertAlmostEqual(f[2], 0.0, 3)
        
    def test_spin(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)
        S = np.sqrt(Sx**2+Sy**2+Sz**2)
        
        np.testing.assert_array_almost_equal(S, np.ones(nu*nv*nw*n_atm))
        
        np.random.seed(13)

        u, v = np.random.rand(nu,nv,nw,n_atm), np.random.rand(nu,nv,nw,n_atm)
        
        theta = np.mod(np.arctan2(Sy,Sx), 2*np.pi)
        phi = np.arccos(Sz/S) 

        np.testing.assert_array_almost_equal(theta, 2*np.pi*u.flatten())
        np.testing.assert_array_almost_equal(phi, np.arccos(1-2*v.flatten()))
        
    def test_transform(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)

        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))
        
        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, 
                                                     Sy, 
                                                     Sz, 
                                                     H, 
                                                     K, 
                                                     L, 
                                                     nu, 
                                                     nv, 
                                                     nw, 
                                                     n_atm)
        
        Sx = Sx.reshape(nu,nv,nw,n_atm)
        Sy = Sy.reshape(nu,nv,nw,n_atm)
        Sz = Sz.reshape(nu,nv,nw,n_atm)
        
        Sx_k = Sx_k.reshape(nu,nv,nw,n_atm)
        Sy_k = Sy_k.reshape(nu,nv,nw,n_atm)
        Sz_k = Sz_k.reshape(nu,nv,nw,n_atm)
        
        n_uvw = nu*nv*nw
        
        np.testing.assert_array_almost_equal(Sx_k[0,0,0,:], \
                                             np.mean(Sx, axis=(0,1,2))*n_uvw)
        np.testing.assert_array_almost_equal(Sy_k[0,0,0,:], \
                                             np.mean(Sy, axis=(0,1,2))*n_uvw)
        np.testing.assert_array_almost_equal(Sz_k[0,0,0,:], \
                                             np.mean(Sz, axis=(0,1,2))*n_uvw)
        
        w = i_dft % nw
        v = i_dft // nw % nv
        u = i_dft // nw // nv % nu
        
        np.testing.assert_array_equal(u, np.mod(H, nu))
        np.testing.assert_array_equal(v, np.mod(K, nv))
        np.testing.assert_array_equal(w, np.mod(L, nw))        
        
if __name__ == '__main__':
    unittest.main()