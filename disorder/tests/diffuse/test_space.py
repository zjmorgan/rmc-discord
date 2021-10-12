#!/usr/bin/env python3

import unittest
import numpy as np

import scipy.ndimage.filters 

from disorder.material import crystal
from disorder.diffuse import space

class test_space(unittest.TestCase):
    
    def test_reciprocal(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        np.random.seed(13)
        
        nh, nk, nl = 3, 2, 5

        mask = np.random.randint(0, 2, size=(nh,nk,nl), dtype=bool)
        
        h_range, k_range, l_range = [0,2], [-2,-1], [-1,1]
        
        Qh, Qk, Ql = space.reciprocal(h_range, k_range, l_range, mask, B)
        
        n = (~mask).sum()
        
        self.assertEqual(n, Qh.size)
        self.assertEqual(n, Qk.size)
        self.assertEqual(n, Ql.size)
        
        Q = np.sqrt(Qh**2+Qk**2+Ql**2)
        
        h_, k_, l_  = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                                  np.linspace(k_range[0],k_range[1],nk), 
                                  np.linspace(l_range[0],l_range[1],nl), 
                                  indexing='ij')
        
        h, k, l = h_[~mask], k_[~mask], l_[~mask]
            
        d = crystal.d(a, b, c, alpha, beta, gamma, h, k, l)
        
        np.testing.assert_array_almost_equal(d, 2*np.pi/Q)

        T = np.array([[-1,1,0],[1,1,0],[0,0,1]])
        
        Qh, Qk, Ql = space.reciprocal(h_range, k_range, l_range, mask, B, T=T)

        h, k, l = np.dot(T, np.array([h_[~mask], k_[~mask], l_[~mask]]))
            
        d = crystal.d(a, b, c, alpha, beta, gamma, h, k, l)
        
        Q = np.sqrt(Qh**2+Qk**2+Ql**2)
        
        np.testing.assert_array_almost_equal(d, 2*np.pi/Q)
        
    def test_nuclear(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)       
        
        h, k, l = -3, 1, 2
        
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
    
        d = crystal.d(a, b, c, alpha, beta, gamma, h, k, l)
        
        Q = np.sqrt(Qh**2+Qk**2+Ql**2)
        
        self.assertAlmostEqual(d, 2*np.pi/Q)
            
    def test_cell(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        h, k, l = -3, 1, 2
        
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        np.testing.assert_array_almost_equal(np.exp(1j*(Qx*Rx+Qy*Ry+Qz*Rz)), 1)
        
    def test_real(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        h, k, l = -3, 2, 5
        
        Qh, Qk, Ql = space.nuclear(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        np.testing.assert_array_almost_equal(np.exp(1j*(Qx*rx+Qy*ry+Qz*rz)), 1)
        
    def test_factor(self):
        
        nu, nv, nw = 5, 3, 4
        
        pf = space.factor(nu, nv, nw)
        
        self.assertEqual(pf.size, (nu*nv*nw)**2)
        
        phase_factor = pf.reshape(nu**2,nv**2,nw**2)
        
        self.assertAlmostEqual(phase_factor[0,0,0], 1+0j)
        
        ki, kj, kk, ri, rj, rk = 2, 1, 3, 4, 2, 1
        
        value = np.exp(2j*np.pi*(ki*ri/nu+kj*rj/nv+kk*rk/nw))
        
        self.assertAlmostEqual(phase_factor[ri+nu*ki,rj+nv*kj,rk+nw*kk], value)

    def test_unit(self):
        
        theta = 2*np.pi*np.random.rand((10))
        phi = np.arccos(1-2*np.random.rand((10)))
        
        nx = np.sin(phi)*np.cos(theta)
        ny = np.sin(phi)*np.sin(theta)
        nz = np.cos(phi)
        
        np.random.seed(13)

        v = np.random.rand((10))
        
        vx, vy, vz = v*nx, v*ny, v*nz
        
        ux, uy, uz, u = space.unit(vx, vy, vz)
        
        np.testing.assert_array_almost_equal(u, v)
        np.testing.assert_array_almost_equal(ux, nx)
        np.testing.assert_array_almost_equal(uy, ny)
        np.testing.assert_array_almost_equal(uz, nz)
        
        v[0] = 0
        
        vx, vy, vz = v*nx, v*ny, v*nz
        
        ux, uy, uz, u = space.unit(vx, vy, vz)
        
        self.assertAlmostEqual(u[0], 0)
        self.assertAlmostEqual(ux[0], 0)
        self.assertAlmostEqual(uy[0], 0)
        self.assertAlmostEqual(uz[0], 0)
        
        np.testing.assert_array_almost_equal(u[1:], v[1:])
        np.testing.assert_array_almost_equal(ux[1:], nx[1:])
        np.testing.assert_array_almost_equal(uy[1:], ny[1:])
        np.testing.assert_array_almost_equal(uz[1:], nz[1:])
        
    def test_transform(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        A_r = np.random.random((nu,nv,nw,n_atm))
        
        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))
        
        A_k, i_dft = space.transform(A_r, H, K, L, nu, nv, nw, n_atm)
        
        A_r = A_r.reshape(nu,nv,nw,n_atm)
        
        A_k = A_k.reshape(nu,nv,nw,n_atm)
        
        n_uvw = nu*nv*nw
        
        np.testing.assert_array_almost_equal(A_k[0,0,0,:], \
                                             np.mean(A_r, axis=(0,1,2))*n_uvw)
        
        w = i_dft % nw
        v = i_dft // nw % nv
        u = i_dft // nw // nv % nu
        
        np.testing.assert_array_equal(u, np.mod(H, nu))
        np.testing.assert_array_equal(v, np.mod(K, nv))
        np.testing.assert_array_equal(w, np.mod(L, nw))     
        
if __name__ == '__main__':
    unittest.main()
