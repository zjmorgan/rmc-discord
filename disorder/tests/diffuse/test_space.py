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

    def test_boxblur(self):
        
        sigma, n = 2, 3
        boxes = space.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
    
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
                
        sigma, n = 3, 2
        boxes = space.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
            
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
        
        sigma, n = 4, 3
        boxes = space.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
            
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
        
        sigma, n = np.array([3,1,2]), 4
        boxes = space.boxblur(sigma, n)
        
        l = (np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1).astype(int)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
                    
        self.assertEqual(boxes.size, n*sigma.size)     
        
        boxes = boxes.reshape(n, sigma.size)
                
        np.testing.assert_array_equal(np.sum(boxes, axis=0), n*(l+1)/2-m)
        
    def test_gaussian(self):
        
        np.random.seed(13)
        
        nh, nk, nl = 16, 27, 36

        mask = np.random.randint(0, 2, size=(nh,nk,nl), dtype=bool)
        
        sigma = [2,1,3]
        v_inv = space.gaussian(mask, sigma).reshape(nh,nk,nl)
                
        v = np.ones(mask.shape)
        v[mask] = 0
            
        w = space.blurring(v, sigma)

        np.testing.assert_array_almost_equal(w*v_inv, np.ones(mask.shape))
        
    def test_boxfilter(self):
        
        np.random.seed(13)
        
        nh, nk, nl = 16, 27, 36

        mask = np.random.randint(0, 2, size=(nh,nk,nl), dtype=bool)
        
        sigma = [2,1,3]
        v_inv = space.gaussian(mask, sigma)
        
        v = np.ones(mask.shape)
        v[mask] = 0
    
        w = space.boxfilter(v, mask, sigma, v_inv)
        
        np.testing.assert_array_almost_equal(w, np.ones(mask.shape))
        
    def test_blurring(self):
        
        nh, nk, nl = 16, 27, 36

        v = np.random.random(size=(nh,nk,nl))
        
        sigma = [2,1,3]
        
        w = space.blurring(v, sigma)
        
        x = scipy.ndimage.filters.gaussian_filter(v, sigma, mode='nearest')
        
        np.testing.assert_array_almost_equal(w, x, decimal=1)
        
if __name__ == '__main__':
    unittest.main()
