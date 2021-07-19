#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.diffuse import displacive

class test_displacive(unittest.TestCase):
    
    def test_expansion(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        c = 0.1
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=c**2)
        U = np.sqrt(Ux**2+Uy**2+Uz**2)
        
        np.testing.assert_array_almost_equal(U, c*np.ones(nu*nv*nw*n_atm))
        
        np.random.seed(13)

        u, v = np.random.rand(nu,nv,nw,n_atm), np.random.rand(nu,nv,nw,n_atm)
        
        theta = np.mod(np.arctan2(Uy,Ux), 2*np.pi)
        phi = np.arccos(Uz/U) 

        np.testing.assert_array_almost_equal(theta, 2*np.pi*u.flatten())
        np.testing.assert_array_almost_equal(phi, np.arccos(1-2*v.flatten()))
    
        np.random.seed(13)
        
        c = np.array([0.1,0.2])
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=c**2)
        U = np.sqrt(Ux**2+Uy**2+Uz**2)
        
        V = np.ones((nu,nv,nw,n_atm))*c
        np.testing.assert_array_almost_equal(U, V.flatten())
        
        np.random.seed(13)

        u, v = np.random.rand(nu,nv,nw,n_atm), np.random.rand(nu,nv,nw,n_atm)
        
        theta = np.mod(np.arctan2(Uy,Ux), 2*np.pi)
        phi = np.arccos(Uz/U) 

        np.testing.assert_array_almost_equal(theta, 2*np.pi*u.flatten())
        np.testing.assert_array_almost_equal(phi, np.arccos(1-2*v.flatten()))
        
    def test_number(self):
        
        m = np.arange(10)
        n = displacive.number(m)
        
        np.testing.assert_array_equal(np.diff(n), 2+np.arange(9))

    def test_factorial(self):

        self.assertEqual(displacive.factorial(0), 1)
        self.assertEqual(displacive.factorial(1), 1)
        self.assertEqual(displacive.factorial(6), 720)
        self.assertEqual(displacive.factorial(10), 3628800)
        
    def test_coefficients(self):
        
        p = 5
        
        coeffs = displacive.coefficients(p)
        
        numbers = displacive.number(np.arange(p+1))
        
        self.assertEqual(coeffs.size, numbers.sum())        
        self.assertEqual(coeffs[0], 1)
        
        even = np.isreal(coeffs)
        odd = ~np.isreal(coeffs)
        
        end = np.cumsum(numbers)
        start = end-numbers

        self.assertTrue(even[start[0]:end[0]].all()) 
        self.assertTrue(odd[start[1]:end[1]].all()) 
        self.assertTrue(even[start[2]:end[2]].all()) 
        self.assertTrue(odd[start[3]:end[3]].all()) 
        self.assertTrue(even[start[4]:end[4]].all()) 
        self.assertTrue(odd[start[5]:end[5]].all()) 
        
        self.assertAlmostEqual(coeffs[0], 1)
        
        self.assertAlmostEqual(coeffs[1], 1j)
        self.assertAlmostEqual(coeffs[2], 1j)
        self.assertAlmostEqual(coeffs[3], 1j)
        
        self.assertAlmostEqual(coeffs[4], -0.5)
        self.assertAlmostEqual(coeffs[5], -1)
        self.assertAlmostEqual(coeffs[6], -0.5)
        self.assertAlmostEqual(coeffs[7], -1)
        self.assertAlmostEqual(coeffs[8], -1)
        self.assertAlmostEqual(coeffs[9], -0.5)
        
    def test_products(self):
        
        p = 5
        
        np.random.seed(13)

        n = 3
        Vx = np.random.random(n)
        Vy = np.random.random(n)
        Vz = np.random.random(n)
        
        V_r = displacive.products(Vx, Vy, Vz, p)
                
        V_r = V_r.reshape(displacive.number(np.arange(p+1)).sum(),n)

        np.testing.assert_array_almost_equal(V_r[0,:], Vx**0*Vy**0*Vz**0)
        
        np.testing.assert_array_almost_equal(V_r[1,:], Vx**1*Vy**0*Vz**0)
        np.testing.assert_array_almost_equal(V_r[2,:], Vx**0*Vy**1*Vz**0)
        np.testing.assert_array_almost_equal(V_r[3,:], Vx**0*Vy**0*Vz**1)
        
        np.testing.assert_array_almost_equal(V_r[4,:], Vx**2*Vy**0*Vz**0)
        np.testing.assert_array_almost_equal(V_r[5,:], Vx**1*Vy**1*Vz**0)
        np.testing.assert_array_almost_equal(V_r[6,:], Vx**0*Vy**2*Vz**0)
        np.testing.assert_array_almost_equal(V_r[7,:], Vx**1*Vy**0*Vz**1)
        np.testing.assert_array_almost_equal(V_r[8,:], Vx**0*Vy**1*Vz**1)
        np.testing.assert_array_almost_equal(V_r[9,:], Vx**0*Vy**0*Vz**2)

    def test_transform(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        c = 0.1
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=c)
        
        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))
        
        p = 2
        U_r = displacive.products(Ux, Uy, Uz, p)
        
        n_prod = U_r.shape[0] // (nu*nv*nw*n_atm)
        
        U_k, i_dft = displacive.transform(U_r, H, K, L, nu, nv, nw, n_atm)
        
        U_r = U_r.reshape(n_prod,nu,nv,nw,n_atm)
        
        U_k = U_k.reshape(n_prod,nu,nv,nw,n_atm)
        
        n_uvw = nu*nv*nw
        
        np.testing.assert_array_almost_equal(U_k[:,0,0,0,:], \
                                             np.mean(U_r, axis=(1,2,3))*n_uvw)
        
        w = i_dft % nw
        v = i_dft // nw % nv
        u = i_dft // nw // nv % nu
        
        np.testing.assert_array_equal(u, np.mod(H, nu))
        np.testing.assert_array_equal(v, np.mod(K, nv))
        np.testing.assert_array_equal(w, np.mod(L, nw))  
        
if __name__ == '__main__':
    unittest.main()