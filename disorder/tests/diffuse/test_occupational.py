#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.diffuse import occupational

class test_occupational(unittest.TestCase):
    
    def test_composition(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        c = 0.7
        A_r = occupational.composition(nu, nv, nw, n_atm, value=c)
        
        delta = c*(1+A_r)
        sigma = 2*delta-1
        
        np.testing.assert_array_almost_equal(sigma, (2*(A_r>0)-1))
        
        np.random.seed(13)

        u = np.random.rand(nu,nv,nw,n_atm)

        np.testing.assert_array_almost_equal(delta, u.flatten()<=c)
        
        c = np.array([0.6,0.4])
        A_r = occupational.composition(nu, nv, nw, n_atm, value=c)
        
        delta = (c*(1+A_r.reshape(nu,nv,nw,n_atm))).flatten()
        sigma = 2*delta-1
        
        np.testing.assert_array_almost_equal(sigma, (2*(A_r>0)-1))
        
    def test_transform(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        c = 0.7
        A_r = occupational.composition(nu, nv, nw, n_atm, value=c)
        
        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))
        
        A_k, i_dft = occupational.transform(A_r, H, K, L, nu, nv, nw, n_atm)
        
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