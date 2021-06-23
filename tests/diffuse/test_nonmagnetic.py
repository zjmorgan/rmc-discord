#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.diffuse import displacive, occupational
from disorder.diffuse import nonmagnetic

class test_nonmagnetic(unittest.TestCase):

    def test_transform(self):
        
        nu, nv, nw, n_atm = 2, 3, 4, 2
        
        np.random.seed(13)
        
        c = 0.1
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=c)
        
        c = 0.7
        A_r = occupational.composition(nu, nv, nw, n_atm, value=c)
        
        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))
        
        p = 2
        U_r = displacive.products(Ux, Uy, Uz, p)
                
        n_prod = U_r.shape[0] // (nu*nv*nw*n_atm)
                
        U_k, A_k, i_dft = nonmagnetic.transform(U_r, 
                                                A_r, 
                                                H, 
                                                K, 
                                                L, 
                                                nu, 
                                                nv, 
                                                nw, 
                                                n_atm)
        
        A_r = np.tile(A_r, n_prod)

        U_r = U_r.reshape(n_prod,nu,nv,nw,n_atm)
        A_r = A_r.reshape(n_prod,nu,nv,nw,n_atm)
        
        U_k = U_k.reshape(n_prod,nu,nv,nw,n_atm)
        A_k = A_k.reshape(n_prod,nu,nv,nw,n_atm)
        
        n_uvw = nu*nv*nw
        
        np.testing.assert_array_almost_equal(U_k[:,0,0,0,:], \
                                             np.mean(U_r, axis=(1,2,3))*n_uvw)
        np.testing.assert_array_almost_equal(A_k[:,0,0,0,:], \
                                             np.mean(U_r*A_r, axis=(1,2,3))*n_uvw)
        
        w = i_dft % nw
        v = i_dft // nw % nv
        u = i_dft // nw // nv % nu
        
        np.testing.assert_array_equal(u, np.mod(H, nu))
        np.testing.assert_array_equal(v, np.mod(K, nv))
        np.testing.assert_array_equal(w, np.mod(L, nw))  
        
if __name__ == '__main__':
    unittest.main()