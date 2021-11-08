#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, scattering

class test_space(unittest.TestCase):
    
    def test_reciprocal(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
        
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        
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
            
    def test_cell(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        h, k, l = -3, 1, 2
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        np.testing.assert_array_almost_equal(np.exp(1j*(Qx*Rx+Qy*Ry+Qz*Rz)), 1)
        
    def test_real(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 3, 4
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe', 'Co'])
        u, v, w = np.array([0,0.2]), np.array([0,0.3]), np.array([0,0.4])
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        h, k, l = -3, 2, 5
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
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
        
    def test_indices(self):
        
        u = np.random.random(13)
        
        mask = u < 0.5
        
        i_mask, i_unmask = space.indices(mask)
        
        indices = np.sort(np.concatenate((i_mask, i_unmask)))
        
        np.testing.assert_array_equal(indices, np.arange(13))
        
    def test_prefactors(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-2,2], 5
        k_range, nk = [-3,3], 7
        l_range, nl = [-4,4], 9
        
        nu, nv, nw = 2, 3, 4
        
        u = np.array([0,1])
        v = np.array([0,1])
        w = np.array([0,1])
        
        atm = np.array(['Fe', 'Mn'])
        occupancy = np.array([0.75,0.5])
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        h, k, l = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                              np.linspace(k_range[0],k_range[1],nk), 
                              np.linspace(l_range[0],l_range[1],nl), 
                              indexing='ij')
         
        h, k, l = h.flatten(), k.flatten(), l.flatten()
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
                    
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
                
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
                                    
        scattering_length = scattering.length(atm, Q.size)
                    
        factors = space.prefactors(scattering_length, phase_factor, occupancy)
        
        bc = scattering.length(atm, 1)
        
        np.testing.assert_array_almost_equal(factors[0::2], bc[0]*occupancy[0])
        np.testing.assert_array_almost_equal(factors[1::2], bc[1]*occupancy[1])
        
    def test_intensity(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-1,1], 5
        k_range, nk = [0,2], 11
        l_range, nl = [-1,0], 5
        
        nu, nv, nw, n_atm = 2, 5, 4, 2
        
        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])
        
        atm = np.array(['Fe', 'Mn'])
        occupancy = np.array([0.75,0.5])
        
        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])
        
        T = space.debye_waller(h_range, k_range, l_range, nh, nk, nl, 
                               U11, U22, U33, U23, U13, U12, a_, b_, c_)
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        delta_r = np.ones((nu,nv,nw,n_atm)).flatten()
        
        index_parameters = space.mapping(h_range, k_range, l_range,
                                         nh, nk, nl, nu, nv, nw)
         
        h, k, l, H, K, L, indices, inverses, operators = index_parameters
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
        
        scattering_length = scattering.length(atm, Q.size)
        
        delta_k, i_dft = space.transform(delta_r, H, K, L, nu, nv, nw, n_atm)
        
        factors = space.prefactors(scattering_length, phase_factor, occupancy)
        
        factors *= T
        
        I = space.intensity(delta_k, i_dft, factors)
        
        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm
        
        i, j = np.triu_indices(n_xyz, 1)
        k, l = np.mod(i, n_atm), np.mod(j, n_atm)
        
        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)
        
        rx_ij = rx[j]-rx[i]
        ry_ij = ry[j]-ry[i]
        rz_ij = rz[j]-rz[i]
                
        bc = scattering.length(atm, 1)
        T = T.reshape(n_hkl,n_atm)
        
        delta_i, delta_j, delta_m = delta_r[i], delta_r[j], delta_r[m]
        c_k, c_l, c_n = occupancy[k], occupancy[l], occupancy[n]
        b_k, b_l, b_n = bc[k], bc[l], bc[n]
        T_k, T_l, T_n = T[:,k], T[:,l], T[:,n]
        
        I_ref = ((c_n**2*(b_n*b_n.conj()).real*delta_m**2*T_n**2).sum(axis=1)\
              + 2*(c_k*c_l*(b_k*b_l.conj()).real*delta_i*delta_j*T_k*T_l*\
                   np.cos(Qx[:,np.newaxis]*rx_ij+\
                          Qy[:,np.newaxis]*ry_ij+\
                          Qz[:,np.newaxis]*rz_ij)).sum(axis=1))/n_xyz
           
        np.testing.assert_array_almost_equal(I, I_ref)
        
    def test_structure(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-1,1], 5
        k_range, nk = [0,2], 11
        l_range, nl = [-1,0], 5
        
        nu, nv, nw, n_atm = 2, 5, 4, 2
        
        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])
        
        atm = np.array(['Fe', 'Mn'])
        occupancy = np.array([0.75,0.5])
        
        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])
        
        T = space.debye_waller(h_range, k_range, l_range, nh, nk, nl, 
                               U11, U22, U33, U23, U13, U12, a_, b_, c_)
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        delta_r = np.ones((nu,nv,nw,n_atm)).flatten()
        
        index_parameters = space.mapping(h_range, k_range, l_range,
                                         nh, nk, nl, nu, nv, nw)
         
        h, k, l, H, K, L, indices, inverses, operators = index_parameters
        
        Qh, Qk, Ql = crystal.vector(h, k, l, B)
        
        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        ix, iy, iz = space.cell(nu, nv, nw, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        phase_factor = scattering.phase(Qx, Qy, Qz, ux, uy, uz)
        
        scattering_length = scattering.length(atm, Q.size)
        
        delta_k, i_dft = space.transform(delta_r, H, K, L, nu, nv, nw, n_atm)
        
        factors = space.prefactors(scattering_length, phase_factor, occupancy)
        
        factors *= T
        
        F, prod = space.structure(delta_k, i_dft, factors)
        
        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm
        
        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)
        
        rx_m = rx[m]
        ry_m = ry[m]
        rz_m = rz[m]
                
        bc = scattering.length(atm, 1)
        T = T.reshape(n_hkl,n_atm)
        
        delta_m = delta_r[m]
        c_n = occupancy[n]
        b_n = bc[n]
        T_n = T[:,n]
        
        prod_ref = (c_n*b_n*delta_m*T_n*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                   Qy[:,np.newaxis]*ry_m+\
                                                   Qz[:,np.newaxis]*rz_m)))
            
        F_ref = prod_ref.sum(axis=1)
        prod_ref = prod_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1).flatten()
           
        np.testing.assert_array_almost_equal(F, F_ref)
        np.testing.assert_array_almost_equal(prod, prod_ref)

    def test_debye_waller(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants
                
        h_range, nh = [-2,2], 9
        k_range, nk = [-3,3], 13
        l_range, nl = [-4,4], 17
        
        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])
        
        T = space.debye_waller(h_range, k_range, l_range, nh, nk, nl, 
                               U11, U22, U33, U23, U13, U12, a_, b_, c_)
        
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        
        h, k, l = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                              np.linspace(k_range[0],k_range[1],nk), 
                              np.linspace(l_range[0],l_range[1],nl), 
                              indexing='ij')
         
        h, k, l = h.flatten(), k.flatten(), l.flatten()

        Qh, Qk, Ql = crystal.vector(h, k, l, B)

        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)
        
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)
        
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy = [], [], [], [], [], []
        for i in range(2):
            U = np.array([[U11[i],U12[i],U13[i]],
                          [U12[i],U22[i],U23[i]],
                          [U13[i],U23[i],U33[i]]])
            Up = np.dot(np.dot(D, U), D.T)
            Uxx.append(Up[0,0])
            Uyy.append(Up[1,1])
            Uzz.append(Up[2,2])
            Uyz.append(Up[1,2])
            Uxz.append(Up[0,2])
            Uxy.append(Up[0,1])
        
        Uxx, Uyy, Uzz = np.array(Uxx), np.array(Uyy), np.array(Uzz)
        Uyz, Uxz, Uxy = np.array(Uyz), np.array(Uxz), np.array(Uxy)

        dw_factors = np.exp(-0.5*(Uxx*(Qx*Qx)[:,np.newaxis]+\
                                  Uyy*(Qy*Qy)[:,np.newaxis]+\
                                  Uzz*(Qz*Qz)[:,np.newaxis])-\
                                 (Uyz*(Qy*Qz)[:,np.newaxis]+\
                                  Uxz*(Qx*Qz)[:,np.newaxis]+\
                                  Uxy*(Qx*Qy)[:,np.newaxis])).flatten()
        
        np.testing.assert_array_almost_equal(T, dw_factors)
        
    def test_condition(self):
    
        h_range, nh = [-2,2], 9
        k_range, nk = [-3,3], 13
        l_range, nl = [-4,4], 17
        
        nu, nv, nw = 2, 5, 4
        
        h, k, l = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                              np.linspace(k_range[0],k_range[1],nk), 
                              np.linspace(l_range[0],l_range[1],nl), 
                              indexing='ij')
         
        h, k, l = h.flatten(), k.flatten(), l.flatten()
        
        H, K, L = (h*nu).astype(int), (k*nv).astype(int), (l*nw).astype(int)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'P')

        np.testing.assert_array_equal(np.mod(h[cond], 1), 0)
        np.testing.assert_array_equal(np.mod(k[cond], 1), 0)
        np.testing.assert_array_equal(np.mod(l[cond], 1), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'I')
        
        np.testing.assert_array_equal(np.mod(h[cond]+k[cond]+l[cond], 2), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'F')
        
        np.testing.assert_array_equal(np.mod(h[cond]+k[cond], 2), 0)
        np.testing.assert_array_equal(np.mod(k[cond]+l[cond], 2), 0)
        np.testing.assert_array_equal(np.mod(l[cond]+h[cond], 2), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, 
                                                    nu, nv, nw, 'R(obv)')
        
        np.testing.assert_array_equal(np.mod(-h[cond]+k[cond]+l[cond], 3), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, 
                                                    nu, nv, nw, 'R(rev)')
        
        np.testing.assert_array_equal(np.mod(h[cond]-k[cond]+l[cond], 3), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'C')
        
        np.testing.assert_array_equal(np.mod(h[cond]+k[cond], 2), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'A')
        
        np.testing.assert_array_equal(np.mod(k[cond]+l[cond], 2), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'B')
        
        np.testing.assert_array_equal(np.mod(l[cond]+h[cond], 2), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'H')
        
        np.testing.assert_array_equal(np.mod(h[cond]-k[cond], 3), 0)
        
        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw, 'D')
        
        np.testing.assert_array_equal(np.mod(h[cond]+k[cond]+l[cond], 3), 0)
                
    def test_mapping(self):
        
        h_range, nh = [-2,2], 9
        k_range, nk = [-3,3], 13
        l_range, nl = [-4,4], 17
        
        nu, nv, nw = 4, 5, 6
        
        array = np.random.random((nh,nk,nl))

        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw)
    
        h, k, l, H, K, L, index, reverses, symops = mapping_params
                
        np.testing.assert_array_almost_equal(np.mod(h[H % nu == 0], 1), 0)
        np.testing.assert_array_almost_equal(np.mod(k[K % nv == 0], 1), 0)
        np.testing.assert_array_almost_equal(np.mod(l[L % nw == 0], 1), 0)
        
        data = array[::,::,::].flatten()
        
        np.testing.assert_array_equal(index, reverses)
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        self.assertEqual(symops, u'x,y,z')
        
        T = np.array([[1,-1,0],[1,1,0],[0,0,1]])
        
        mapping_params = space.mapping(h_range, k_range, l_range,
                                       nh, nk, nl, nu, nv, nw, T=T)
    
        h, k, l, H, K, L, index, reverses, symops = mapping_params
                
        np.testing.assert_array_almost_equal(np.mod(h[H % nu == 0], 1), 0)
        np.testing.assert_array_almost_equal(np.mod(k[K % nv == 0], 1), 0)
        np.testing.assert_array_almost_equal(np.mod(l[L % nw == 0], 1), 0)
        
        h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                                 np.linspace(k_range[0],k_range[1],nk), 
                                 np.linspace(l_range[0],l_range[1],nl), 
                                 indexing='ij')
        
        np.testing.assert_array_almost_equal(h+k, 2*h_.flatten())
        np.testing.assert_array_almost_equal(k-h, 2*k_.flatten())
        np.testing.assert_array_almost_equal(l, l_.flatten())
                
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='-1')
    
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data += array[::-1,::-1,::-1].flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='2/m')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data += (array[::-1,::,::-1]+array[::,::-1,::]).flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='mmm')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data += (array[::-1,::-1,::]+array[::,::-1,::-1]+
                 array[::,::,::-1]+array[::-1,::,::]).flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        h_range, nh = [-3,3], 13
        k_range, nk = [-3,3], 13
        l_range, nl = [-4,4], 17
        
        nu, nv, nw = 5, 5, 4
        
        array = np.random.random((nh,nk,nl))
                
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='4/m')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data = (array[::,::,::]+array[::-1,::-1,::]+
                array[::-1,::-1,::-1]+array[::,::,::-1]+
                np.swapaxes(array[::,::-1,::], 0, 1)+
                np.swapaxes(array[::-1,::,::], 0, 1)+
                np.swapaxes(array[::-1,::,::-1], 0, 1)+
                np.swapaxes(array[::,::-1,::-1], 0, 1)).flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                        nh, nk, nl, nu, nv, nw, laue='4/mmm')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data += (array[::-1,::,::-1]+array[::,::-1,::-1]+
                 array[::,::-1,::]+array[::-1,::,::]+
                 np.swapaxes(array[::,::,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::-1,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::-1,::], 0, 1)+
                 np.swapaxes(array[::,::,::], 0, 1)).flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        h_range, nh = [-3,3], 13
        k_range, nk = [-3,3], 13
        l_range, nl = [-3,3], 13
        
        nu, nv, nw = 5, 5, 5
        
        array = np.random.random((nh,nk,nl))
                
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='m-3')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data = (array[::,::,::]+array[::-1,::-1,::]+
                array[::-1,::,::-1]+array[::,::-1,::-1]+
                array[::-1,::-1,::-1]+array[::,::,::-1]+
                array[::,::-1,::]+array[::-1,::,::]+
                np.swapaxes(np.swapaxes(array[::,::,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::-1,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::-1,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::,::-1], 1, 2), 0, 1)+   
                np.swapaxes(np.swapaxes(array[::,::-1,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::-1,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::,::], 0, 1), 1, 2)
                ).flatten()
                        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        mapping_params = space.mapping(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='m-3m')
        
        h, k, l, H, K, L, index, reverses, symops = mapping_params
        
        data += (np.swapaxes(array[::,::,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::-1,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::,::], 0, 1)+
                 np.swapaxes(array[::,::-1,::], 0, 1)+
                 np.swapaxes(array[::,::-1,::], 1, 2)+
                 np.swapaxes(array[::-1,::,::], 1, 2)+
                 np.swapaxes(array[::-1,::-1,::-1], 1, 2)+
                 np.swapaxes(array[::,::,::-1], 1, 2)+
                 np.swapaxes(array[::-1,::,::], 2, 0)+
                 np.swapaxes(array[::,::-1,::], 2, 0)+
                 np.swapaxes(array[::,::,::-1], 2, 0)+
                 np.swapaxes(array[::-1,::-1,::-1], 2, 0)+
                 np.swapaxes(array[::,::,::], 0, 1)+ 
                 np.swapaxes(array[::,::-1,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::,::-1], 0, 1)+
                 np.swapaxes(array[::-1,::-1,::], 0, 1)+
                 np.swapaxes(array[::-1,::,::-1], 1, 2)+
                 np.swapaxes(array[::,::-1,::-1], 1, 2)+
                 np.swapaxes(array[::,::,::], 1, 2)+
                 np.swapaxes(array[::-1,::-1,::], 1, 2)+
                 np.swapaxes(array[::,::-1,::-1], 2, 0)+
                 np.swapaxes(array[::-1,::,::-1], 2, 0)+
                 np.swapaxes(array[::-1,::-1,::], 2, 0)+
                 np.swapaxes(array[::,::,::], 2, 0)).flatten()

        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
    def test_reduced(self):
        
        h_range, nh = [-2,2], 33
        k_range, nk = [-3,3], 49
        l_range, nl = [-4,4], 65
                
        array = np.random.random((nh,nk,nl))

        nu, nv, nw = 4, 5, 6
        
        reduced_params = space.reduced(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw)
    
        index, reverses, symops, Nu, Nv, Nw = reduced_params
                        
        data = array[::,::,::].flatten()
        
        np.testing.assert_array_equal(index, reverses)
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
        self.assertEqual(symops, u'x,y,z')
        
        self.assertEqual(Nu, 2*nu)
        self.assertEqual(Nv, nv)
        self.assertEqual(Nw, nw)
        
        nu, nv, nw = 8, 4, 10
                
        T = np.array([[1,-1,0],[1,1,0],[0,0,1]])
        
        reduced_params = space.reduced(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw)
    
        index, reverses, symops, Nu, Nv, Nw = reduced_params
                
        np.testing.assert_array_equal(index, reverses)
        np.testing.assert_array_almost_equal(data[index][reverses], data)

        self.assertEqual(Nu, nu)
        self.assertEqual(Nv, 2*nv)
        self.assertEqual(Nw, nw)
        
        h_range, nh = [-2,2], 33
        k_range, nk = [-2,2], 33
        l_range, nl = [-2,2], 33
        
        array = np.random.random((nh,nk,nl))
        
        nu, nv, nw = 4, 4, 4
                
        data = (array[::,::,::]+array[::-1,::-1,::]+
                array[::-1,::,::-1]+array[::,::-1,::-1]+
                array[::-1,::-1,::-1]+array[::,::,::-1]+
                array[::,::-1,::]+array[::-1,::,::]+
                np.swapaxes(np.swapaxes(array[::,::,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::-1,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::-1,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::-1], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::,::,::-1], 1, 2), 0, 1)+   
                np.swapaxes(np.swapaxes(array[::,::-1,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::,::], 1, 2), 0, 1)+
                np.swapaxes(np.swapaxes(array[::-1,::-1,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::,::-1], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::,::-1,::], 0, 1), 1, 2)+
                np.swapaxes(np.swapaxes(array[::-1,::,::], 0, 1), 1, 2)+
                np.swapaxes(array[::,::,::-1], 0, 1)+
                np.swapaxes(array[::-1,::-1,::-1], 0, 1)+
                np.swapaxes(array[::-1,::,::], 0, 1)+
                np.swapaxes(array[::,::-1,::], 0, 1)+
                np.swapaxes(array[::,::-1,::], 1, 2)+
                np.swapaxes(array[::-1,::,::], 1, 2)+
                np.swapaxes(array[::-1,::-1,::-1], 1, 2)+
                np.swapaxes(array[::,::,::-1], 1, 2)+
                np.swapaxes(array[::-1,::,::], 2, 0)+
                np.swapaxes(array[::,::-1,::], 2, 0)+
                np.swapaxes(array[::,::,::-1], 2, 0)+
                np.swapaxes(array[::-1,::-1,::-1], 2, 0)+
                np.swapaxes(array[::,::,::], 0, 1)+ 
                np.swapaxes(array[::,::-1,::-1], 0, 1)+
                np.swapaxes(array[::-1,::,::-1], 0, 1)+
                np.swapaxes(array[::-1,::-1,::], 0, 1)+
                np.swapaxes(array[::-1,::,::-1], 1, 2)+
                np.swapaxes(array[::,::-1,::-1], 1, 2)+
                np.swapaxes(array[::,::,::], 1, 2)+
                np.swapaxes(array[::-1,::-1,::], 1, 2)+
                np.swapaxes(array[::,::-1,::-1], 2, 0)+
                np.swapaxes(array[::-1,::,::-1], 2, 0)+
                np.swapaxes(array[::-1,::-1,::], 2, 0)+
                np.swapaxes(array[::,::,::], 2, 0)).flatten()
        
        reduced_params = space.reduced(h_range, k_range, l_range, 
                                       nh, nk, nl, nu, nv, nw, laue='m-3m')
    
        index, reverses, symops, Nu, Nv, Nw = reduced_params
        
        np.testing.assert_array_almost_equal(data[index][reverses], data)
        
if __name__ == '__main__':
    unittest.main()