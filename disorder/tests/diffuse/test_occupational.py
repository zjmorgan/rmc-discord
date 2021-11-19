#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import occupational, space, scattering

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

        atm = np.array(['Fe','Mn'])
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

        A_r = occupational.composition(nu, nv, nw, n_atm, value=occupancy)

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

        A_k, i_dft = occupational.transform(A_r, H, K, L, nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        factors *= T

        I = occupational.intensity(A_k, i_dft, factors)

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

        A_i, A_j, A_m = A_r[i], A_r[j], A_r[m]
        c_k, c_l, c_n = occupancy[k], occupancy[l], occupancy[n]
        b_k, b_l, b_n = bc[k], bc[l], bc[n]
        T_k, T_l, T_n = T[:,k], T[:,l], T[:,n]

        I_ref = ((c_n**2*(b_n*b_n.conj()).real*A_m**2*T_n**2).sum(axis=1)\
              + 2*(c_k*c_l*(b_k*b_l.conj()).real*A_i*A_j*T_k*T_l*\
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

        atm = np.array(['Fe','Mn'])
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

        A_r = occupational.composition(nu, nv, nw, n_atm, value=occupancy)

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

        A_k, i_dft = occupational.transform(A_r, H, K, L, nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        factors *= T

        F, prod = occupational.structure(A_k, i_dft, factors)

        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm

        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)

        rx_m = rx[m]
        ry_m = ry[m]
        rz_m = rz[m]

        bc = scattering.length(atm, 1)
        T = T.reshape(n_hkl,n_atm)

        A_m = A_r[m]
        c_n = occupancy[n]
        b_n = bc[n]
        T_n = T[:,n]

        prod_ref = (c_n*b_n*A_m*T_n*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                               Qy[:,np.newaxis]*ry_m+\
                                               Qz[:,np.newaxis]*rz_m)))

        F_ref = prod_ref.sum(axis=1)
        prod_ref = prod_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1).flatten()

        np.testing.assert_array_almost_equal(F, F_ref)
        np.testing.assert_array_almost_equal(prod, prod_ref)
        
if __name__ == '__main__':
    unittest.main()