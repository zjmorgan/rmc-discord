#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import displacive, space, scattering

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

        Uxx = np.array([0.5,0.3])
        Uyy = np.array([0.6,0.4])
        Uzz = np.array([0.4,0.6])
        Uyz = np.array([0.0,0.0])
        Uxz = np.array([0.0,0.0])
        Uxy = np.array([0.0,0.0])

        U = np.row_stack((Uxx,Uyy,Uzz,Uyz,Uxz,Uxy))
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, U, False)

        l = 16.26623619623813 # (99.9%) 1/1000
        rx = l*np.sqrt(Uxx)
        ry = l*np.sqrt(Uyy)
        rz = l*np.sqrt(Uzz)

        np.testing.assert_array_equal(Ux.reshape(nu*nv*nw,n_atm) < rx, True)
        np.testing.assert_array_equal(Uy.reshape(nu*nv*nw,n_atm) < ry, True)
        np.testing.assert_array_equal(Uz.reshape(nu*nv*nw,n_atm) < rz, True)

    def test_number(self):

        m = np.arange(10)
        n = displacive.number(m)

        np.testing.assert_array_equal(np.diff(n), 2+m[:-1])

    def test_numbers(self):

        m = np.arange(10)
        n = displacive.numbers(m)

        np.testing.assert_array_equal(np.diff(n), displacive.number(m[1:]))

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

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)

        U = np.row_stack((U11,U22,U33,U23,U13,U12))
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=U)

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

        p = 3

        coeffs = displacive.coefficients(p)

        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L,
                                                    nu, nv, nw, centering='P')

        U_r = displacive.products(Ux, Uy, Uz, p)
        Q_k = displacive.products(Qx, Qy, Qz, p)

        U_k, i_dft = displacive.transform(U_r, H, K, L, nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        # I = displacive.intensity(U_k, Q_k, coeffs, cond, p, i_dft, factors)
        I, F_nuc = displacive.intensity(U_k, Q_k, coeffs, cond, p, i_dft,
                                        factors, subtract=False)

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
        U_r = U_r.reshape(coeffs.shape[0],n_xyz)
        Q_k = Q_k.reshape(coeffs.shape[0],n_hkl)

        U_i, U_j, U_m = U_r[:,i], U_r[:,j], U_r[:,m]
        c_k, c_l, c_n = occupancy[k], occupancy[l], occupancy[n]
        b_k, b_l, b_n = bc[k], bc[l], bc[n]

        exp_iQ_dot_U_m = np.dot(coeffs*U_m.T, Q_k).T
        exp_iQ_dot_U_i = np.dot(coeffs*U_i.T, Q_k).T
        exp_iQ_dot_U_j = np.dot(coeffs*U_j.T, Q_k).T

        I_ref = ((c_n**2*(b_n*b_n.conj()).real*\
                  (exp_iQ_dot_U_m*exp_iQ_dot_U_m.conj()).real).sum(axis=1)\
              + 2*(c_k*c_l*(b_k*b_l.conj()).real*
                   ((exp_iQ_dot_U_i*exp_iQ_dot_U_j.conj()*\
                    np.cos(Qx[:,np.newaxis]*rx_ij+\
                           Qy[:,np.newaxis]*ry_ij+\
                           Qz[:,np.newaxis]*rz_ij)).real+
                    (exp_iQ_dot_U_i*exp_iQ_dot_U_j.conj()*\
                    np.sin(Qx[:,np.newaxis]*rx_ij+\
                           Qy[:,np.newaxis]*ry_ij+\
                           Qz[:,np.newaxis]*rz_ij)).imag)).sum(axis=1))/n_xyz

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

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)

        U = np.row_stack((U11,U22,U33,U23,U13,U12))
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=U)

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

        p = 3

        coeffs = displacive.coefficients(p)

        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L, nu, nv, nw)

        U_r = displacive.products(Ux, Uy, Uz, p)
        Q_k = displacive.products(Qx, Qy, Qz, p)

        U_k, i_dft = displacive.transform(U_r, H, K, L, nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        F, F_nuc, \
        prod, prod_nuc, \
        V_k, V_k_nuc, \
        even, bragg = displacive.structure(U_k, Q_k, coeffs, cond,
                                           p, i_dft, factors)

        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm

        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)

        rx_m = rx[m]
        ry_m = ry[m]
        rz_m = rz[m]

        bc = scattering.length(atm, 1)
        U_r = U_r.reshape(coeffs.shape[0],n_xyz)
        Q_k = Q_k.reshape(coeffs.shape[0],n_hkl)

        U_m = U_r[:,m]
        c_n = occupancy[n]
        b_n = bc[n]

        exp_iQ_dot_U_m = np.dot(coeffs*U_m.T, Q_k).T

        prod_ref = c_n*b_n*exp_iQ_dot_U_m*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                     Qy[:,np.newaxis]*ry_m+\
                                                     Qz[:,np.newaxis]*rz_m))

        F_ref = prod_ref.sum(axis=1)
        prod_ref = prod_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1).flatten()

        np.testing.assert_array_almost_equal(F, F_ref)
        np.testing.assert_array_almost_equal(prod, prod_ref)

        cos_iQ_dot_U_m = np.dot((coeffs*U_m.T)[:,even], Q_k[even,:]).T

        prod_nuc_ref = c_n*b_n*cos_iQ_dot_U_m*\
                       np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                  Qy[:,np.newaxis]*ry_m+\
                                  Qz[:,np.newaxis]*rz_m))

        F_nuc_ref = prod_nuc_ref.sum(axis=1)[cond]
        prod_nuc_ref = prod_nuc_ref.reshape(n_hkl,
                                            nu*nv*nw,
                                            n_atm).sum(axis=1)[cond].flatten()

        np.testing.assert_array_almost_equal(F_nuc, F_nuc_ref)
        np.testing.assert_array_almost_equal(prod_nuc, prod_nuc_ref)

        factors = (c_n*b_n*cos_iQ_dot_U_m).flatten()

        F_nuc_ref = space.bragg(Qx, Qy, Qz, rx, ry, rz, factors, cond)

        np.testing.assert_array_almost_equal(F_nuc, F_nuc_ref)

    def test_parameters(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        nu, nv, nw, n_atm = 2, 3, 4, 2

        np.random.seed(13)

        c = np.array([0.1,0.2])
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=c**2)

        U11, U22, U33, \
        U23, U13, U12 = displacive.parameters(Ux, Uy, Uz, D, n_atm)

        Uiso = displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

        U_sq = Ux**2+Uy**2+Uz**2

        Uiso_ref = np.mean(U_sq.reshape(nu*nv*nw,n_atm), axis=0)/3

        np.testing.assert_array_almost_equal(Uiso, Uiso_ref)

    def test_equivalent(self):

        Uiso = np.array([1.5,1.2])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        U11, U22, U33, U23, U13, U12  = displacive.equivalent(Uiso, D)

        np.testing.assert_array_almost_equal(U11, Uiso)
        np.testing.assert_array_almost_equal(U22, Uiso)
        np.testing.assert_array_almost_equal(U33, Uiso)
        np.testing.assert_array_almost_equal(U23, 0)
        np.testing.assert_array_almost_equal(U13, 0)
        np.testing.assert_array_almost_equal(U12, 0)

    def test_isotropic(self):

        U11 = np.array([0.5,0.3])
        U22 = np.array([0.6,0.4])
        U33 = np.array([0.4,0.6])
        U23 = np.array([0.05,-0.03])
        U13 = np.array([-0.04,0.02])
        U12 = np.array([0.03,-0.02])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        Uiso = displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

        np.testing.assert_array_almost_equal(Uiso, (U11+U22+U33)/3)

    def test_principal(self):

        U11 = np.array([0.4,0.3])
        U22 = np.array([0.5,0.4])
        U33 = np.array([0.6,0.6])
        U23 = np.array([0.0,0.0])
        U13 = np.array([0.0,0.0])
        U12 = np.array([0.0,0.0])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        U1, U2, U3 = displacive.principal(U11, U22, U33, U23, U13, U12, D)

        np.testing.assert_array_almost_equal(U1, U11)
        np.testing.assert_array_almost_equal(U2, U22)
        np.testing.assert_array_almost_equal(U3, U33)

    def test_cartesian(self):

        U11 = np.array([0.4,0.3])
        U22 = np.array([0.5,0.4])
        U33 = np.array([0.6,0.6])
        U23 = np.array([0.0,0.0])
        U13 = np.array([0.0,0.0])
        U12 = np.array([0.0,0.0])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        Uxx, Uyy, Uzz, \
        Uyz, Uxz, Uxy = displacive.cartesian(U11, U22, U33,
                                             U23, U13, U12, D)

        np.testing.assert_array_almost_equal(Uxx, U11)
        np.testing.assert_array_almost_equal(Uyy, U22)
        np.testing.assert_array_almost_equal(Uzz, U33)
        np.testing.assert_array_almost_equal(Uyz, U23)
        np.testing.assert_array_almost_equal(Uxz, U13)
        np.testing.assert_array_almost_equal(Uyz, U12)

if __name__ == '__main__':
    unittest.main()