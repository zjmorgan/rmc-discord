#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import displacive, occupational
from disorder.diffuse import nonmagnetic, space, scattering

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

        U_k, A_k, i_dft = nonmagnetic.transform(U_r, A_r, H, K, L,
                                                nu, nv, nw, n_atm)

        A_r = np.tile(A_r, n_prod)

        U_r = U_r.reshape(n_prod,nu,nv,nw,n_atm)
        A_r = A_r.reshape(n_prod,nu,nv,nw,n_atm)

        U_k = U_k.reshape(n_prod,nu,nv,nw,n_atm)
        A_k = A_k.reshape(n_prod,nu,nv,nw,n_atm)

        n_uvw = nu*nv*nw

        V_r = U_r*A_r

        np.testing.assert_array_almost_equal(U_k[:,0,0,0,:], \
                                             np.mean(U_r, axis=(1,2,3))*n_uvw)
        np.testing.assert_array_almost_equal(A_k[:,0,0,0,:], \
                                             np.mean(V_r, axis=(1,2,3))*n_uvw)

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

        p = 3

        coeffs = displacive.coefficients(p)

        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L,
                                                    nu, nv, nw, centering='P')

        U_r = displacive.products(Ux, Uy, Uz, p)
        Q_k = displacive.products(Qx, Qy, Qz, p)

        U_k, A_k, i_dft = nonmagnetic.transform(U_r, A_r, H, K, L,
                                                nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        I, F_nuc = nonmagnetic.intensity(U_k, A_k, Q_k, coeffs, cond, p, i_dft,
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
        A_i, A_j, A_m = A_r[i], A_r[j], A_r[m]
        c_k, c_l, c_n = occupancy[k], occupancy[l], occupancy[n]
        b_k, b_l, b_n = bc[k], bc[l], bc[n]

        exp_iQ_dot_V_m = np.dot(coeffs*(U_m+A_m*U_m).T, Q_k).T
        exp_iQ_dot_V_i = np.dot(coeffs*(U_i+A_i*U_i).T, Q_k).T
        exp_iQ_dot_V_j = np.dot(coeffs*(U_j+A_j*U_j).T, Q_k).T

        I_ref = ((c_n**2*(b_n*b_n.conj()).real*\
                  (exp_iQ_dot_V_m*exp_iQ_dot_V_m.conj()).real).sum(axis=1)\
              + 2*(c_k*c_l*(b_k*b_l.conj()).real*
                   ((exp_iQ_dot_V_i*exp_iQ_dot_V_j.conj()*\
                    np.cos(Qx[:,np.newaxis]*rx_ij+\
                           Qy[:,np.newaxis]*ry_ij+\
                           Qz[:,np.newaxis]*rz_ij)).real+
                    (exp_iQ_dot_V_i*exp_iQ_dot_V_j.conj()*\
                    np.sin(Qx[:,np.newaxis]*rx_ij+\
                           Qy[:,np.newaxis]*ry_ij+\
                           Qz[:,np.newaxis]*rz_ij)).imag
                        )).sum(axis=1))/n_xyz

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

        p = 3

        coeffs = displacive.coefficients(p)

        H_nuc, K_nuc, L_nuc, cond = space.condition(H, K, L,
                                                    nu, nv, nw, centering='P')

        U_r = displacive.products(Ux, Uy, Uz, p)
        Q_k = displacive.products(Qx, Qy, Qz, p)

        U_k, A_k, i_dft = nonmagnetic.transform(U_r, A_r, H, K, L,
                                                nu, nv, nw, n_atm)

        factors = space.prefactors(scattering_length, phase_factor, occupancy)

        F, F_nuc, \
        prod, prod_nuc, \
        V_k, V_k_nuc, \
        even, bragg = nonmagnetic.structure(U_k, A_k, Q_k, coeffs, cond,
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
        A_m = A_r[m]
        c_n = occupancy[n]
        b_n = bc[n]

        exp_iQ_dot_V_m = np.dot(coeffs*(U_m+A_m*U_m).T, Q_k).T

        prod_ref = (c_n*b_n*exp_iQ_dot_V_m*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                      Qy[:,np.newaxis]*ry_m+\
                                                      Qz[:,np.newaxis]*rz_m)))

        F_ref = prod_ref.sum(axis=1)
        prod_ref = prod_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1).flatten()

        np.testing.assert_array_almost_equal(F, F_ref)
        np.testing.assert_array_almost_equal(prod, prod_ref)

if __name__ == '__main__':
    unittest.main()