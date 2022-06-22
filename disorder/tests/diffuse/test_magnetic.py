#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import tables, crystal
from disorder.diffuse import magnetic, space, scattering

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

        c = np.array([0.6,0.4])
        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm, value=c, fixed=False)
        S = np.sqrt(Sx**2+Sy**2+Sz**2)

        np.testing.assert_array_equal(S.reshape(nu*nv*nw,n_atm) < c, True)

    def test_transform(self):

        nu, nv, nw, n_atm = 2, 3, 4, 2

        np.random.seed(13)

        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)

        H = np.random.randint(0, 4*nu, size=(16))
        K = np.random.randint(0, 5*nv, size=(16))
        L = np.random.randint(0, 6*nw, size=(16))

        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, Sy, Sz, H, K, L,
                                                     nu, nv, nw, n_atm)

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

        atm = np.array(['Fe3+','Mn3+'])
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

        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)

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

        form_factor = magnetic.form(Q, atm, g=2)

        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, Sy, Sz, H, K, L,
                                                     nu, nv, nw, n_atm)

        factors = space.prefactors(form_factor, phase_factor, occupancy)

        factors *= T

        I = magnetic.intensity(Qx_norm, Qy_norm, Qz_norm, \
                               Sx_k, Sy_k, Sz_k, i_dft, factors)

        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm

        i, j = np.triu_indices(n_xyz, 1)
        k, l = np.mod(i, n_atm), np.mod(j, n_atm)

        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)

        rx_ij = rx[j]-rx[i]
        ry_ij = ry[j]-ry[i]
        rz_ij = rz[j]-rz[i]

        mf = form_factor.reshape(n_hkl,n_atm)
        T = T.reshape(n_hkl,n_atm)

        Sx_i, Sx_j, Sx_m = Sx[i], Sx[j], Sx[m]
        Sy_i, Sy_j, Sy_m = Sy[i], Sy[j], Sy[m]
        Sz_i, Sz_j, Sz_m = Sz[i], Sz[j], Sz[m]
        c_k, c_l, c_n = occupancy[k], occupancy[l], occupancy[n]
        f_k, f_l, f_n = mf[:,k], mf[:,l], mf[:,n]
        T_k, T_l, T_n = T[:,k], T[:,l], T[:,n]

        Q_norm_dot_S_i = Qx_norm[:,np.newaxis]*Sx_i\
                       + Qy_norm[:,np.newaxis]*Sy_i\
                       + Qz_norm[:,np.newaxis]*Sz_i
        Q_norm_dot_S_j = Qx_norm[:,np.newaxis]*Sx_j\
                       + Qy_norm[:,np.newaxis]*Sy_j\
                       + Qz_norm[:,np.newaxis]*Sz_j
        Q_norm_dot_S_m = Qx_norm[:,np.newaxis]*Sx_m\
                       + Qy_norm[:,np.newaxis]*Sy_m\
                       + Qz_norm[:,np.newaxis]*Sz_m

        Sx_perp_i = Sx_i-(Q_norm_dot_S_i)*Qx_norm[:,np.newaxis]
        Sx_perp_j = Sx_j-(Q_norm_dot_S_j)*Qx_norm[:,np.newaxis]
        Sx_perp_m = Sx_m-(Q_norm_dot_S_m)*Qx_norm[:,np.newaxis]

        Sy_perp_i = Sy_i-(Q_norm_dot_S_i)*Qy_norm[:,np.newaxis]
        Sy_perp_j = Sy_j-(Q_norm_dot_S_j)*Qy_norm[:,np.newaxis]
        Sy_perp_m = Sy_m-(Q_norm_dot_S_m)*Qy_norm[:,np.newaxis]

        Sz_perp_i = Sz_i-(Q_norm_dot_S_i)*Qz_norm[:,np.newaxis]
        Sz_perp_j = Sz_j-(Q_norm_dot_S_j)*Qz_norm[:,np.newaxis]
        Sz_perp_m = Sz_m-(Q_norm_dot_S_m)*Qz_norm[:,np.newaxis]

        I_ref = ((c_n**2*(f_n*f_n.conj()).real*T_n**2*\
                  (Sx_perp_m**2+Sy_perp_m**2+Sz_perp_m**2)).sum(axis=1)\
              + 2*(c_k*c_l*(f_k*f_l.conj()).real*T_k*T_l*\
                  (Sx_perp_i*Sx_perp_j+Sy_perp_i*Sy_perp_j+Sz_perp_i*Sz_perp_j)*\
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

        atm = np.array(['Fe3+','Mn3+'])
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

        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)

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

        form_factor = magnetic.form(Q, atm, g=2)

        Sx_k, Sy_k, Sz_k, i_dft = magnetic.transform(Sx, Sy, Sz, H, K, L,
                                                     nu, nv, nw, n_atm)

        factors = space.prefactors(form_factor, phase_factor, occupancy)

        factors *= T

        Fx, Fy, Fz, \
        prod_x, prod_y, prod_z = magnetic.structure(Qx_norm, Qy_norm, Qz_norm,
                                                    Sx_k, Sy_k, Sz_k, i_dft,
                                                    factors)

        n_hkl = Q.size
        n_xyz = nu*nv*nw*n_atm

        m = np.arange(n_xyz)
        n = np.mod(m, n_atm)

        rx_m = rx[m]
        ry_m = ry[m]
        rz_m = rz[m]

        mf = form_factor.reshape(n_hkl,n_atm)
        T = T.reshape(n_hkl,n_atm)

        Sx_m = Sx[m]
        Sy_m = Sy[m]
        Sz_m = Sz[m]
        c_n = occupancy[n]
        f_n = mf[:,n]
        T_n = T[:,n]

        prod_x_ref = (c_n*f_n*Sx_m*T_n*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                  Qy[:,np.newaxis]*ry_m+\
                                                  Qz[:,np.newaxis]*rz_m)))

        prod_y_ref = (c_n*f_n*Sy_m*T_n*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                  Qy[:,np.newaxis]*ry_m+\
                                                  Qz[:,np.newaxis]*rz_m)))

        prod_z_ref = (c_n*f_n*Sz_m*T_n*np.exp(1j*(Qx[:,np.newaxis]*rx_m+\
                                                  Qy[:,np.newaxis]*ry_m+\
                                                  Qz[:,np.newaxis]*rz_m)))

        Fx_ref = prod_x_ref.sum(axis=1)
        Fy_ref = prod_y_ref.sum(axis=1)
        Fz_ref = prod_z_ref.sum(axis=1)

        prod_x_ref = prod_x_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1)
        prod_y_ref = prod_y_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1)
        prod_z_ref = prod_z_ref.reshape(n_hkl,nu*nv*nw,n_atm).sum(axis=1)

        prod_x_ref = prod_x_ref.flatten()
        prod_y_ref = prod_y_ref.flatten()
        prod_z_ref = prod_z_ref.flatten()

        np.testing.assert_array_almost_equal(Fx, Fx_ref)
        np.testing.assert_array_almost_equal(Fy, Fy_ref)
        np.testing.assert_array_almost_equal(Fz, Fz_ref)
        np.testing.assert_array_almost_equal(prod_x, prod_x_ref)
        np.testing.assert_array_almost_equal(prod_y, prod_y_ref)
        np.testing.assert_array_almost_equal(prod_z, prod_z_ref)

    def test_magnitude(self):

        mu1 = np.array([0.4,0.3])
        mu2 = np.array([0.5,0.4])
        mu3 = np.array([0.6,0.6])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        C = crystal.cartesian_moment(a, b, c, alpha, beta, gamma)

        mu = magnetic.magnitude(mu1, mu2, mu3, C)

        np.testing.assert_array_almost_equal(mu, np.sqrt(mu1**2+mu2**2+mu3**2))

    def test_cartesian(self):

        mu1 = np.array([0.4,0.3])
        mu2 = np.array([0.5,0.4])
        mu3 = np.array([0.6,0.6])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        C = crystal.cartesian_moment(a, b, c, alpha, beta, gamma)

        mux, muy, muz = magnetic.cartesian(mu1, mu2, mu3, C)

        np.testing.assert_array_almost_equal(mux, mu1)
        np.testing.assert_array_almost_equal(muy, mu2)
        np.testing.assert_array_almost_equal(muz, mu3)

if __name__ == '__main__':
    unittest.main()