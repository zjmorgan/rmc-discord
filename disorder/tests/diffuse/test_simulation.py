#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.diffuse import space, simulation, interaction
from disorder.material import crystal

import pyximport

args = ['--force', '--define', 'CYTHON_TRACE_NOGIL']
pyximport.install(setup_args={ 'script_args': args}, language_level=3)

from disorder.tests.diffuse.test_c_simulation import test_c_simulation

class test_simulation(unittest.TestCase):

    def test_c(self):

        self.assertEqual(test_c_simulation.__bases__[0], unittest.TestCase)

    def test_dipole_dipole_interaction_energy(self):

        nu, nv, nw, n_atm = 3, 4, 5, 2

        n = nu*nv*nw*n_atm

        M = 2

        Sx = np.random.random((nu,nv,nw,n_atm,M))
        Sy = np.random.random((nu,nv,nw,n_atm,M))
        Sz = np.random.random((nu,nv,nw,n_atm,M))

        u = np.array([0.2,0.3])
        v = np.array([0.5,0.4])
        w = np.array([0.7,0.2])

        atm = np.array(['',''])

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/2, 2*np.pi/3

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(*crystal.reciprocal(a, b, c, alpha, beta, gamma))
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        i, j = np.triu_indices(n)

        Qijm = np.zeros((n,n,6))
        Qijkl = np.zeros((n,n,3,3))

        Q = interaction.dipole_dipole_matrix(rx, ry, rz,
                                             nu, nv, nw,
                                             n_atm, A, B, R)

        Qijm[i,j,:] = Q

        Qijm[j,i,:] = Qijm[i,j,:]

        Qijkl[:,:,0,0] = Qijm[:,:,0]
        Qijkl[:,:,1,1] = Qijm[:,:,1]
        Qijkl[:,:,2,2] = Qijm[:,:,2]
        Qijkl[:,:,1,2] = Qijkl[:,:,2,1] = Qijm[:,:,3]
        Qijkl[:,:,0,2] = Qijkl[:,:,2,0] = Qijm[:,:,4]
        Qijkl[:,:,0,1] = Qijkl[:,:,1,0] = Qijm[:,:,5]

        e0 = simulation.dipole_dipole_interaction_energy(Sx, Sy, Sz, Q)

        Sx, Sy, Sz = Sx.reshape(n,M), Sy.reshape(n,M), Sz.reshape(n,M)

        S = np.column_stack((Sx,Sy,Sz)).reshape(-1,3,M)

        e = np.einsum('ijklm,jlm->iklm',np.einsum('ijkl,jkm->jiklm',Qijkl,S),S)

        np.testing.assert_array_almost_equal(e, e0)

        E = np.einsum('ijk,ijk->...',np.einsum('ijkl,jkm->ilm',Qijkl,S),S)

        np.testing.assert_array_almost_equal(E, e0.sum())

    def test_dipole_dipole_interaction_potential(self):

        nu, nv, nw, n_atm = 3, 4, 5, 2

        n = nu*nv*nw*n_atm

        M = 2

        Sx = np.random.random((nu,nv,nw,n_atm,M))
        Sy = np.random.random((nu,nv,nw,n_atm,M))
        Sz = np.random.random((nu,nv,nw,n_atm,M))

        u = np.array([0.2,0.3])
        v = np.array([0.5,0.4])
        w = np.array([0.7,0.2])

        atm = np.array(['',''])

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/2, 2*np.pi/3

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(*crystal.reciprocal(a, b, c, alpha, beta, gamma))
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        i, j = np.triu_indices(n)

        Qijm = np.zeros((n,n,6))
        Qijkl = np.zeros((n,n,3,3))

        Q = interaction.dipole_dipole_matrix(rx, ry, rz,
                                             nu, nv, nw,
                                             n_atm, A, B, R)

        Qijm[i,j,:] = Q

        Qijm[j,i,:] = Qijm[i,j,:]

        Qijkl[:,:,0,0] = Qijm[:,:,0]
        Qijkl[:,:,1,1] = Qijm[:,:,1]
        Qijkl[:,:,2,2] = Qijm[:,:,2]
        Qijkl[:,:,1,2] = Qijkl[:,:,2,1] = Qijm[:,:,3]
        Qijkl[:,:,0,2] = Qijkl[:,:,2,0] = Qijm[:,:,4]
        Qijkl[:,:,0,1] = Qijkl[:,:,1,0] = Qijm[:,:,5]

        p0 = simulation.dipole_dipole_interaction_potential(Sx, Sy, Sz, Q)

        Sx, Sy, Sz = Sx.reshape(n,M), Sy.reshape(n,M), Sz.reshape(n,M)

        S = np.column_stack((Sx,Sy,Sz)).reshape(-1,3,M)

        p = np.einsum('ijkl,jlm->ijklm',Qijkl,S).sum(axis=1)

        np.testing.assert_array_almost_equal(p, p0)

    def test_magnetic_energy(self):

        nu, nv, nw = 3, 4, 5

        n_atm = 2

        atm = np.array(['Fe3+','Fe3+'])

        u, v, w = np.array([0,0.5]), np.array([0,0.5]), np.array([0,0.5])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        ux, uy, uz = crystal.transform(u, v, w, A)

        pair_info = interaction.pairs(u, v, w, atm, A, extend=True)

        bond_info = interaction.bonds(pair_info, u, v, w, A, tol=1e-2)

        dx, dy, dz, img_i, img_j, img_k, *indices = bond_info

        atm_ind, pair_inv, pair_ind, pair_trans = indices

        d = np.sqrt(dx**2+dy**2+dz**2)

        mask = d < 0.99*np.sqrt(2*a**2)

        n_pair = mask.sum() // n_atm

        dx = dx[mask].reshape(n_atm,n_pair)
        dy = dy[mask].reshape(n_atm,n_pair)
        dz = dz[mask].reshape(n_atm,n_pair)

        pair_ind = pair_ind[mask].reshape(n_atm,n_pair)

        img_i = img_i[mask].reshape(n_atm,n_pair)
        img_j = img_j[mask].reshape(n_atm,n_pair)
        img_k = img_k[mask].reshape(n_atm,n_pair)

        atm_ind = atm_ind[mask].reshape(n_atm,n_pair)
        pair_trans = pair_trans[mask].reshape(n_atm,n_pair)

        J = np.zeros((n_pair,3,3))
        K = np.zeros((n_atm,3,3))
        g = np.zeros((n_atm,3,3))
        B = np.zeros(3)

        Jx, Jy, Jz = -3, -2, -1
        Kx, Ky, Kz = -1, -3, -2
        gx, gy, gz = 2, 3, 4
        Bx, By, Bz = 1, 2, 3

        J[:n_pair,:,:] = np.array([[Jx, 0, 0],
                                   [0, Jy, 0],
                                   [0, 0, Jz]], dtype=float)

        K[:n_atm,:,:] = np.array([[Kx, 0, 0],
                                  [0, Ky, 0],
                                  [0, 0, Kz]], dtype=float)

        g[:n_atm,:,:] = np.array([[gx, 0, 0],
                                  [0, gy, 0],
                                  [0, 0, gz]], dtype=float)

        B[:] = Bx, By, Bz

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, ion = space.real(ux, uy, uz, ix, iy, iz, atm)

        n = nu*nv*nw*n_atm

        M = 3

        Sx = np.zeros((nu,nv,nw,n_atm,M))
        Sy = np.zeros((nu,nv,nw,n_atm,M))
        Sz = np.zeros((nu,nv,nw,n_atm,M))

        Sx[...,0], Sy[...,0], Sz[...,0] = 1, 0, 0
        Sx[...,1], Sy[...,1], Sz[...,1] = 0, 1, 0
        Sx[...,2], Sy[...,2], Sz[...,2] = 0, 0, 1

        E = simulation.magnetic_energy(Sx, Sy, Sz, J, K, g, B, atm_ind,
                                       img_i, img_j, img_k,
                                       pair_ind, pair_trans)

        self.assertAlmostEqual(E[...,0].sum(), -(0.5*Jx*n_pair+Kx+Bx*gx)*n)
        self.assertAlmostEqual(E[...,1].sum(), -(0.5*Jy*n_pair+Ky+By*gy)*n)
        self.assertAlmostEqual(E[...,2].sum(), -(0.5*Jz*n_pair+Kz+Bz*gz)*n)

    def test_heisenberg(self):

        np.random.seed(13)

        nu, nv, nw = 3, 4, 5

        n_atm = 2

        atm = np.array(['Fe3+','Fe3+'])

        u, v, w = np.array([0,0.5]), np.array([0,0.5]), np.array([0,0.5])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        ux, uy, uz = crystal.transform(u, v, w, A)

        pair_info = interaction.pairs(u, v, w, atm, A, extend=True)

        bond_info = interaction.bonds(pair_info, u, v, w, A, tol=1e-2)

        dx, dy, dz, img_i, img_j, img_k, *indices = bond_info

        atm_ind, pair_inv, pair_ind, pair_trans = indices

        d = np.sqrt(dx**2+dy**2+dz**2)

        mask = d < 0.99*np.sqrt(2*a**2)

        n_pair = mask.sum() // n_atm

        dx = dx[mask].reshape(n_atm,n_pair)
        dy = dy[mask].reshape(n_atm,n_pair)
        dz = dz[mask].reshape(n_atm,n_pair)

        pair_ind = pair_ind[mask].reshape(n_atm,n_pair)

        img_i = img_i[mask].reshape(n_atm,n_pair)
        img_j = img_j[mask].reshape(n_atm,n_pair)
        img_k = img_k[mask].reshape(n_atm,n_pair)

        atm_ind = atm_ind[mask].reshape(n_atm,n_pair)
        pair_trans = pair_trans[mask].reshape(n_atm,n_pair)

        J = np.zeros((n_pair,3,3))
        K = np.zeros((n_atm,3,3))
        g = np.zeros((n_atm,3,3))
        B = np.zeros(3)

        Jx, Jy, Jz = -3, -2, -1
        Kx, Ky, Kz = -1, -3, -2
        gx, gy, gz = 2, 3, 4
        Bx, By, Bz = 1, 2, 3

        J[:n_pair,:,:] = np.array([[Jx, 0, 0],
                                   [0, Jy, 0],
                                   [0, 0, Jz]], dtype=float)

        K[:n_atm,:,:] = np.array([[Kx, 0, 0],
                                  [0, Ky, 0],
                                  [0, 0, Kz]], dtype=float)

        g[:n_atm,:,:] = np.array([[gx, 0, 0],
                                  [0, gy, 0],
                                  [0, 0, gz]], dtype=float)

        B[:] = Bx, By, Bz

        n = nu*nv*nw*n_atm

        Q = np.random.random((n*(n+1)//2,6))

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, ion = space.real(ux, uy, uz, ix, iy, iz, atm)

        M, N = 6, 5

        T0, T1 = 0.01, 5

        T_range = np.logspace(np.log2(T0), np.log2(T1), M, base=2)

        kB = 0.08617

        theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm,M)
        phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm,M))

        Sx = np.sin(phi)*np.cos(theta)
        Sy = np.sin(phi)*np.sin(theta)
        Sz = np.cos(phi)

        E, T_range = simulation.heisenberg(Sx, Sy, Sz, J, K, g, B, Q, atm_ind,
                                           img_i, img_j, img_k, pair_ind,
                                           pair_trans, T_range, kB, N)

        E_ref = simulation.magnetic_energy(Sx, Sy, Sz, J, K, g, B, atm_ind,
                                           img_i, img_j, img_k,
                                           pair_ind, pair_trans)

        V_ref = simulation.dipole_dipole_interaction_energy(Sx, Sy, Sz, Q)

        E0 = E_ref.sum(axis=(0,1,2,3,4))+V_ref.sum(axis=(0,1,2))

        np.testing.assert_array_almost_equal(E, E0)

    def test_heisenberg_cluster(self):

        np.random.seed(13)

        nu, nv, nw = 3, 4, 5

        n_atm = 2

        atm = np.array(['Fe3+','Fe3+'])

        u, v, w = np.array([0,0.5]), np.array([0,0.5]), np.array([0,0.5])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        ux, uy, uz = crystal.transform(u, v, w, A)

        pair_info = interaction.pairs(u, v, w, atm, A, extend=True)

        bond_info = interaction.bonds(pair_info, u, v, w, A, tol=1e-2)

        dx, dy, dz, img_i, img_j, img_k, *indices = bond_info

        atm_ind, pair_inv, pair_ind, pair_trans = indices

        d = np.sqrt(dx**2+dy**2+dz**2)

        mask = d < 0.99*np.sqrt(2*a**2)

        n_pair = mask.sum() // n_atm

        dx = dx[mask].reshape(n_atm,n_pair)
        dy = dy[mask].reshape(n_atm,n_pair)
        dz = dz[mask].reshape(n_atm,n_pair)

        pair_ind = pair_ind[mask].reshape(n_atm,n_pair)

        img_i = img_i[mask].reshape(n_atm,n_pair)
        img_j = img_j[mask].reshape(n_atm,n_pair)
        img_k = img_k[mask].reshape(n_atm,n_pair)

        atm_ind = atm_ind[mask].reshape(n_atm,n_pair)
        pair_trans = pair_trans[mask].reshape(n_atm,n_pair)

        J = np.zeros((n_pair,3,3))
        K = np.zeros((n_atm,3,3))
        g = np.zeros((n_atm,3,3))
        B = np.zeros(3)

        Jx, Jy, Jz = -3, -2, -1
        Kx, Ky, Kz = -1, -3, -2
        gx, gy, gz = 2, 3, 4
        Bx, By, Bz = 1, 2, 3

        J[:n_pair,:,:] = np.array([[Jx, 0, 0],
                                   [0, Jy, 0],
                                   [0, 0, Jz]], dtype=float)

        K[:n_atm,:,:] = np.array([[Kx, 0, 0],
                                  [0, Ky, 0],
                                  [0, 0, Kz]], dtype=float)

        g[:n_atm,:,:] = np.array([[gx, 0, 0],
                                  [0, gy, 0],
                                  [0, 0, gz]], dtype=float)

        B[:] = Bx, By, Bz

        n = nu*nv*nw*n_atm

        Q = np.random.random((n*(n+1)//2,6))*0

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, ion = space.real(ux, uy, uz, ix, iy, iz, atm)

        M, N = 2, 10

        T0, T1 = 0.01, 5

        T_range = np.logspace(np.log2(T0), np.log2(T1), M, base=2)

        kB = 0.08617

        theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm,M)
        phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm,M))

        Sx = np.sin(phi)*np.cos(theta)
        Sy = np.sin(phi)*np.sin(theta)
        Sz = np.cos(phi)

        E, T_range = simulation.heisenberg_cluster(Sx, Sy, Sz, J, K, g, B, Q,
                                                   atm_ind, img_i,
                                                   img_j, img_k, pair_ind,
                                                   pair_inv, pair_trans,
                                                   T_range, kB, N)

        E_ref = simulation.magnetic_energy(Sx, Sy, Sz, J, K, g, B, atm_ind,
                                           img_i, img_j, img_k,
                                           pair_ind, pair_trans)

        V_ref = simulation.dipole_dipole_interaction_energy(Sx, Sy, Sz, Q)

        E0 = E_ref.sum(axis=(0,1,2,3,4))+V_ref.sum(axis=(0,1,2))

        np.testing.assert_array_almost_equal(E, E0)

if __name__ == '__main__':
    unittest.main()
