#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#cython: language_level=3

import unittest
import numpy as np

from disorder.diffuse import space
from disorder.material import crystal

from disorder.diffuse cimport simulation

class test_c_simulation(unittest.TestCase):

    def test_iszero(self):

        self.assertTrue(simulation.iszero(1e-15))
        self.assertFalse(simulation.iszero(1e-7))

    def test_sqrt_babylonian(self):

        self.assertEqual(simulation.sqrt_babylonian(121), 11)
        self.assertEqual(simulation.sqrt_babylonian(484), 22)

    def test_alpha(self):

        E = 0.01
        T = 5

        kB = 0.08617
        beta = 1/(kB*T)

        self.assertAlmostEqual(simulation.alpha(E,beta), np.exp(-E*beta))

    def test_random_uniform(self):

        N = 100000

        u = np.zeros(N)

        for i in range(N):
            u[i] = simulation.random_uniform()

        self.assertAlmostEqual(u.mean(), 0.5, 2)
        self.assertGreaterEqual(u.min(), 0)
        self.assertLessEqual(u.mean(), 1)

    def test_random_vector_candidate(self):

        N = 100000

        ux, uy, uz = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            ux[i], uy[i], uz[i] = simulation.random_vector_candidate()

        u = np.sqrt(ux**2+uy**2+uz**2)
        t = np.arccos(uz/u)
        p = np.arctan2(ux,uy)

        np.testing.assert_array_almost_equal(u, 1)

        self.assertAlmostEqual(np.cos(t).mean(), 0.0, 1)
        self.assertAlmostEqual(p.mean(), 0.0, 1)

    def test_random_gaussian(self):

        N = 100000

        u = np.zeros(N)

        for i in range(N):
            u[i] = simulation.random_gaussian()

        self.assertAlmostEqual(u.mean(), 0.0, 1)
        self.assertAlmostEqual(u.std(), 1.0, 1)

    def test_random_vector_length_candidate(self):

        N = 100000

        ux, uy, uz = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            ux[i], uy[i], uz[i] = simulation.random_vector_length_candidate()

        u = np.sqrt(ux**2+uy**2+uz**2)
        t = np.arccos(uz/u)
        p = np.arctan2(ux,uy)

        np.testing.assert_array_equal(u <= 1, True)

        self.assertAlmostEqual(np.cos(t).mean(), 0.0, 1)
        self.assertAlmostEqual(p.mean(), 0.0, 1)

    def test_random_gaussian_3d(self):

        N = 100000

        u, v, w = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            u[i], v[i], w[i] = simulation.random_gaussian_3d()

        self.assertAlmostEqual(u.mean(), 0.0, 1)
        self.assertAlmostEqual(v.mean(), 0.0, 1)
        self.assertAlmostEqual(w.mean(), 0.0, 1)
        self.assertAlmostEqual(u.std(), 1.0, 1)
        self.assertAlmostEqual(v.std(), 1.0, 1)
        self.assertAlmostEqual(w.std(), 1.0, 1)

    def test_gaussian_vector_candidate(self):

        ux, uy, uz = simulation.random_vector_candidate()

        vx, vy, vz = simulation.gaussian_vector_candidate(ux, uy, uz, 0)

        self.assertAlmostEqual(ux, vx)
        self.assertAlmostEqual(uy, vy)
        self.assertAlmostEqual(uz, vz)

    def test_interpolated_vector_candidate(self):

        ux, uy, uz = simulation.random_vector_candidate()

        vx, vy, vz = simulation.interpolated_vector_candidate(ux, uy, uz, 0)

        self.assertAlmostEqual(ux, vx)
        self.assertAlmostEqual(uy, vy)
        self.assertAlmostEqual(uz, vz)

    def test_energy_single_dipole(self):

        n, m = 10, 3

        Q = np.zeros((n,n,3,3))

        X0 = np.random.random((n,n))
        X1 = np.random.random((n,n))
        X2 = np.random.random((n,n))
        X3 = np.random.random((n,n))
        X4 = np.random.random((n,n))
        X5 = np.random.random((n,n))

        ux = np.random.random((n,m))
        uy = np.random.random((n,m))
        uz = np.random.random((n,m))

        Q[:,:,0,0] = 0.5*(X0+X0.T)
        Q[:,:,1,1] = 0.5*(X1+X1.T)
        Q[:,:,2,2] = 0.5*(X2+X2.T)
        Q[:,:,1,2] = Q[:,:,2,1] = 0.5*(X3+X3.T)
        Q[:,:,0,2] = Q[:,:,2,0] = 0.5*(X4+X4.T)
        Q[:,:,0,1] = Q[:,:,1,0] = 0.5*(X5+X5.T)

        u = np.column_stack((ux,uy,uz)).reshape(-1,3,m)
        p = np.einsum('ijklm,jlm->iklm',np.einsum('ijkl,jkm->jiklm',Q,u),u)

        i, t = np.random.randint(n), np.random.randint(m)

        E0 = np.einsum('ijk,ijk->...',np.einsum('ijkl,jkm->ilm',Q,u),u)

        ox, oy, oz = ux[i,t].copy(), uy[i,t].copy(), uz[i,t].copy()
        cx, cy, cz = np.random.random(), np.random.random(), np.random.random()

        ux[i,t], uy[i,t], uz[i,t] = cx, cy, cz

        u = np.column_stack((ux,uy,uz)).reshape(-1,3,m)
        E1 = np.einsum('ijk,ijk->...',np.einsum('ijkl,jkm->ilm',Q,u),u)

        k, l = np.array([0,1,2,1,0,0]), np.array([0,1,2,2,2,1])

        Q = np.ascontiguousarray(Q[np.triu_indices(n)][:,k,l])

        E = simulation.energy_single_dipole(p, Q, cx, cy, cz, ox, oy, oz, i, t)

        self.assertAlmostEqual(E, E1-E0)

    def test_update_single_dipole(self):

        np.random.seed(13)

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n, m = nu*nv*nw*n_atm, 1

        Q = np.zeros((n,n,3,3))

        X0 = np.random.random((n,n))
        X1 = np.random.random((n,n))
        X2 = np.random.random((n,n))
        X3 = np.random.random((n,n))
        X4 = np.random.random((n,n))
        X5 = np.random.random((n,n))

        ux = np.random.random((n,m))
        uy = np.random.random((n,m))
        uz = np.random.random((n,m))

        Q[:,:,0,0] = 0.5*(X0+X0.T)
        Q[:,:,1,1] = 0.5*(X1+X1.T)
        Q[:,:,2,2] = 0.5*(X2+X2.T)
        Q[:,:,1,2] = Q[:,:,2,1] = 0.5*(X3+X3.T)
        Q[:,:,0,2] = Q[:,:,2,0] = 0.5*(X4+X4.T)
        Q[:,:,0,1] = Q[:,:,1,0] = 0.5*(X5+X5.T)

        u = np.column_stack((ux,uy,uz)).reshape(-1,3,m)
        p = np.einsum('ijklm,jlm->iklm',np.einsum('ijkl,jkm->jiklm',Q,u),u)

        i, t = np.random.randint(n), np.random.randint(m)

        ox, oy, oz = ux[i,t].copy(), uy[i,t].copy(), uz[i,t].copy()
        cx, cy, cz = np.random.random(), np.random.random(), np.random.random()

        Sx = ux.reshape(nu,nv,nw,n_atm,m).copy()
        Sy = uy.reshape(nu,nv,nw,n_atm,m).copy()
        Sz = uz.reshape(nu,nv,nw,n_atm,m).copy()

        ux[i,t], uy[i,t], uz[i,t] = cx, cy, cz

        u = np.column_stack((ux,uy,uz)).reshape(-1,3,m)
        q = np.einsum('ijklm,jlm->iklm',np.einsum('ijkl,jkm->jiklm',Q,u),u)

        k, l = np.array([0,1,2,1,0,0]), np.array([0,1,2,2,2,1])

        Q = np.ascontiguousarray(Q[np.triu_indices(n)][:,k,l])

        simulation.update_single_dipole(Sx, Sy, Sz, p, Q,
                                        cx, cy, cz, 
                                        ox, oy, oz, i, t)

        np.testing.assert_array_almost_equal(p[...,t], q[...,t])