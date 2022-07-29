 #!/usr/bin/env python3

import io
import os
import sys

import unittest
import numpy as np

import itertools

from disorder.diffuse import refinement

import pyximport

pyximport.install(setup_args={ 'script_args': ['--force']}, language_level=3)

from disorder.tests.diffuse.test_c_refinement import test_c_refinement

class test_refinement(unittest.TestCase):

    def test_c(self):

        self.assertEqual(test_c_refinement.__bases__[0], unittest.TestCase)

    def test_parallelism(self):

        out = io.StringIO()
        sys.stdout = out

        refinement.parallelism(app=False)

        sys.stdout = sys.__stdout__

        num_threads = os.environ.get('OMP_NUM_THREADS')

        self.assertEqual(out.getvalue(), 'threads: {}\n'.format(num_threads))

    def test_threads(self):

        out = io.StringIO()
        sys.stdout = out

        refinement.threads()

        sys.stdout = sys.__stdout__

        num_threads = os.environ.get('OMP_NUM_THREADS')

        self.assertEqual(out.getvalue(),
                         ''.join(['id: {}\n'.format(i_thread) \
                                  for i_thread in range(int(num_threads))]))

    def test_original_scalar(self):

        A = np.random.random(16)

        A_orig, i = refinement.original_scalar(A)

        self.assertAlmostEqual(A_orig, A[i])

    def test_original_vector(self):

        A = np.random.random(16)
        B = np.random.random(16)
        C = np.random.random(16)

        A_orig, B_orig, C_orig, i = refinement.original_vector(A, B, C)

        self.assertAlmostEqual(A_orig, A[i])
        self.assertAlmostEqual(B_orig, B[i])
        self.assertAlmostEqual(C_orig, C[i])

    def test_candidate_composition(self):

        A_orig = -1

        c = 0.5

        A_cand = refinement.candidate_composition(A_orig, c)
        A_orig = refinement.candidate_composition(A_cand, c)

        self.assertAlmostEqual(A_orig, -1)

        self.assertAlmostEqual((A_orig+1)*c, 0)
        self.assertAlmostEqual((A_cand+1)*c, 1)

        c = 0.85

        A_cand = refinement.candidate_composition(A_orig, c)
        A_orig = refinement.candidate_composition(A_cand, c)

        self.assertAlmostEqual(A_orig, -1)

        self.assertAlmostEqual((A_orig+1)*c, 0)
        self.assertAlmostEqual((A_cand+1)*c, 1)

    def test_candidate_moment(self):

        theta = 2*np.pi*np.random.random()
        phi = np.arccos(1-2*np.random.random())
        S = np.random.random()

        Sx_orig = S*np.sin(phi)*np.cos(theta)
        Sy_orig = S*np.sin(phi)*np.sin(theta)
        Sz_orig = S*np.cos(phi)

        Sx_cand, Sy_cand, Sz_cand = refinement.candidate_moment(Sx_orig,
                                                                Sy_orig,
                                                                Sz_orig, S)

        self.assertAlmostEqual(Sx_orig**2+Sy_orig**2+Sz_orig**2, S**2)
        self.assertAlmostEqual(Sx_cand**2+Sy_cand**2+Sz_cand**2, S**2)

    def test_candidate_displacement(self):

        Uxx, Uyy, Uzz = 1.2, 2.3, 1.75
        Uyz, Uxz, Uxy = 0.1, -0.25, 0.15

        U = np.array([[Uxx,Uxy,Uxz],
                      [Uxy,Uyy,Uyz],
                      [Uxz,Uyz,Uzz]])

        U_eq = np.trace(U)/3

        L = np.linalg.cholesky(U)

        Lxx, Lyy, Lzz = L[0,0], L[1,1], L[2,2]
        Lyz, Lxz, Lxy = L[2,1], L[2,0], L[1,0]

        N = 100000

        Ux, Uy, Uz = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):

            Ux_cand, Uy_cand, Uz_cand = refinement.candidate_displacement(Lxx,
                                                                          Lyy,
                                                                          Lzz,
                                                                          Lyz,
                                                                          Lxz,
                                                                          Lxy)


            Ux[i], Uy[i], Uz[i] = Ux_cand, Uy_cand, Uz_cand

        self.assertAlmostEqual(np.mean(Ux**2+Uy**2+Uz**2)/3, U_eq, 1)

    def test_extract_complex(self):

        n_hkl, n_atm = 101, 3

        n = n_hkl*n_atm

        data = np.random.random(n)+1j*np.random.random(n)
        values = np.zeros(n_hkl, dtype=complex)

        j = 1
        refinement.extract_complex(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

        j = 2
        refinement.extract_complex(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

    def test_insert_complex(self):

        n_hkl, n_atm = 101, 3

        n = n_hkl*n_atm

        data = np.random.random(n)+1j*np.random.random(n)
        values = np.random.random(n_hkl)+1j*np.random.random(n_hkl)

        j = 1
        refinement.insert_complex(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

        j = 2
        refinement.insert_complex(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

    def test_extract_real(self):

        n_hkl, n_atm = 101, 3

        n = n_hkl*n_atm

        data = np.random.random(n)
        values = np.zeros(n_hkl, dtype=float)

        j = 1
        refinement.extract_real(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

        j = 2
        refinement.extract_real(values, data, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

    def test_insert_real(self):

        n_hkl, n_atm = 101, 3

        n = n_hkl*n_atm

        data = np.random.random(n)
        values = np.random.random(n_hkl)

        j = 1
        refinement.insert_real(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

        j = 2
        refinement.insert_real(data, values, j, n_atm)
        np.testing.assert_array_almost_equal(values, data[j::n_atm])

    def test_extract_many_complex(self):

        n_hkl, n_atm = 101, 16

        n = n_hkl*n_atm

        ind = np.array([0,2,3])
        n_ind = ind.shape[0]

        data = np.random.random(n)+1j*np.random.random(n)
        values = np.zeros(n_hkl*n_ind, dtype=complex)

        refinement.extract_many_complex(values, data, ind, n_atm)

        data = data.reshape(n_hkl,n_atm)
        np.testing.assert_array_almost_equal(values, data[:,ind].flatten())

    def test_insert_many_complex(self):

        n_hkl, n_atm = 101, 16

        n = n_hkl*n_atm

        ind = np.array([0,2,3])
        n_ind = ind.shape[0]

        data = np.random.random(n)+1j*np.random.random(n)
        values = np.random.random(n_hkl*n_ind)+1j*np.random.random(n_hkl*n_ind)

        refinement.insert_many_complex(data, values, ind, n_atm)

        data = data.reshape(n_hkl,n_atm)
        np.testing.assert_array_almost_equal(values, data[:,ind].flatten())

    def test_extract_many_real(self):

        n_hkl, n_atm = 101, 16

        n = n_hkl*n_atm

        ind = np.array([0,2,3])
        n_ind = ind.shape[0]

        data = np.random.random(n)
        values = np.zeros(n_hkl*n_ind, dtype=float)

        refinement.extract_many_real(values, data, ind, n_atm)

        data = data.reshape(n_hkl,n_atm)
        np.testing.assert_array_almost_equal(values, data[:,ind].flatten())

    def test_insert_many_real(self):

        n_hkl, n_atm = 101, 16

        n = n_hkl*n_atm

        ind = np.array([0,2,3])
        n_ind = ind.shape[0]

        data = np.random.random(n)
        values = np.random.random(n_hkl*n_ind)

        refinement.insert_many_real(data, values, ind, n_atm)

        data = data.reshape(n_hkl,n_atm)
        np.testing.assert_array_almost_equal(values, data[:,ind].flatten())

    def test_copy_complex(self):

        n_hkl, n_atm = 101, 16

        n = n_hkl*n_atm

        data = np.random.random(n)+1j*np.random.random(n)
        values = np.zeros(n, dtype=complex)

        refinement.copy_complex(values, data)
        np.testing.assert_array_almost_equal(values, data)

    def test_scattering_intensity(self):

        n = 101

        I = np.random.random(n)

        mask = I < 0.2

        i_mask = np.arange(n)[mask]

        inverses = np.arange(n) % i_mask.size

        I_calc = np.random.random(i_mask.size)

        I0 = I_calc[inverses].copy()
        I0[mask] = 0

        refinement.scattering_intensity(I, I_calc, inverses, i_mask)

        np.testing.assert_array_almost_equal(I, I0)

    def test_unmask_intensity(self):

        n = 101

        I = np.random.random(n)

        mask = I < 0.2

        i_mask = np.arange(n)[mask]

        I_calc = np.random.random(i_mask.size)

        I0_calc = I[mask].copy()

        refinement.unmask_intensity(I_calc, I, i_mask)

        np.testing.assert_array_almost_equal(I_calc, I0_calc)

    def test_reduced_chi_square(self):

        n = 101

        x = np.linspace(-3,3,n)

        y_fit = 5*np.exp(-0.5*x**2)
        y_obs = 2*y_fit+0.01*(2*np.random.random(n)-1)

        e = np.sqrt(y_obs)
        inv_err_sq = 1/e**2

        chi_sq, scale = refinement.reduced_chi_square(y_fit, y_obs, inv_err_sq)

        self.assertAlmostEqual(scale, 2, 2)

        self.assertAlmostEqual(chi_sq, np.sum((2*y_fit-y_obs)**2/e**2), 3)

    def test_products(self):

        Vx, Vy, Vz = 1.2, 2.3, 1.75

        p = 3
        V = np.zeros((p+1)*(p+2)*(p+3)//6)

        refinement.products(V, Vx, Vy, Vz, p)

        exponents = np.array(list(itertools.product(np.arange(p+1), repeat=3)))
        exponents = exponents[np.sum(exponents, axis=1) <= p]

        V0 = np.product(np.power([Vx,Vy,Vz], exponents), axis=1)

        np.testing.assert_array_almost_equal(np.sort(V), np.sort(V0))

        p = 4
        V = np.zeros((p+1)*(p+2)*(p+3)//6)

        refinement.products(V, Vx, Vy, Vz, p)

        exponents = np.array(list(itertools.product(np.arange(p+1), repeat=3)))
        exponents = exponents[np.sum(exponents, axis=1) <= p]

        V0 = np.product(np.power([Vx,Vy,Vz], exponents), axis=1)

        np.testing.assert_array_almost_equal(np.sort(V), np.sort(V0))

    def test_products_mol(self):

        Vx = np.array([1.2, 1.6])
        Vy = np.array([2.3, 4.5])
        Vz = np.array([1.75, 2.25])

        p = 3
        V = np.zeros(((p+1)*(p+2)*(p+3)//6,2))

        refinement.products_mol(V, Vx, Vy, Vz, p)

        exponents = np.array(list(itertools.product(np.arange(p+1), repeat=3)))
        exponents = exponents[np.sum(exponents, axis=1) <= p]

        V0 = np.zeros(((p+1)*(p+2)*(p+3)//6,2))
        for i in range(2):
            V0[:,i] = np.product(np.power([Vx[i],Vy[i],Vz[i]],
                                          exponents), axis=1)

        np.testing.assert_array_almost_equal(np.sort(V.flatten()),
                                             np.sort(V0.flatten()))

    def test_magnetic_intensity(self):

        n_hkl = 101
        n_xyz = 1000

        I = np.zeros(n_hkl)

        Fx = np.random.random(n_hkl)+1j*np.random.random(n_hkl)
        Fy = np.random.random(n_hkl)+1j*np.random.random(n_hkl)
        Fz = np.random.random(n_hkl)+1j*np.random.random(n_hkl)

        theta = 2*np.pi*np.random.random(n_hkl)
        phi = np.arccos(1-2*np.random.random(n_hkl))

        Qx_norm = np.sin(phi)*np.cos(theta)
        Qy_norm = np.sin(phi)*np.sin(theta)
        Qz_norm = np.cos(phi)

        refinement.magnetic_intensity(I, Qx_norm, Qy_norm, Qz_norm,
                                      Fx, Fy, Fz, n_xyz)

        Q_hat = np.stack((Qx_norm,Qy_norm,Qz_norm))
        F = np.stack((Fx,Fy,Fz))

        F_cross_Q_hat = np.cross(F, Q_hat, axis=0)

        Q_hat_cross_F_cross_Q_hat = np.cross(Q_hat, F_cross_Q_hat, axis=0)

        I0 = np.linalg.norm(Q_hat_cross_F_cross_Q_hat, axis=0)**2/n_xyz

        np.testing.assert_array_almost_equal(I, I0)

    def test_occupational_intensity(self):

        n_hkl = 101
        n_xyz = 1000

        I = np.zeros(n_hkl)

        F = np.random.random(n_hkl)+1j*np.random.random(n_hkl)

        refinement.occupational_intensity(I, F, n_xyz)

        I0 = np.abs(F)**2/n_xyz

        np.testing.assert_array_almost_equal(I, I0)

    def test_displacive_intensity(self):

        n_hkl = 101
        n_xyz = 1000

        I = np.zeros(n_hkl)

        F = np.random.random(n_hkl)+1j*np.random.random(n_hkl)

        bragg = np.arange(n_hkl)[np.abs(F) < 0.5]

        n_nuc = bragg.size

        F_nuc = np.random.random(n_nuc)+1j*np.random.random(n_nuc)

        F0 = F.copy()
        F0[bragg] = F_nuc-F[bragg]

        refinement.displacive_intensity(I, F, F_nuc, bragg, n_xyz)

        I0 = np.abs(F0)**2/n_xyz

        np.testing.assert_array_almost_equal(I, I0)
        np.testing.assert_array_almost_equal(F, F0)

    def test_update_spin(self):

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        ku, kv, kw = np.fft.fftfreq(nu), np.fft.fftfreq(nv), np.fft.fftfreq(nw)
        ru, rv, rw = np.arange(nu), np.arange(nv), np.arange(nw)

        k_dot_r = np.kron(ku,ru)[:,np.newaxis,np.newaxis]+\
                  np.kron(kv,rv)[:,np.newaxis]+\
                  np.kron(kw,rw)

        space_factor = np.exp(2j*np.pi*k_dot_r).flatten()

        Sx = np.random.random((nu,nv,nw,n_atm)).flatten()
        Sy = np.random.random((nu,nv,nw,n_atm)).flatten()
        Sz = np.random.random((nu,nv,nw,n_atm)).flatten()

        Sx_k = np.fft.ifftn(Sx.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()
        Sy_k = np.fft.ifftn(Sy.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()
        Sz_k = np.fft.ifftn(Sz.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()

        i = np.random.randint(n)
        j = i % n_atm

        Sx_orig, Sy_orig, Sz_orig = Sx[i].copy(), Sy[i].copy(), Sz[i].copy()

        Sx_k_orig = Sx_k[j::n_atm]*n_uvw
        Sy_k_orig = Sy_k[j::n_atm]*n_uvw
        Sz_k_orig = Sz_k[j::n_atm]*n_uvw

        Sx_cand = np.random.random()
        Sy_cand = np.random.random()
        Sz_cand = np.random.random()

        Sx_k_cand = Sx_k[j::n_atm]*n_uvw
        Sy_k_cand = Sy_k[j::n_atm]*n_uvw
        Sz_k_cand = Sz_k[j::n_atm]*n_uvw

        refinement.update_spin(Sx_k_cand, Sy_k_cand, Sz_k_cand,
                               Sx_cand, Sy_cand, Sz_cand,
                               Sx_k_orig, Sy_k_orig, Sz_k_orig,
                               Sx_orig, Sy_orig, Sz_orig,
                               space_factor, i, nu, nv, nw, n_atm)

        Sx[i], Sy[i], Sz[i] = Sx_cand, Sy_cand, Sz_cand

        Sx_k = np.fft.ifftn(Sx.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()
        Sy_k = np.fft.ifftn(Sy.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()
        Sz_k = np.fft.ifftn(Sz.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()

        np.testing.assert_array_almost_equal(Sx_k_cand, Sx_k[j::n_atm]*n_uvw)
        np.testing.assert_array_almost_equal(Sy_k_cand, Sy_k[j::n_atm]*n_uvw)
        np.testing.assert_array_almost_equal(Sz_k_cand, Sz_k[j::n_atm]*n_uvw)

    def test_update_composition(self):

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        ku, kv, kw = np.fft.fftfreq(nu), np.fft.fftfreq(nv), np.fft.fftfreq(nw)
        ru, rv, rw = np.arange(nu), np.arange(nv), np.arange(nw)

        k_dot_r = np.kron(ku,ru)[:,np.newaxis,np.newaxis]+\
                  np.kron(kv,rv)[:,np.newaxis]+\
                  np.kron(kw,rw)

        space_factor = np.exp(2j*np.pi*k_dot_r).flatten()

        A_r = np.random.random((nu,nv,nw,n_atm)).flatten()

        A_k = np.fft.ifftn(A_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()

        i = np.random.randint(n)
        j = i % n_atm

        A_r_orig = A_r[i].copy()

        A_k_orig = A_k[j::n_atm]*n_uvw

        A_r_cand = np.random.random()

        A_k_cand = A_k[j::n_atm]*n_uvw

        refinement.update_composition(A_k_cand, A_r_cand, A_k_orig, A_r_orig,
                                      space_factor, i, nu, nv, nw, n_atm)

        A_r[i] = A_r_cand

        A_k = np.fft.ifftn(A_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2)).flatten()

        np.testing.assert_array_almost_equal(A_k_cand, A_k[j::n_atm]*n_uvw)

    def test_update_expansion(self):

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        n_prod = 6

        ku, kv, kw = np.fft.fftfreq(nu), np.fft.fftfreq(nv), np.fft.fftfreq(nw)
        ru, rv, rw = np.arange(nu), np.arange(nv), np.arange(nw)

        k_dot_r = np.kron(ku,ru)[:,np.newaxis,np.newaxis]+\
                  np.kron(kv,rv)[:,np.newaxis]+\
                  np.kron(kw,rw)

        space_factor = np.exp(2j*np.pi*k_dot_r).flatten()

        U_r = np.random.random((n_prod,nu,nv,nw,n_atm)).flatten()

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))
        U_k = U_k.flatten()

        i = np.random.randint(n)
        j = i % n_atm

        U_r_orig = U_r[i+n*np.arange(n_prod)].copy()

        U_k_orig = U_k[j::n_atm]*n_uvw

        U_r_cand = np.random.random(n_prod)

        U_k_cand = U_k[j::n_atm]*n_uvw

        refinement.update_expansion(U_k_cand, U_r_cand, U_k_orig, U_r_orig,
                                    space_factor, i, nu, nv, nw, n_atm)

        U_r[i+n*np.arange(n_prod)] = U_r_cand

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))
        U_k = U_k.flatten()

        np.testing.assert_array_almost_equal(U_k_cand, U_k[j::n_atm]*n_uvw)

    def test_magnetic_structure_factor(self):

        n_hkl = 101

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        factors =    np.random.random((n_hkl,n_atm))\
                + 1j*np.random.random((n_hkl,n_atm))

        i_dft = np.random.randint(n_uvw, size=n_hkl)

        Sx = np.random.random((nu,nv,nw,n_atm)).flatten()
        Sy = np.random.random((nu,nv,nw,n_atm)).flatten()
        Sz = np.random.random((nu,nv,nw,n_atm)).flatten()

        Sx_k = np.fft.ifftn(Sx.reshape(nu,nv,nw,n_atm), axes=(0,1,2))
        Sy_k = np.fft.ifftn(Sy.reshape(nu,nv,nw,n_atm), axes=(0,1,2))
        Sz_k = np.fft.ifftn(Sz.reshape(nu,nv,nw,n_atm), axes=(0,1,2))

        Sx_k = Sx_k.reshape(n_uvw,n_atm)
        Sy_k = Sy_k.reshape(n_uvw,n_atm)
        Sz_k = Sz_k.reshape(n_uvw,n_atm)

        i = np.random.randint(n)
        j = i % n_atm

        Sx_k_orig = Sx_k[:,j].copy()
        Sy_k_orig = Sy_k[:,j].copy()
        Sz_k_orig = Sz_k[:,j].copy()

        p_x_orig = factors[:,j]*Sx_k_orig[i_dft]
        p_y_orig = factors[:,j]*Sy_k_orig[i_dft]
        p_z_orig = factors[:,j]*Sz_k_orig[i_dft]

        Fx_orig = np.sum(factors*Sx_k[i_dft,:], axis=1)
        Fy_orig = np.sum(factors*Sy_k[i_dft,:], axis=1)
        Fz_orig = np.sum(factors*Sz_k[i_dft,:], axis=1)

        Sx_cand = np.random.random()
        Sy_cand = np.random.random()
        Sz_cand = np.random.random()

        Sx[i], Sy[i], Sz[i] = Sx_cand, Sy_cand, Sz_cand

        Sx_k = np.fft.ifftn(Sx.reshape(nu,nv,nw,n_atm), axes=(0,1,2))
        Sy_k = np.fft.ifftn(Sy.reshape(nu,nv,nw,n_atm), axes=(0,1,2))
        Sz_k = np.fft.ifftn(Sz.reshape(nu,nv,nw,n_atm), axes=(0,1,2))

        Sx_k = Sx_k.reshape(n_uvw,n_atm)
        Sy_k = Sy_k.reshape(n_uvw,n_atm)
        Sz_k = Sz_k.reshape(n_uvw,n_atm)

        Sx_k_cand = Sx_k[:,j].copy()
        Sy_k_cand = Sy_k[:,j].copy()
        Sz_k_cand = Sz_k[:,j].copy()

        p_x_cand = factors[:,j]*Sx_k_cand[i_dft]
        p_y_cand = factors[:,j]*Sy_k_cand[i_dft]
        p_z_cand = factors[:,j]*Sz_k_cand[i_dft]

        Fx_cand = np.sum(factors*Sx_k[i_dft,:], axis=1)
        Fy_cand = np.sum(factors*Sy_k[i_dft,:], axis=1)
        Fz_cand = np.sum(factors*Sz_k[i_dft,:], axis=1)

        Fx = Fx_cand.copy()
        Fy = Fy_cand.copy()
        Fz = Fz_cand.copy()

        Fx_cand[:] = 0
        Fy_cand[:] = 0
        Fz_cand[:] = 0

        factors = factors.flatten()

        refinement.magnetic_structure_factor(Fx_cand, Fy_cand, Fz_cand,
                                             p_x_cand, p_y_cand, p_z_cand,
                                             Sx_k_cand, Sy_k_cand, Sz_k_cand,
                                             Fx_orig, Fy_orig, Fz_orig,
                                             p_x_orig, p_y_orig, p_z_orig,
                                             factors, j, i_dft, n_atm)

        np.testing.assert_array_almost_equal(Fx_cand, Fx)
        np.testing.assert_array_almost_equal(Fy_cand, Fy)
        np.testing.assert_array_almost_equal(Fz_cand, Fz)

    def test_occupational_structure_factor(self):

        n_hkl = 101

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        factors =    np.random.random((n_hkl,n_atm))\
                + 1j*np.random.random((n_hkl,n_atm))

        i_dft = np.random.randint(n_uvw, size=n_hkl)

        A_r = np.random.random((nu,nv,nw,n_atm)).flatten()

        A_k = np.fft.ifftn(A_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2))

        A_k = A_k.reshape(n_uvw,n_atm)

        i = np.random.randint(n)
        j = i % n_atm

        A_k_orig = A_k[:,j].copy()

        p_orig = factors[:,j]*A_k_orig[i_dft]

        F_orig = np.sum(factors*A_k[i_dft,:], axis=1)

        A_r_cand = np.random.random()

        A_r[i] = A_r_cand

        A_k = np.fft.ifftn(A_r.reshape(nu,nv,nw,n_atm), axes=(0,1,2))

        A_k = A_k.reshape(n_uvw,n_atm)

        A_k_cand = A_k[:,j].copy()

        p_cand = factors[:,j]*A_k_cand[i_dft]

        F_cand = np.sum(factors*A_k[i_dft,:], axis=1)

        F = F_cand.copy()

        F_cand[:] = 0

        factors = factors.flatten()

        refinement.occupational_structure_factor(F_cand, p_cand, A_k_cand,
                                                 F_orig, p_orig,
                                                 factors, j, i_dft, n_atm)

        np.testing.assert_array_almost_equal(F_cand, F)

    def test_displacive_structure_factor(self):

        n_hkl = 101

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        p = 2
        n_prod = 10

        even = np.array([0, 4, 5, 6, 7, 8, 9])

        factors =    np.random.random((n_hkl,n_atm))\
                + 1j*np.random.random((n_hkl,n_atm))

        i_dft = np.random.randint(n_uvw, size=n_hkl)

        Q_k = np.random.random((n_prod,n_hkl))

        coeffs = np.random.random(n_prod)+1j*np.random.random(n_prod)

        cond = np.random.random(n_hkl) < 0.5
        bragg = np.arange(n_hkl)[cond]

        U_r = np.random.random((n_prod,nu,nv,nw,n_atm)).flatten()

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))

        U_k = U_k.reshape(n_prod,n_uvw,n_atm)

        i = np.random.randint(n)
        j = i % n_atm

        U_k_orig = U_k[...,j].copy()

        V_k_orig = np.einsum('jk,kj->j', coeffs*U_k_orig[:,i_dft].T, Q_k)
        V_k_nuc_orig = np.einsum('jk,kj->j',
                                 coeffs[even]*U_k_orig[:,i_dft][even,:].T,
                                 Q_k[even,:])[cond]

        p_orig = factors[:,j]*V_k_orig
        p_nuc_orig = factors[cond,j]*V_k_nuc_orig

        V_k = np.einsum('ijk,kj->ji', coeffs*U_k[:,i_dft,:].T, Q_k)
        V_k_nuc = np.einsum('ijk,kj->ji',
                            coeffs[even]*U_k[:,i_dft,:][even,:].T,
                            Q_k[even,:])[cond]

        F_orig = np.sum(factors*V_k, axis=1)
        F_nuc_orig = np.sum(factors[cond,:]*V_k_nuc, axis=1)

        U_r_cand = np.random.random(n_prod)

        U_r[i::n] = U_r_cand

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))

        U_k = U_k.reshape(n_prod,n_uvw,n_atm)

        U_k_cand = U_k[...,j].copy()

        V_k_cand = np.einsum('jk,kj->j', coeffs*U_k_cand[:,i_dft].T, Q_k)
        V_k_nuc_cand = np.einsum('jk,kj->j',
                                 coeffs[even]*U_k_cand[:,i_dft][even,:].T,
                                 Q_k[even,:])[cond]

        p_cand = factors[:,j]*V_k_cand
        p_nuc_cand = factors[cond,j]*V_k_nuc_cand

        V_k = np.einsum('ijk,kj->ji', coeffs*U_k[:,i_dft,:].T, Q_k)
        V_k_nuc = np.einsum('ijk,kj->ji',
                            coeffs[even]*U_k[:,i_dft,:][even,:].T,
                            Q_k[even,:])[cond]

        F_cand = np.sum(factors*V_k, axis=1)
        F_nuc_cand = np.sum(factors[cond,:]*V_k_nuc, axis=1)

        F = F_cand.copy()
        F_nuc = F_nuc_cand.copy()

        F_cand[:] = 0
        F_nuc_cand[:] = 0

        U_k_orig = U_k_orig.flatten()
        U_k_cand = U_k_cand.flatten()
        Q_k = Q_k.flatten()

        factors = factors.flatten()

        refinement.displacive_structure_factor(F_cand, F_nuc_cand,
                                               p_cand, p_nuc_cand,
                                               V_k_cand, V_k_nuc_cand,
                                               U_k_cand,
                                               F_orig, F_nuc_orig,
                                               p_orig, p_nuc_orig,
                                               V_k_orig, V_k_nuc_orig,
                                               U_k_orig, Q_k, factors,
                                               coeffs, even, bragg, i_dft,
                                               p, j, n_atm)

        np.testing.assert_array_almost_equal(F_cand, F)
        np.testing.assert_array_almost_equal(F_nuc_cand, F_nuc)
        
    def test_nonmagnetic_structure_factor(self):

        n_hkl = 101

        nu, nv, nw, n_atm = 2, 3, 4, 3

        n_uvw = nu*nv*nw
        n = n_uvw*n_atm

        p = 2
        n_prod = 10

        even = np.array([0, 4, 5, 6, 7, 8, 9])

        factors =    np.random.random((n_hkl,n_atm))\
                + 1j*np.random.random((n_hkl,n_atm))

        i_dft = np.random.randint(n_uvw, size=n_hkl)

        Q_k = np.random.random((n_prod,n_hkl))

        coeffs = np.random.random(n_prod)+1j*np.random.random(n_prod)

        cond = np.random.random(n_hkl) < 0.5
        bragg = np.arange(n_hkl)[cond]

        U_r = np.random.random((n_prod,nu,nv,nw,n_atm)).flatten()
        A_r = np.random.random((nu,nv,nw,n_atm)).flatten()
        
        A_r = np.tile(A_r, n_prod)

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))
        A_k = np.fft.ifftn(A_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))

        U_k = U_k.reshape(n_prod,n_uvw,n_atm)
        A_k = A_k.reshape(n_prod,n_uvw,n_atm)

        i = np.random.randint(n)
        j = i % n_atm

        U_k_orig = U_k[...,j].copy()
        A_k_orig = A_k[...,j].copy()    

        V_k_orig = np.einsum('jk,kj->j', coeffs*(U_k_orig[:,i_dft]+
                                                 A_k_orig[:,i_dft]).T, Q_k)
        V_k_nuc_orig = np.einsum('jk,kj->j',
                                 coeffs[even]*(U_k_orig[:,i_dft][even,:]+
                                               A_k_orig[:,i_dft][even,:]).T,
                                 Q_k[even,:])[cond]

        p_orig = factors[:,j]*V_k_orig
        p_nuc_orig = factors[cond,j]*V_k_nuc_orig

        V_k = np.einsum('ijk,kj->ji', coeffs*(U_k[:,i_dft,:]+
                                              A_k[:,i_dft,:]).T, Q_k)
        V_k_nuc = np.einsum('ijk,kj->ji',
                            coeffs[even]*(U_k[:,i_dft,:][even,:]+
                                          A_k[:,i_dft,:][even,:]).T,
                            Q_k[even,:])[cond]

        F_orig = np.sum(factors*V_k, axis=1)
        F_nuc_orig = np.sum(factors[cond,:]*V_k_nuc, axis=1)
        
        U_r_orig = U_r.reshape(n_prod,n_uvw*n_atm)[:,i]

        U_r_cand = np.random.random(n_prod)
        A_r_cand = np.random.random()*U_r_orig

        U_r[i::n] = U_r_cand
        A_r[i::n] = A_r_cand

        U_k = np.fft.ifftn(U_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))
        A_k = np.fft.ifftn(A_r.reshape(n_prod,nu,nv,nw,n_atm), axes=(1,2,3))

        U_k = U_k.reshape(n_prod,n_uvw,n_atm)
        A_k = A_k.reshape(n_prod,n_uvw,n_atm)

        U_k_cand = U_k[...,j].copy()
        A_k_cand = A_k[...,j].copy()

        V_k_cand = np.einsum('jk,kj->j', coeffs*(U_k_cand[:,i_dft]+
                                                 A_k_cand[:,i_dft]).T, Q_k)
        V_k_nuc_cand = np.einsum('jk,kj->j',
                                 coeffs[even]*(U_k_cand[:,i_dft][even,:]+
                                               A_k_cand[:,i_dft][even,:]).T,
                                 Q_k[even,:])[cond]

        p_cand = factors[:,j]*V_k_cand
        p_nuc_cand = factors[cond,j]*V_k_nuc_cand

        V_k = np.einsum('ijk,kj->ji', coeffs*(U_k[:,i_dft,:]+
                                              A_k[:,i_dft,:]).T, Q_k)
        V_k_nuc = np.einsum('ijk,kj->ji',
                            coeffs[even]*(U_k[:,i_dft,:][even,:]+
                                          A_k[:,i_dft,:][even,:]).T,
                            Q_k[even,:])[cond]

        F_cand = np.sum(factors*V_k, axis=1)
        F_nuc_cand = np.sum(factors[cond,:]*V_k_nuc, axis=1)

        F = F_cand.copy()
        F_nuc = F_nuc_cand.copy()

        F_cand[:] = 0
        F_nuc_cand[:] = 0

        U_k_orig = U_k_orig.flatten()
        A_k_orig = A_k_orig.flatten()
        U_k_cand = U_k_cand.flatten()
        A_k_cand = A_k_cand.flatten()
        Q_k = Q_k.flatten()

        factors = factors.flatten()

        refinement.nonmagnetic_structure_factor(F_cand, F_nuc_cand,
                                                p_cand, p_nuc_cand,
                                                V_k_cand, V_k_nuc_cand,
                                                U_k_cand, A_k_cand,
                                                F_orig, F_nuc_orig,
                                                p_orig, p_nuc_orig,
                                                V_k_orig, V_k_nuc_orig,
                                                U_k_orig, A_k_orig, 
                                                Q_k, factors,
                                                coeffs, even, bragg, i_dft,
                                                p, j, n_atm)

        np.testing.assert_array_almost_equal(F_cand, F)
        np.testing.assert_array_almost_equal(F_nuc_cand, F_nuc)
        
if __name__ == '__main__':
    unittest.main()