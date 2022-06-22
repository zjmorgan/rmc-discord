#!/usr/bin/env python3U

import io
import os
import sys

import unittest
import numpy as np

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

        self.assertAlmostEqual(chi_sq, np.sum((2*y_fit-y_obs)**2/e**2), 4)

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

if __name__ == '__main__':
    unittest.main()