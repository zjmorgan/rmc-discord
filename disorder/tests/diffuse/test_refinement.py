#!/usr/bin/env python3U

import io
import os
import sys

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, refinement

class test_refinement(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()