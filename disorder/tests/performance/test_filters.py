#!/usr/bin/env python3

import unittest
import numpy as np

from scipy import ndimage

from disorder.diffuse import filters

import pstats, cProfile

class test_filters(unittest.TestCase):

    def setUp(self):

        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):

        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()

    # def test_blurring(self):

    #     nh, nk, nl = 121, 241, 31

    #     v = np.random.random(size=(nh,nk,nl))

    #     sigma = [4,6,1]

    #     w = filters.blurring(v, sigma)

    #     x = ndimage.gaussian_filter(v, sigma, mode='nearest')

    #     np.testing.assert_array_almost_equal(w, x, decimal=1)

    # def test_median(self):

    #     a = np.random.random((121,241,31))

    #     b = ndimage.median_filter(a, size=3, mode='nearest')
    #     c = filters.median(a, 3)

    #     np.testing.assert_array_almost_equal(b, c)

    #     b = ndimage.median_filter(a, size=5, mode='nearest')
    #     c = filters.median(a, 5)

    #     np.testing.assert_array_almost_equal(b, c)

if __name__ == '__main__':
    unittest.main()
