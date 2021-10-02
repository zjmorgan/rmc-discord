#!/usr/bin/env python3

import unittest
import numpy as np

from scipy import ndimage

from disorder.diffuse import filters

class test_filters(unittest.TestCase):
    
    def test_boxblur(self):
        
        sigma, n = 2, 3
        boxes = filters.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
    
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
                
        sigma, n = 3, 2
        boxes = filters.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
            
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
        
        sigma, n = 4, 3
        boxes = filters.boxblur(sigma, n)
        
        l = int(np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
            
        self.assertEqual(boxes.size, n)        
        self.assertEqual(np.sum(boxes), n*(l+1)/2-m)
        
        sigma, n = np.array([3,1,2]), 4
        boxes = filters.boxblur(sigma, n)
        
        l = (np.floor(np.floor(np.sqrt(12*sigma**2/n+1))/2-0.5)*2+1).astype(int)
        m = np.round((n*(l*(l+4)+3)-12*sigma**2)/(l+1)/4)
                    
        self.assertEqual(boxes.size, n*sigma.size)     
        
        boxes = boxes.reshape(n, sigma.size)
                
        np.testing.assert_array_equal(np.sum(boxes, axis=0), n*(l+1)/2-m)
        
    def test_gaussian(self):
        
        np.random.seed(13)
        
        nh, nk, nl = 16, 27, 36

        mask = np.random.randint(0, 2, size=(nh,nk,nl), dtype=bool)
        
        sigma = [2,1,3]
        v_inv = filters.gaussian(mask, sigma).reshape(nh,nk,nl)
                
        v = np.ones(mask.shape)
        v[mask] = 0
            
        w = filters.blurring(v, sigma)

        np.testing.assert_array_almost_equal(w*v_inv, np.ones(mask.shape))
        
    def test_boxfilter(self):
        
        np.random.seed(13)
        
        nh, nk, nl = 16, 27, 36

        mask = np.random.randint(0, 2, size=(nh,nk,nl), dtype=bool)
        
        sigma = [2,1,3]
        v_inv = filters.gaussian(mask, sigma)
        
        v = np.ones(mask.shape)
        v[mask] = 0
    
        w = filters.boxfilter(v, mask, sigma, v_inv)
        
        np.testing.assert_array_almost_equal(w, np.ones(mask.shape))
        
    def test_blurring(self):
        
        nh, nk, nl = 16, 27, 36

        v = np.random.random(size=(nh,nk,nl))
        
        sigma = [2,1,3]
        
        w = filters.blurring(v, sigma)
        
        x = ndimage.filters.gaussian_filter(v, sigma, mode='nearest')
        
        np.testing.assert_array_almost_equal(w, x, decimal=1)
    
    def test_median(self):
        
        a = np.random.random((13,14,15))
            
        b = ndimage.median_filter(a, size=3, mode='nearest')
        c = filters.median(a, 3)
    
        np.testing.assert_array_almost_equal(b, c)
         
        b = ndimage.median_filter(a, size=5, mode='nearest')
        c = filters.median(a, 5)
    
        np.testing.assert_array_almost_equal(b, c)
        
if __name__ == '__main__':
    unittest.main()
