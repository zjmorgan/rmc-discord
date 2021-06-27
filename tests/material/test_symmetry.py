#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import symmetry

class test_symmetry(unittest.TestCase):
        
    def test_unique(self):
        
        data = np.array([[2,0,1],
                         [0,2,1],
                         [2,0,1],
                         [0,2,1],
                         [2,0,1],
                         [1,2,-3],
                         [-1,2,3],
                         [-1,2,3]])
        
        array, ind, inv = symmetry.unique(data)
                
        np.testing.assert_array_equal(array, np.array([[0,2,1],
                                                       [1,2,-3],
                                                       [2,0,1],
                                                       [-1,2,3]]))
        
        np.testing.assert_array_equal(ind, np.array([1,5,0,6]))
        np.testing.assert_array_equal(inv, np.array([2,0,2,0,2,1,3,3]))
        
    def test_evaluate(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        coordinates = [1,2,-3]
        
        x, y, z = coordinates
        
        coord = symmetry.evaluate(operator, coordinates, translate=True)        
        np.testing.assert_array_almost_equal(coord, 
                                             np.array([-y+1/2,x-y,z-1/2]))
        
        coord = symmetry.evaluate(operator, coordinates, translate=False)        
        np.testing.assert_array_almost_equal(coord, np.array([-y,x-y,z]))
        
    def test_evaluate_mag(self):
        
        operator = u'-mx,-my,-mz'
        moments = [1,2,-3]
        
        mx, my, mz = moments
        
        mom = symmetry.evaluate_mag(operator, moments)        
        np.testing.assert_array_almost_equal(mom, np.array([-mx,-my,-mz]))
        
    def test_reverse(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        coordinates = [1,2,-3]
        
        x, y, z = coordinates
        
        coord = symmetry.evaluate(operator, coordinates, translate=True)     
        rev_operator = symmetry.reverse(operator)[0]
        coordinates = symmetry.evaluate(rev_operator, coord, translate=True)     
        
        np.testing.assert_array_almost_equal(coordinates, np.array([x,y,z]))
        
    def test_inverse(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        
        inv_operator = symmetry.inverse(operator)[0]
        
        self.assertEqual(inv_operator, np.array(['-x-y,x,z']))
        
    def test_binary(self):
        
        operator0 = u'-y+1/2,x-y,z-1/2'
        operator1 = u'-x,-y,-z'
        
        operator = symmetry.binary(operator0, operator1)
        self.assertEqual(operator, 'y+1/2,-x+y,-z-1/2')
        
        operator = symmetry.binary(operator1, operator0)
        self.assertEqual(operator, 'y-1/2,-x+y,-z+1/2')
        
if __name__ == '__main__':
    unittest.main()