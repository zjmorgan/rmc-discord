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
                
        np.testing.assert_array_equal(array, [[0,2,1],
                                              [1,2,-3], 
                                              [2,0,1],
                                              [-1,2,3]])
        
        np.testing.assert_array_equal(ind, [1,5,0,6])
        np.testing.assert_array_equal(inv, [2,0,2,0,2,1,3,3])
        
    def test_evaluate(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        coordinates = [1,2,-3]
        
        x, y, z = coordinates
        
        coord = symmetry.evaluate(operator, coordinates, translate=True)        
        np.testing.assert_array_almost_equal(coord, [-y+1/2,x-y,z-1/2])
        
        coord = symmetry.evaluate(operator, coordinates, translate=False)        
        np.testing.assert_array_almost_equal(coord, [-y,x-y,z])
        
    def test_evaluate_mag(self):
        
        operator = u'-mx,-my,-mz'
        moments = [1,2,-3]
        
        mx, my, mz = moments
        
        mom = symmetry.evaluate_mag(operator, moments)        
        np.testing.assert_array_almost_equal(mom, [-mx,-my,-mz])
        
    def test_evaluate_disp(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        displacements = [1.35,2.12,3.04,0.1,0.2,-0.3]
        
        U11, U22, U33, U23, U13, U12 = displacements
        
        disp = symmetry.evaluate_disp(operator, displacements)        
        np.testing.assert_array_almost_equal(disp, [U22, U11+U22-2*U12, U33, 
                                                    U13-U23, -U23, U22-U12])
        
    def test_reverse(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        coordinates = [1,2,-3]
        
        x, y, z = coordinates
        
        coord = symmetry.evaluate(operator, coordinates, translate=True)     
        rev_operator = symmetry.reverse(operator)[0]
        coordinates = symmetry.evaluate(rev_operator, coord, translate=True)     
        
        np.testing.assert_array_almost_equal(coordinates, [x,y,z])
        
    def test_inverse(self):
        
        operator = u'-y+1/2,x-y,z-1/2'
        
        inv_operator = symmetry.inverse(operator)[0]
        
        self.assertEqual(inv_operator, '-x-y,x,z')
        
    def test_binary(self):
        
        operator0 = u'-y+1/2,x-y,z-1/2'
        operator1 = u'-x,-y,-z'
        
        operator = symmetry.binary(operator0, operator1)
        self.assertEqual(operator, 'y+1/2,-x+y,-z-1/2')
        
        operator = symmetry.binary(operator1, operator0)
        self.assertEqual(operator, 'y-1/2,-x+y,-z+1/2')
        
    def test_classification(self):
        
        operator = u'-z,-x+1/2,y'
        
        rotation, k, wg = symmetry.classification(operator)
        
        self.assertEqual(rotation, '3')
        self.assertEqual(k, 3)
        
        np.testing.assert_array_almost_equal(wg, [-1/6,1/6,1/6])
        
        operator = u'-y+1/2,-x,z+3/4'
        
        rotation, k, wg = symmetry.classification(operator)
        
        self.assertEqual(rotation, 'm')
        self.assertEqual(k, 2)
        
        np.testing.assert_array_almost_equal(wg, [1/4,-1/4,3/4])
        
    def test_absence(self):
        
        operators = [u'x,y,z',
                     u'x+1/2,y+1/2,z+1/2',
                     u'x,-y,-z',
                     u'-y,-x,-z-1/4']
        
        absent = symmetry.absence(operators, 1, 1, 0)
        self.assertEqual(absent, False)
        
        absent = symmetry.absence(operators, 1, 1, 1)
        self.assertEqual(absent, True)        
        
        absent = symmetry.absence(operators, 0, 0, 4)
        self.assertEqual(absent, False)
        
        absent = symmetry.absence(operators, 0, 0, 5)
        self.assertEqual(absent, True)
        
if __name__ == '__main__':
    unittest.main()