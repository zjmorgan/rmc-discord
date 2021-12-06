#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import symmetry, crystal

class test_symmetry(unittest.TestCase):
    
    def test_translation_operator(self):

        self.assertEqual(symmetry.translation_operator(0.0), '+0')
        self.assertEqual(symmetry.translation_operator(1.0), '+1')
        self.assertEqual(symmetry.translation_operator(-1.0), '-1')

        self.assertEqual(symmetry.translation_operator(3/4), '+3/4')
        self.assertEqual(symmetry.translation_operator(-3/4), '-3/4')

        self.assertEqual(symmetry.translation_operator(2/3), '+2/3')
        self.assertEqual(symmetry.translation_operator(-2/3), '-2/3')

        self.assertEqual(symmetry.translation_operator(1/2), '+1/2')
        self.assertEqual(symmetry.translation_operator(-1/2), '-1/2')

        self.assertEqual(symmetry.translation_operator(1/3), '+1/3')
        self.assertEqual(symmetry.translation_operator(-1/3), '-1/3')

        self.assertEqual(symmetry.translation_operator(1/4), '+1/4')
        self.assertEqual(symmetry.translation_operator(-1/4), '-1/4')

        self.assertEqual(symmetry.translation_operator(3/2), '+3/2')
        self.assertEqual(symmetry.translation_operator(-3/2), '-3/2')

        self.assertEqual(symmetry.translation_operator(0.999), '+1')
        self.assertEqual(symmetry.translation_operator(0.499), '+1/2')

        self.assertEqual(symmetry.translation_operator(1.333), '+4/3')
        self.assertEqual(symmetry.translation_operator(1.666), '+5/3')

        self.assertEqual(symmetry.translation_operator(1.234), '+11/9')

        self.assertEqual(symmetry.translation_operator(1/10), '+1/10')
        self.assertEqual(symmetry.translation_operator(3/10), '+3/10')
        self.assertEqual(symmetry.translation_operator(7/10), '+7/10')
        self.assertEqual(symmetry.translation_operator(9/10), '+9/10')

        self.assertEqual(symmetry.translation_operator(1/9), '+1/9')
        self.assertEqual(symmetry.translation_operator(2/9), '+2/9')
        self.assertEqual(symmetry.translation_operator(4/9), '+4/9')
        self.assertEqual(symmetry.translation_operator(5/9), '+5/9')
        self.assertEqual(symmetry.translation_operator(7/9), '+7/9')
        self.assertEqual(symmetry.translation_operator(8/9), '+8/9')

        self.assertEqual(symmetry.translation_operator(1/8), '+1/8')
        self.assertEqual(symmetry.translation_operator(3/8), '+3/8')
        self.assertEqual(symmetry.translation_operator(5/8), '+5/8')
        self.assertEqual(symmetry.translation_operator(7/8), '+7/8')

        self.assertEqual(symmetry.translation_operator(1/6), '+1/6')
        self.assertEqual(symmetry.translation_operator(5/6), '+5/6')

        self.assertEqual(symmetry.translation_operator(1/7), '+1/7')
        self.assertEqual(symmetry.translation_operator(2/7), '+2/7')
        self.assertEqual(symmetry.translation_operator(3/7), '+3/7')
        self.assertEqual(symmetry.translation_operator(4/7), '+4/7')
        self.assertEqual(symmetry.translation_operator(5/7), '+5/7')
        self.assertEqual(symmetry.translation_operator(6/7), '+6/7')

        self.assertEqual(symmetry.translation_operator(1/5), '+1/5')
        self.assertEqual(symmetry.translation_operator(2/5), '+2/5')
        self.assertEqual(symmetry.translation_operator(3/5), '+3/5')
        self.assertEqual(symmetry.translation_operator(4/5), '+4/5')
        
    def test_rotation_operator(self):
        
        self.assertEqual(symmetry.rotation_operator(0.0, col=0), '')
        self.assertEqual(symmetry.rotation_operator(1.0, col=1), '+y')
        self.assertEqual(symmetry.rotation_operator(-1.0, col=2), '-z')
        
        self.assertEqual(symmetry.rotation_operator(3/4, col=0), '3*x/4')
        self.assertEqual(symmetry.rotation_operator(-3/4, col=0), '-3*x/4')
        
        self.assertEqual(symmetry.rotation_operator(2/3, col=0), '2*x/3')
        self.assertEqual(symmetry.rotation_operator(-2/3, col=0), '-2*x/3')
        
        self.assertEqual(symmetry.rotation_operator(1/2, col=0), 'x/2')
        self.assertEqual(symmetry.rotation_operator(-1/2, col=0), '-x/2')
    
        self.assertEqual(symmetry.rotation_operator(1/3, col=0), 'x/3')
        self.assertEqual(symmetry.rotation_operator(-1/3, col=0), '-x/3')
        
        self.assertEqual(symmetry.rotation_operator(1/4, col=0), 'x/4')
        self.assertEqual(symmetry.rotation_operator(-1/4, col=0), '-x/4')
        
        self.assertEqual(symmetry.rotation_operator(3/2, col=0), '3*x/2')
        self.assertEqual(symmetry.rotation_operator(-3/2, col=0), '-3*x/2')
        
        self.assertEqual(symmetry.rotation_operator(0.999, col=0), 'x')
        self.assertEqual(symmetry.rotation_operator(0.499, col=0), 'x/2')
        
        self.assertEqual(symmetry.rotation_operator(1.333, col=0), '4*x/3')
        self.assertEqual(symmetry.rotation_operator(1.666, col=0), '5*x/3')
        
        self.assertEqual(symmetry.rotation_operator(1.234, col=0), '11*x/9')
        
        self.assertEqual(symmetry.rotation_operator(1/10, col=0), 'x/10')
        self.assertEqual(symmetry.rotation_operator(3/10, col=0), '3*x/10')
        self.assertEqual(symmetry.rotation_operator(7/10, col=0), '7*x/10')
        self.assertEqual(symmetry.rotation_operator(9/10, col=0), '9*x/10')
        
        self.assertEqual(symmetry.rotation_operator(1/9, col=0), 'x/9')
        self.assertEqual(symmetry.rotation_operator(2/9, col=0), '2*x/9')
        self.assertEqual(symmetry.rotation_operator(4/9, col=0), '4*x/9')
        self.assertEqual(symmetry.rotation_operator(5/9, col=0), '5*x/9')
        self.assertEqual(symmetry.rotation_operator(7/9, col=0), '7*x/9')
        self.assertEqual(symmetry.rotation_operator(8/9, col=0), '8*x/9')
        
        self.assertEqual(symmetry.rotation_operator(1/8, col=0), 'x/8')
        self.assertEqual(symmetry.rotation_operator(3/8, col=0), '3*x/8')
        self.assertEqual(symmetry.rotation_operator(5/8, col=0), '5*x/8')
        self.assertEqual(symmetry.rotation_operator(7/8, col=0), '7*x/8')
        
        self.assertEqual(symmetry.rotation_operator(1/6, col=0), 'x/6')
        self.assertEqual(symmetry.rotation_operator(5/6, col=0), '5*x/6')
        
        self.assertEqual(symmetry.rotation_operator(1/7, col=0), 'x/7')
        self.assertEqual(symmetry.rotation_operator(2/7, col=0), '2*x/7')
        self.assertEqual(symmetry.rotation_operator(3/7, col=0), '3*x/7')
        self.assertEqual(symmetry.rotation_operator(4/7, col=0), '4*x/7')
        self.assertEqual(symmetry.rotation_operator(5/7, col=0), '5*x/7')
        self.assertEqual(symmetry.rotation_operator(6/7, col=0), '6*x/7')
        
        self.assertEqual(symmetry.rotation_operator(1/5, col=0), 'x/5')
        self.assertEqual(symmetry.rotation_operator(2/5, col=0), '2*x/5')
        self.assertEqual(symmetry.rotation_operator(3/5, col=0), '3*x/5')
        self.assertEqual(symmetry.rotation_operator(4/5, col=0), '4*x/5')
        
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

        operator = [u'-y+1/2,x-y,z-1/2']
        coordinate = [1,2,-3]

        x, y, z = coordinate

        uvw = symmetry.evaluate(operator, coordinate, translate=True)
        np.testing.assert_array_almost_equal(uvw, [[-y+1/2,x-y,z-1/2]])

        uvw = symmetry.evaluate(operator, coordinate, translate=False)
        np.testing.assert_array_almost_equal(uvw, [[-y,x-y,z]])

        coordinates = [np.random.random(4),
                       np.random.random(4),
                       np.random.random(4)]

        x, y, z = coordinates

        uvw = symmetry.evaluate(operator, coordinates, translate=True)
        np.testing.assert_array_almost_equal(uvw, [[-y+1/2,x-y,z-1/2]])

        uvw = symmetry.evaluate(operator, coordinates, translate=False)
        np.testing.assert_array_almost_equal(uvw, [[-y,x-y,z]])

        operators = [u'-y+1/2,x-y,z-1/2',u'-y-1/2,y-x,z+1/2',u'z,x,y']

        x, y, z = coordinate

        uvw = symmetry.evaluate(operators, coordinate, translate=True)
        np.testing.assert_array_almost_equal(uvw, [[-y+1/2,x-y,z-1/2],
                                                   [-y-1/2,y-x,z+1/2],
                                                   [z,x,y]])

        uvw = symmetry.evaluate(operators, coordinate, translate=False)
        np.testing.assert_array_almost_equal(uvw, [[-y,x-y,z],
                                                   [-y,y-x,z],
                                                   [z,x,y]])

    def test_evaluate_mag(self):

        operator = [u'-mx,-my,-mz']
        moments = [1,2,-3]

        mx, my, mz = moments

        mom = symmetry.evaluate_mag(operator, moments)
        np.testing.assert_array_almost_equal(mom, [[-mx,-my,-mz]])

    def test_evaluate_disp(self):

        operator = [u'-y+1/2,x-y,z-1/2']
        displacements = [1.35,2.12,3.04,0.1,0.2,-0.3]

        U11, U22, U33, U23, U13, U12 = displacements

        disp = symmetry.evaluate_disp(operator, displacements)
        np.testing.assert_array_almost_equal(disp, [U22, U11+U22-2*U12, U33,
                                                    U13-U23, -U23, U22-U12])

    def test_reverse(self):

        operator = [u'-y+1/2,x-y,z-1/2']
        coordinates = [1,2,-3]

        x, y, z = coordinates

        uvw = symmetry.evaluate(operator, coordinates, translate=True)

        rev_operator = symmetry.reverse(operator)

        coordinates = np.array(uvw).flatten().tolist()

        uvw = symmetry.evaluate(rev_operator, coordinates, translate=True)

        np.testing.assert_array_almost_equal(uvw, [[x,y,z]])

    def test_inverse(self):

        operator = [u'-y+1/2,x-y,z-1/2']

        inv_operator = symmetry.inverse(operator)

        self.assertEqual(inv_operator, ['-x-y,x,z'])

    def test_binary(self):

        operator0 = [u'-y+1/2,x-y,z-1/2']
        operator1 = [u'-x,-y,-z']

        operator = symmetry.binary(operator0, operator1)
        self.assertEqual(operator, ['y+1/2,-x+y,-z-1/2'])

        operator = symmetry.binary(operator1, operator0)
        self.assertEqual(operator, ['y-1/2,-x+y,-z+1/2'])

        operator0 = [u'-y+1/2,x-y,z-1/2',u'-x,-y,-z']
        operator1 = [u'-x,-y,-z',u'-y+1/2,x-y,z-1/2']

        operator = symmetry.binary(operator0, operator1)
        self.assertEqual(operator, ['y+1/2,-x+y,-z-1/2','y-1/2,-x+y,-z+1/2'])

    def test_classification(self):

        operator = [u'-z,-x+1/2,y']

        rotation, k, wg = symmetry.classification(operator)

        self.assertEqual(rotation, ['3'])
        self.assertEqual(k, [3])

        np.testing.assert_array_almost_equal(wg, [[-1/6,1/6,1/6]])

        operator = [u'-y+1/2,-x,z+3/4']

        rotation, k, wg = symmetry.classification(operator)

        self.assertEqual(rotation, ['m'])
        self.assertEqual(k, [2])

        np.testing.assert_array_almost_equal(wg, [[1/4,-1/4,3/4]])

        operator = [u'-z,-x+1/2,y',u'-y+1/2,-x,z+3/4']

        rotation, k, wg = symmetry.classification(operator)

        self.assertEqual(rotation, ['3','m'])
        self.assertEqual(k, [3,2])

        np.testing.assert_array_almost_equal(wg, [[-1/6,1/6,1/6],
                                                  [1/4,-1/4,3/4]])

    def test_absence(self):

        operators = [u'x,y,z',u'x+1/2,y+1/2,z+1/2',
                     u'x,-y,-z',u'-y,-x,-z-1/4']

        absent = symmetry.absence(operators, 1, 1, 0)
        self.assertEqual(absent, False)

        absent = symmetry.absence(operators, 1, 1, 1)
        self.assertEqual(absent, True)

        absent = symmetry.absence(operators, 0, 0, 4)
        self.assertEqual(absent, False)

        absent = symmetry.absence(operators, 0, 0, 5)
        self.assertEqual(absent, True)

    def test_site(self):

        a = b = 10
        c = 13

        alpha = beta = np.pi/2
        gamma = 2*np.pi/3

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        operators = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',
                     u'-x,-y,z',u'y,-x+y,z',u'x-y,x,z']

        coordinates = [0.35,0.65,0.1234]
        pg, mult, sp_pos = symmetry.site(operators, coordinates, A, tol=1e-1)
        self.assertEqual(pg, '1')
        self.assertEqual(mult, 6)
        self.assertEqual(sp_pos, 'x,y,z')

        coordinates = [0.5,0.0,0.1234]
        pg, mult, sp_pos = symmetry.site(operators, coordinates, A, tol=1e-1)
        self.assertEqual(pg, '2')
        self.assertEqual(mult, 3)
        self.assertEqual(sp_pos, '1/2,0,z')
        
        coordinates = [0.3333,0.6667,0.1234]
        pg, mult, sp_pos = symmetry.site(operators, coordinates, A, tol=1e-1)
        self.assertEqual(pg, '3')
        self.assertEqual(mult, 2)
        self.assertEqual(sp_pos, '1/3,2/3,z')

        coordinates = [0.0,0.0,0.1234]
        pg, mult, sp_pos = symmetry.site(operators, coordinates, A, tol=1e-1)
        self.assertEqual(pg, '6')
        self.assertEqual(mult, 1)
        self.assertEqual(sp_pos, '0,0,z')

        gamma = np.pi/2

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        operators = [u'x,y,z',u'-x,-y,z',
                     u'-y+1/2,x+1/2,z',u'y+1/2,-x+1/2,z',
                     u'-x+1/2,y+1/2,-z',u'x+1/2,-y+1/2,-z',
                     u'y,x,-z',u'-y,-x,-z']

        coordinates = [0.2,0.2,0.5]
        pg, mult, sp_pos = symmetry.site(operators, coordinates, A, tol=1e-1)
        self.assertEqual(pg, '2')
        self.assertEqual(mult, 4)
        self.assertEqual(sp_pos, 'x/2+y/2,x/2+y/2,1/2')

if __name__ == '__main__':
    unittest.main()