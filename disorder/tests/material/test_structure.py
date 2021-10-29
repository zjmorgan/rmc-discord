#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import structure, crystal

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_structure(unittest.TestCase):
    
    def test_structure(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
                                        
        uc_dict = crystal.unitcell(folder=folder, 
                                   filename='Cu3Au.cif', 
                                   tol=1e-4)
        
        u = uc_dict['u']
        v = uc_dict['v']
        w = uc_dict['w']
        occ = uc_dict['occupancy']
        disp = uc_dict['displacement']
        atm = uc_dict['atom']
        n_atm = uc_dict['n_atom']
        
        constants = crystal.parameters(folder=folder, filename='Cu3Au.cif')
        
        a, b, c, alpha, beta, gamma = constants
        
        print(structure.factor(u, v, w, atm, occ, a, b, c, alpha, beta, gamma, 
                              dmin=0.3, source='Neutron'))
                
if __name__ == '__main__':
    unittest.main()