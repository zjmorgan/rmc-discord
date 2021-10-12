#!/usr/bin/env python3

import unittest

from disorder.material import tables

class test_tables(unittest.TestCase):
    
    def test_magnetic_form_factor_coefficients_j0(self):
        
        j0 = tables.j0
        
        j0_fe3p = j0.get('Fe3+')
        
        self.assertEqual(len(j0_fe3p), 7)
        
        coeff_fe3p = (0.3972, 13.2442, 0.6295, 4.9034, -0.0314, 0.3496, 0.0044)
        
        for i, c in enumerate(coeff_fe3p):
            self.assertAlmostEqual(j0_fe3p[i], c, 6)
            
    def test_magnetic_form_factor_coefficients_j2(self):
        
        j2 = tables.j2
        
        j2_fe3p = j2.get('Fe3+')
        
        self.assertEqual(len(j2_fe3p), 7)
        
        coeff_fe3p = (1.649, 16.5593, 1.9064, 6.1325, 0.5206, 2.137, 0.0035)
        
        for i, c in enumerate(coeff_fe3p):
            self.assertAlmostEqual(j2_fe3p[i], c, 6)
    
    def test_neutron_scattering_length_b(self):
        
        bc = tables.bc
        
        bc_O = bc.get('O')
                
        self.assertAlmostEqual(bc_O.real, 5.803, 6)
        self.assertAlmostEqual(bc_O.imag, 0, 6)
        
    def test_xray_form_factor_coefficients(self):
        
        X = tables.X
        
        X_mn3p = X.get('Mn3+')
        
        self.assertEqual(len(X_mn3p), 9)
        
        coeff_mn3p = (9.84521, 4.91797, 7.87194, 
                      0.294393, 3.56531, 10.8171, 
                      0.323613, 24.1281, 0.393974)
        
        for i, c in enumerate(coeff_mn3p):
            self.assertAlmostEqual(X_mn3p[i], c, 6)
        
    def test_electron_form_factor_coefficients(self):
        
        E = tables.E
        
        E_co = E.get('Co')
        
        self.assertEqual(len(E_co), 10)
        
        coeff_co = (0.573, 0.3799, 1.9219, 3.1572, 2.3358, 
                    17.8168, 2.0177, 68.4867, 0.0, 0.0)
        
        for i, c in enumerate(coeff_co):
            self.assertAlmostEqual(E_co[i], c, 6)
            
    def test_atomic_number(self):
        
        Z = tables.Z
        
        Z_og = Z.get('Og')
        
        self.assertEqual(Z_og, (118,))
        
    def test_space_group(self):
        
        sg = tables.sg
        
        sg_62 = sg.get('Pnma')
        sg_227 = sg.get('Fd-3m')
        
        self.assertEqual(sg_62, (62,))
        self.assertEqual(sg_227, (227,))
        
    def test_element_radii(self):
        
        r = tables.r
        
        r_Sn = r.get('Sn')
        
        self.assertEqual(r_Sn, (1.58,2.02,0.69,))
        
    def test_element_colors(self):
        
        rgb = tables.rgb
        
        r_Zn = rgb.get('Zn')
        
        self.assertEqual(r_Zn, (0.49,0.502,0.69,))
        
if __name__ == '__main__':
    unittest.main()