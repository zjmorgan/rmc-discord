#!/usr/bin/env python3

import unittest
import numpy as np
import scipy.linalg

from disorder.material import crystal
from disorder.material import symmetry

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_crystal(unittest.TestCase):
    
    def test_unitcell(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
                                        
        u, v, w, atm, n_atm = crystal.unitcell(folder=folder, 
                                               filename='Cu3Au.cif', 
                                               occupancy=False, 
                                               displacement=False, 
                                               site=False, 
                                               tol=1e-4)
                
        self.assertEqual(n_atm, 4)
                
        np.testing.assert_array_almost_equal(u[atm == 'Au'], 0)
        np.testing.assert_array_almost_equal(v[atm == 'Au'], 0)
        np.testing.assert_array_almost_equal(w[atm == 'Au'], 0)
        
        u, v, w, site, atm, n_atm = crystal.unitcell(folder=folder, 
                                                     filename='Cu3Au.cif', 
                                                     occupancy=False, 
                                                     displacement=False, 
                                                     site=True, 
                                                     tol=1e-4)
               
        np.testing.assert_array_equal(atm[site == 0], 'Au')
        np.testing.assert_array_equal(atm[site == 1], 'Cu') 
        
        u, v, w, occ, atm, n_atm = crystal.unitcell(folder=folder, 
                                                    filename='Cu3Au.cif', 
                                                    occupancy=True, 
                                                    displacement=False, 
                                                    site=False, 
                                                    tol=1e-4)
               
        np.testing.assert_array_almost_equal(occ, 1)
        
        u, v, w, disp, atm, n_atm = crystal.unitcell(folder=folder, 
                                                     filename='Cu3Au.cif', 
                                                     occupancy=False, 
                                                     displacement=True, 
                                                     site=False, 
                                                     tol=1e-4)
        
        np.testing.assert_array_almost_equal(disp, 0)

        # ---
        
        u, v, w, occ, atm, n_atm = crystal.unitcell(folder=folder, 
                                                    filename='H2O.cif', 
                                                    occupancy=True,
                                                    displacement=False, 
                                                    site=False, 
                                                    tol=1e-4)
        
        np.testing.assert_array_almost_equal(occ[atm == '0'], 1.0)
        np.testing.assert_array_almost_equal(occ[atm == 'H'], 0.5)
        
        u, v, w, disp, atm, n_atm = crystal.unitcell(folder=folder, 
                                                     filename='CaTiOSiO4.cif', 
                                                     occupancy=False, 
                                                     displacement=True, 
                                                     site=False, 
                                                     tol=1e-4)
        
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,0], 0.02200)
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,1], 0.00497)
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,2], 0.00537)
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,5], 0.00069)
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,4], -0.00098)
        np.testing.assert_array_almost_equal(disp[atm == 'Ca'][:,3], 0.00029)
        
        u, \
        v, \
        w, \
        occ, \
        disp, \
        site, \
        op, \
        atm, \
        n_atm = crystal.unitcell(folder=folder, 
                                 filename='chlorastrolite.cif', 
                                 occupancy=True, 
                                 displacement=True, 
                                 site=True, 
                                 operator=True, 
                                 tol=1e-4)
        
        np.testing.assert_array_almost_equal(occ[atm == 'FeX']+\
                                             occ[atm == 'AlX']+\
                                             occ[atm == 'MgX'], 1.0, 2)
        np.testing.assert_array_almost_equal(occ[atm == 'FeY']+\
                                             occ[atm == 'AlY'], 1.0)
            
        np.testing.assert_array_almost_equal(disp[atm == 'FeX'], 
                                             disp[atm == 'AlX'])
        np.testing.assert_array_almost_equal(disp[atm == 'AlX'], 
                                             disp[atm == 'MgX'])
            
        x, y, z = np.array([0.2541,0.242,0.4976])  
        for i in range(8):
            ux, vy, wz = symmetry.evaluate(op[atm == 'FeY'][i], [x,y,z])
            ux += 1*(ux < 0)-1*(ux >= 1)
            vy += 1*(vy < 0)-1*(vy >= 1)
            wz += 1*(wz < 0)-1*(wz >= 1)
            np.testing.assert_array_almost_equal(u[atm == 'FeY'][i], ux)
            np.testing.assert_array_almost_equal(v[atm == 'FeY'][i], vy)
            np.testing.assert_array_almost_equal(w[atm == 'FeY'][i], wz)
        
        # ---
        
        u, \
        v, \
        w, \
        mom, \
        mag_op, \
        atm, \
        n_atm = crystal.unitcell(folder=folder, 
                                 filename='CuMnO2.mcif', 
                                 occupancy=False, 
                                 displacement=False, 
                                 moment=True, 
                                 site=False, 
                                 operator=False, 
                                 magnetic_operator=True, 
                                 tol=1e-4)
        
        np.testing.assert_array_almost_equal(mom[atm == 'Cu'], 0.0)
        np.testing.assert_array_almost_equal(mom[atm == 'O'], 0.0)        
        
        mx, my, mz = np.array([1.8,0.0,1.4])  
        for i in range(8):
            moment = symmetry.evaluate_mag(mag_op[atm == 'Mn'][i], [mx,my,mz])
            np.testing.assert_array_almost_equal(mom[atm == 'Mn'][i], moment)
        
    def test_supercell(self):

        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
                
        u, \
        v, \
        w, \
        occ, \
        disp, \
        mom, \
        atm, \
        n_atm = crystal.unitcell(folder=folder, 
                                 filename='CaTiOSiO4.cif', 
                                 occupancy=True, 
                                 displacement=True, 
                                 moment=True, 
                                 site=False, 
                                 tol=1e-4)
        
        a, \
        b, \
        c, \
        alpha, \
        beta, \
        gamma = crystal.parameters(folder=folder, filename='CaTiOSiO4.cif')
        
        nu, nv, nw = 2, 3, 1
      
        crystal.supercell(atm,
                          occ,
                          disp,
                          mom,
                          u, 
                          v, 
                          w,
                          nu,
                          nv,
                          nw,
                          folder+'/supercell_CaTiOSiO4.cif',
                          folder=folder,
                          filename='CaTiOSiO4.cif')
        
        U, \
        V, \
        W, \
        Occ, \
        Disp, \
        Atm, \
        N_atm = crystal.unitcell(folder=folder, 
                                 filename='supercell_CaTiOSiO4.cif', 
                                 occupancy=True, 
                                 displacement=True, 
                                 site=False, 
                                 tol=1e-4)
        
        A, \
        B, \
        C, \
        Alpha, \
        Beta, \
        Gamma = crystal.parameters(folder=folder, 
                                   filename='supercell_CaTiOSiO4.cif')
        
        n_uvw = nu*nv*nw
        
        i, j, k = np.meshgrid(np.arange(nu), 
                              np.arange(nv), 
                              np.arange(nw), indexing='ij')
        
        i = i.flatten()[:,np.newaxis]
        j = j.flatten()[:,np.newaxis]
        k = k.flatten()[:,np.newaxis]
                
        np.testing.assert_array_almost_equal((i+u).flatten(), nu*U, 4)
        np.testing.assert_array_almost_equal((j+v).flatten(), nv*V, 4)
        np.testing.assert_array_almost_equal((k+w).flatten(), nw*W, 4)
                
        np.testing.assert_array_equal(np.tile(atm, n_uvw), Atm)
        np.testing.assert_array_almost_equal(np.tile(occ, n_uvw), Occ)
        np.testing.assert_array_almost_equal(np.tile(disp.T, n_uvw).T, Disp)
        
        self.assertEqual(n_atm*n_uvw, N_atm)
        
        self.assertAlmostEqual(a*nu, A)
        self.assertAlmostEqual(b*nv, B)
        self.assertAlmostEqual(c*nw, C)
        
        self.assertAlmostEqual(alpha, Alpha)
        self.assertAlmostEqual(beta, Beta)
        self.assertAlmostEqual(gamma, Gamma)
        
        os.remove(folder+'/supercell_CaTiOSiO4.cif')

        # ---
        
        u, \
        v, \
        w, \
        occ, \
        disp, \
        mom, \
        atm, \
        n_atm = crystal.unitcell(folder=folder, 
                                 filename='CuMnO2.mcif', 
                                 occupancy=True, 
                                 displacement=True, 
                                 moment=True, 
                                 site=False, 
                                 tol=1e-4)
                
        nu, nv, nw = 1, 2, 3
      
        crystal.supercell(atm,
                          occ,
                          disp,
                          mom,
                          u, 
                          v, 
                          w,
                          nu,
                          nv,
                          nw,
                          folder+'/supercell_CuMnO2.mcif',
                          folder=folder,
                          filename='CuMnO2.mcif')
        
        U, \
        V, \
        W, \
        Occ, \
        Disp, \
        Mom, \
        Atm, \
        N_atm = crystal.unitcell(folder=folder, 
                                 filename='supercell_CuMnO2.mcif', 
                                 occupancy=True, 
                                 displacement=True, 
                                 moment=True, 
                                 site=False, 
                                 tol=1e-4)
        
        n_uvw = nu*nv*nw
                
        np.testing.assert_array_almost_equal(np.tile(mom.T, n_uvw).T, Mom)
        
        self.assertEqual((atm == 'Cu').sum()*n_uvw, (Atm == 'Cu').sum())
        self.assertEqual((atm == 'Mn').sum()*n_uvw, (Atm == 'Mn').sum())
        self.assertEqual((atm == 'O').sum()*n_uvw, (Atm == 'O').sum())
        
        os.remove(folder+'/supercell_CuMnO2.mcif')
        
    def test_parameters(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
        
        a, \
        b, \
        c, \
        alpha, \
        beta, \
        gamma = crystal.parameters(folder=folder, filename='CaTiOSiO4.cif')
        
        self.assertAlmostEqual(a, 7.069)
        self.assertAlmostEqual(b, 8.722)
        self.assertAlmostEqual(c, 6.566)
        self.assertAlmostEqual(alpha, np.deg2rad(90))
        self.assertAlmostEqual(beta, np.deg2rad(113.86))
        self.assertAlmostEqual(gamma, np.deg2rad(90))
        
    def test_group(self):
        
        folder = os.path.abspath(os.path.join(directory, '..', 'data'))
        
        hm, sg = crystal.group(folder=folder, filename='CaTiOSiO4.cif')
        
        self.assertEqual(sg, 'P121/a1')
        self.assertEqual(hm, 14)
                                     
    def test_d(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        h, k, l = 1, 2, -3

        d = crystal.d(a, b, c, alpha, beta, gamma, h, k, l)

        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        u_, v_, w_ = np.dot(B, [1,0,0]), np.dot(B, [0,1,0]), np.dot(B, [0,0,1])

        self.assertAlmostEqual(d, 1/np.sqrt(np.dot(h*u_+k*v_+l*w_, 
                                                   h*u_+k*v_+l*w_)))

    def test_interplanar(self):

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        h0, k0, l0 = 1, 2, -3
        h1, k1, l1 = 2, -1, -4

        angle = crystal.interplanar(a, b, c, alpha, beta, gamma, \
                                    h0, k0, l0, h1, k1, l1)
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        u_, v_, w_ = np.dot(B, [1,0,0]), np.dot(B, [0,1,0]), np.dot(B, [0,0,1])

        self.assertAlmostEqual(angle, np.arccos(np.dot(h0*u_+k0*v_+l0*w_, 
                                                       h1*u_+k1*v_+l1*w_)
                                      / np.linalg.norm(h0*u_+k0*v_+l0*w_)
                                      / np.linalg.norm(h1*u_+k1*v_+l1*w_)))
        
    def test_volume(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        V = crystal.volume(a, b, c, alpha, beta, gamma)
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)

        u, v, w = np.dot(A, [1,0,0]), np.dot(A, [0,1,0]), np.dot(A, [0,0,1])
        
        self.assertAlmostEqual(V, np.dot(u, np.cross(v, w)))
        
    def test_reciprocal(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        a_, b_, c_, alpha_, beta_, gamma_ = crystal.reciprocal(a, 
                                                               b, 
                                                               c, 
                                                               alpha, 
                                                               beta, 
                                                               gamma)
        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)

        u, v, w = np.dot(A, [1,0,0]), np.dot(A, [0,1,0]), np.dot(A, [0,0,1])
        
        V = np.dot(u, np.cross(v, w))
        
        self.assertAlmostEqual(a_, np.linalg.norm(np.cross(v, w)/V))
        self.assertAlmostEqual(b_, np.linalg.norm(np.cross(w, u)/V))
        self.assertAlmostEqual(c_, np.linalg.norm(np.cross(u, v)/V))
        
        u_, v_, w_ = np.dot(B, [1,0,0]), np.dot(B, [0,1,0]), np.dot(B, [0,0,1])
        
        self.assertAlmostEqual(alpha_, np.arccos(np.dot(v_, w_)/
                                                 np.linalg.norm(v_)/
                                                 np.linalg.norm(w_)))
        self.assertAlmostEqual(beta_, np.arccos(np.dot(w_, u_)/
                                                 np.linalg.norm(w_)/
                                                 np.linalg.norm(u_)))
        self.assertAlmostEqual(gamma_, np.arccos(np.dot(u_, v_)/
                                                 np.linalg.norm(u_)/
                                                 np.linalg.norm(v_)))
    
    def test_metric(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
    
        G = crystal.metric(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = crystal.reciprocal(a, 
                                                               b, 
                                                               c, 
                                                               alpha, 
                                                               beta, 
                                                               gamma)
        
        G_ = crystal.metric(a_, b_, c_, alpha_, beta_, gamma_)
        
        np.testing.assert_array_almost_equal(np.eye(3), np.dot(G, G_))
    
    def test_matrices(self):
        
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
                        
        A, B, R = crystal.matrices(a, b, c, alpha, beta, gamma)
        
        G = crystal.metric(a, b, c, alpha, beta, gamma)
        
        U = scipy.linalg.cholesky(G, lower=False)
        np.testing.assert_array_almost_equal(A, U)
        
        a_, b_, c_, alpha_, beta_, gamma_ = crystal.reciprocal(a, 
                                                               b, 
                                                               c, 
                                                               alpha, 
                                                               beta, 
                                                               gamma)
        
        G_ = crystal.metric(a_, b_, c_, alpha_, beta_, gamma_)
        
        U_ = scipy.linalg.cholesky(G_, lower=False)
        np.testing.assert_array_almost_equal(B, U_)
        
        hkl = np.array([-3.3,1.1,2.5])
        uvw = np.array([4.2,-2.7,1.8])
        
        self.assertAlmostEqual(hkl.dot(uvw), A.dot(uvw).dot(R.dot(B.dot(hkl))))
        
    def test_orthogonalized(self):
        
        a, b, c, alpha, beta, gamma = 5, 5, 7, np.pi/2, np.pi/2, 2*np.pi/3
                                
        C, D = crystal.orthogonalized(a, b, c, alpha, beta, gamma)
        
        a_, b_, c_, alpha_, beta_, gamma_ = crystal.reciprocal(a, 
                                                               b, 
                                                               c, 
                                                               alpha, 
                                                               beta, 
                                                               gamma)

        U = np.array([[1,0.1,0.2],
                      [0.1,3,0.3],
                      [0.2,0.3,4]])
                 
        Up, _ = np.linalg.eig(np.dot(np.dot(D, U), D.T))
        Uiso = np.mean(Up).real
        
        Viso = (U[0,0]*(a*a_)**2+U[1,1]*(b*b_)**2+U[2,2]*(c*c_)**2+\
                2*U[0,1]*a*a_*b*b_*np.cos(gamma)+\
                2*U[0,2]*a*a_*c*c_*np.cos(beta)+\
                2*U[1,2]*b*b_*c*c_*np.cos(alpha))/3
            
        self.assertAlmostEqual(Uiso, Viso)
                
        self.assertAlmostEqual(np.arccos(np.dot(C[:,0],C[:,1])), gamma)
        self.assertAlmostEqual(np.arccos(np.dot(C[:,1],C[:,2])), alpha)
        self.assertAlmostEqual(np.arccos(np.dot(C[:,2],C[:,0])), beta)

    def test_transform(self):
        
        u, v, w = np.array([1,1,0]), np.array([-1,1,2]), np.array([2,-2,2])
        
        u = u/np.linalg.norm(u)
        v = v/np.linalg.norm(v)
        w = w/np.linalg.norm(w)
        
        R = np.stack((u,v,w)).T
        
        x = np.array([3.5, 4, -1.2])        
        y = crystal.transform(x[0], x[1], x[2], R)
        
        np.testing.assert_array_almost_equal(y, np.dot(R, x))
        
    def test_lattice(self):
        
        lat = crystal.lattice(5, 5, 5, np.pi/2, np.pi/2, np.pi/2)
        self.assertEqual(lat, 'Cubic')
        
        lat = crystal.lattice(5, 5, 7, np.pi/2, np.pi/2, 2*np.pi/3)
        self.assertEqual(lat, 'Hexagonal')
        
        lat = crystal.lattice(5, 5, 6, np.pi/2, np.pi/2, np.pi/2)
        self.assertEqual(lat, 'Tetragonal')
        
        lat = crystal.lattice(5, 6, 7, np.pi/2, np.pi/2, np.pi/2)
        self.assertEqual(lat, 'Orthorhombic')
        
        lat = crystal.lattice(5, 5, 5, np.pi/3, np.pi/3, np.pi/3)
        self.assertEqual(lat, 'Rhombohedral')
        
        lat = crystal.lattice(5, 6, 7, np.pi/2, np.pi/3, np.pi/2)
        self.assertEqual(lat, 'Monoclinic')
        
        lat = crystal.lattice(5, 6, 7, np.pi/2, np.pi/2, np.pi/3)
        self.assertEqual(lat, 'Monoclinic')
        
        lat = crystal.lattice(5, 6, 7, np.pi/2, np.pi/3, np.pi/4)
        self.assertEqual(lat, 'Triclinic')
        
if __name__ == '__main__':
    unittest.main()