#!/usr/bin/env python3U

import unittest
import numpy as np

from disorder.material import crystal, symmetry
from disorder.diffuse import space
from disorder.correlation import functions

class test_functions(unittest.TestCase):

    def test_pairs1d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 3, 4, 8
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co','Ni'])
        u = np.array([0.0,0.2,0.25])
        v = np.array([0.01,0.31,0.1])
        w = np.array([0.1,0.4,0.62])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, A, fract=1.0)
        distance, atm_pair, counts, search, coordinate, N = data
       
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        ns = (1+nu-2*mu)*nv*nw+(1+nv-2*mv)*nw*nu+(1+nw-2*mw)*nu*nv
        
        nd = (1+nu-2*mu)*(1+nv-2*mv)*nw\
           + (1+nv-2*mv)*(1+nw-2*mw)*nu\
           + (1+nw-2*mw)*(1+nu-2*mu)*nv
          
        nt = (1+nu-2*mu)*(1+nv-2*mv)*(1+nw-2*mw)
        
        nr = n_atm**2*n_uvw*(ns-nd+nt)//2 # even removal
        
        self.assertEqual(counts.sum(), nc+nl-nr)
        
        self.assertEqual(search.size, N)
        
        self.assertEqual(atm_pair.shape[0], search.shape[0])
        
        np.testing.assert_array_equal(np.diff(search), counts)
        
        i, j = coordinate.T
                
        Dx = (rx[j]-rx[i])[search[:-1]]
        Dy = (ry[j]-ry[i])[search[:-1]]
        Dz = (rz[j]-rz[i])[search[:-1]]
        
        Du, Dv, Dw = crystal.transform(Dx, Dy, Dz, np.linalg.inv(A))
        
        Du[Du < -mu] += nu
        Dv[Dv < -mv] += nv
        Dw[Dw < -mw] += nw
    
        Du[Du > mu] -= nu
        Dv[Dv > mv] -= nv
        Dw[Dw > mw] -= nw
    
        Dx, Dy, Dz = crystal.transform(Du, Dv, Dw, A)

        Distance = np.sqrt(Dx**2+Dy**2+Dz**2)

        np.testing.assert_array_almost_equal(Distance, distance[:-1])
        
        atm_pair_ij = np.stack((atms[i],atms[j])).T[search[:-1]]
        atm_pair_ij = np.sort(atm_pair_ij, axis=1)
        
        unique_pairs = np.stack([ap.split('_') for ap in atm_pair[:-1]])
        
        np.testing.assert_array_equal(atm_pair_ij, unique_pairs)
        
        dx_c, dy_c, dz_c = crystal.transform(mu, mv, mw, A)
        
        dc = np.sqrt(dx_c**2+dy_c**2+dz_c**2)
                
        np.testing.assert_array_equal(distance <= dc, True)
        
        data = functions.pairs1d(rx, ry, rz, atms, nu, nv, nw, A, fract=0.25)
        distance, atm_pair, counts, search, coordinate, N = data
                
        np.testing.assert_array_equal(distance <= Distance.max()*0.25, True)
        
    def test_pairs3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 4, 2, 8
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co','Ni','Mn'])
        u = np.array([0.0,0.2,0.25,0.25])
        v = np.array([0.01,0.31,0.1,0.1])
        w = np.array([0.1,0.4,0.62,0.62])
        
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        data = functions.pairs3d(rx, ry, rz, atms, nu, nv, nw, A, fract=1.0)
        dx, dy, dz, atm_pair, counts, search, coordinate, N = data
        
        mu = (nu+1) // 2
        mv = (nv+1) // 2
        mw = (nw+1) // 2
        
        n_uvw = nu*nv*nw
        
        nc = n_atm*(n_atm-1)//2*n_uvw
        nl = n_atm**2*n_uvw*(n_uvw-1)//2
        
        ns = (1+nu-2*mu)*nv*nw+(1+nv-2*mv)*nw*nu+(1+nw-2*mw)*nu*nv
        
        nd = (1+nu-2*mu)*(1+nv-2*mv)*nw\
           + (1+nv-2*mv)*(1+nw-2*mw)*nu\
           + (1+nw-2*mw)*(1+nu-2*mu)*nv
          
        nt = (1+nu-2*mu)*(1+nv-2*mv)*(1+nw-2*mw)
        
        nr = n_atm**2*n_uvw*(ns-nd+nt)//2 # even removal
        
        self.assertEqual(counts.sum(), nc+nl-nr)
        
        self.assertEqual(search.size, N)
        
        self.assertEqual(atm_pair.shape[0], search.shape[0])
        
        np.testing.assert_array_equal(np.diff(search), counts)
        
        i, j = coordinate.T
                
        Dx = (rx[j]-rx[i])[search[:-1]]
        Dy = (ry[j]-ry[i])[search[:-1]]
        Dz = (rz[j]-rz[i])[search[:-1]]
        
        Du, Dv, Dw = crystal.transform(Dx, Dy, Dz, np.linalg.inv(A))
        
        Du[Du < -mu] += nu
        Dv[Dv < -mv] += nv
        Dw[Dw < -mw] += nw
    
        Du[Du > mu] -= nu
        Dv[Dv > mv] -= nv
        Dw[Dw > mw] -= nw
        
        Dx, Dy, Dz = crystal.transform(Du, Dv, Dw, A)
        
        Distance = np.sqrt(Dx**2+Dy**2+Dz**2)

        np.testing.assert_array_almost_equal(Dx, dx[:-1])
        np.testing.assert_array_almost_equal(Dy, dy[:-1])
        np.testing.assert_array_almost_equal(Dz, dz[:-1])
        
        atm_pair_ij = np.stack((atms[i],atms[j])).T[search[:-1]]
        atm_pair_ij = np.sort(atm_pair_ij, axis=1)
        
        unique_pairs = np.stack([ap.split('_') for ap in atm_pair[:-1]])
        
        np.testing.assert_array_equal(atm_pair_ij, unique_pairs)
        
        du, dv, dw = crystal.transform(dx, dy, dz, np.linalg.inv(A))
        
        np.testing.assert_array_equal(du <= mu, True)
        np.testing.assert_array_equal(dv <= mv, True)
        np.testing.assert_array_equal(dw <= mw, True)
        
        data = functions.pairs3d(rx, ry, rz, atms, nu, nv, nw, A, fract=0.25)
        dx, dy, dz, atm_pair, counts, search, coordinate, N = data
                
        np.testing.assert_array_equal(dx <= Distance.max()*0.25, True)
        np.testing.assert_array_equal(dy <= Distance.max()*0.25, True)
        np.testing.assert_array_equal(dz <= Distance.max()*0.25, True)
        
    def test_vector1d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 3, 5, 7
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co'])
        u = np.array([0.0,0.2])
        v = np.array([0.0,0.3])
        w = np.array([0.0,0.4])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        Sx = np.zeros((nu,nv,nw,n_atm))
        Sy = np.zeros((nu,nv,nw,n_atm))
        Sz = np.zeros((nu,nv,nw,n_atm))
        
        sx, sy, sz = 1, 1, 1
        theta = np.mod(np.arctan2(sy,sx), 2*np.pi)
        phi = np.arccos(sz/np.sqrt(sx**2+sy**2+sz**2))
        
        Sx[:,:,:,0] = np.sin(phi)*np.cos(theta)
        Sy[:,:,:,0] = np.sin(phi)*np.sin(theta)
        Sz[:,:,:,0] = np.cos(phi)
        
        sx, sy, sz = -1, -1, 2
        theta = np.mod(np.arctan2(sy,sx), 2*np.pi)
        phi = np.arccos(sz/np.sqrt(sx**2+sy**2+sz**2))
        
        Sx[:,:,:,1] = np.sin(phi)*np.cos(theta)
        Sy[:,:,:,1] = np.sin(phi)*np.sin(theta)
        Sz[:,:,:,1] = np.cos(phi)
        
        Sx, Sy, Sz = Sx.flatten(), Sy.flatten(), Sz.flatten()
        
        args = [Sx, Sy, Sz, rx, ry, rz, atms, nu, nv, nw, A, 0.5]
        data = functions.vector1d(*args)
        S_corr, S_coll, S_corr_, S_coll_, distance, atm_pair = data
        
        self.assertAlmostEqual(S_corr[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_coll[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_corr_[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_coll_[np.isclose(distance, 0)], 1.0)
        
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Fe'], 0.0)
        
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Co_Fe'], 0.0)
        
    def test_vector3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 4, 6, 8
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co'])
        u = np.array([0.0,0.2])
        v = np.array([0.0,0.3])
        w = np.array([0.0,0.4])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        Sx = np.zeros((nu,nv,nw,n_atm))
        Sy = np.zeros((nu,nv,nw,n_atm))
        Sz = np.zeros((nu,nv,nw,n_atm))
        
        sx, sy, sz = 2, 1, 1
        theta = np.mod(np.arctan2(sy,sx), 2*np.pi)
        phi = np.arccos(sz/np.sqrt(sx**2+sy**2+sz**2))
        
        Sx[:,:,:,0] = np.sin(phi)*np.cos(theta)
        Sy[:,:,:,0] = np.sin(phi)*np.sin(theta)
        Sz[:,:,:,0] = np.cos(phi)
        
        sx, sy, sz = 1, -1, -1
        theta = np.mod(np.arctan2(sy,sx), 2*np.pi)
        phi = np.arccos(sz/np.sqrt(sx**2+sy**2+sz**2))
        
        Sx[:,:,:,1] = np.sin(phi)*np.cos(theta)
        Sy[:,:,:,1] = np.sin(phi)*np.sin(theta)
        Sz[:,:,:,1] = np.cos(phi)
        
        Sx, Sy, Sz = Sx.flatten(), Sy.flatten(), Sz.flatten()
        
        args = [Sx, Sy, Sz, rx, ry, rz, atms, nu, nv, nw, A, 0.5]
        data = functions.vector3d(*args)
        S_corr, S_coll, S_corr_, S_coll_, dx, dy, dz, atm_pair = data
        
        distance = np.sqrt(dx**2+dy**2+dz**2)
        
        self.assertAlmostEqual(S_corr[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_coll[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_corr_[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_coll_[np.isclose(distance, 0)], 1.0)
        
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Fe'], 0.0)
        
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_coll[atm_pair == 'Co_Fe'], 0.0)

    def test_scalar1d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 5, 5, 6
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co'])
        u = np.array([0.0,0.2])
        v = np.array([0.0,0.3])
        w = np.array([0.0,0.4])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        A_r = np.zeros((nu,nv,nw,n_atm))
        
        A_r[:,:,:,0] = 0.5
        A_r[:,:,:,1] = -1

        A_r = A_r.flatten()
        
        args = [A_r, rx, ry, rz, atms, nu, nv, nw, A, 0.5]
        data = functions.scalar1d(*args)
        S_corr, S_corr_, distance, atm_pair = data
                
        self.assertAlmostEqual(S_corr[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_corr_[np.isclose(distance, 0)], 1.0)
        
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Fe'], -1.0)
        
    def test_scalar3d(self):
    
        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 4, 8, 5
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Co'])
        u = np.array([0.0,0.2])
        v = np.array([0.0,0.3])
        w = np.array([0.0,0.4])
                
        n_atm = atm.shape[0]
        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
        
        A_r = np.zeros((nu,nv,nw,n_atm))
        
        A_r[:,:,:,0] = 0.75
        A_r[:,:,:,1] = -1

        A_r = A_r.flatten()
        
        args = [A_r, rx, ry, rz, atms, nu, nv, nw, A, 0.5]
        data = functions.scalar3d(*args)
        S_corr, S_corr_, dx, dy, dz, atm_pair = data
        
        distance = np.sqrt(dx**2+dy**2+dz**2)
        
        self.assertAlmostEqual(S_corr[np.isclose(distance, 0)], 1.0)
        self.assertAlmostEqual(S_corr_[np.isclose(distance, 0)], 1.0)
        
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Co'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Fe_Fe'], 1.0)
        np.testing.assert_array_almost_equal(S_corr[atm_pair == 'Co_Fe'], -1.0)
        
    def test_symmetrize(self):
    
        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2
        
        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        
        nu, nv, nw = 3, 3, 3
        
        Rx, Ry, Rz = space.cell(nu, nv, nw, A)
        
        atm = np.array(['Fe','Fe'])
        u = np.array([0.0,0.5])
        v = np.array([0.0,0.5])
        w = np.array([0.0,0.5])
                        
        ux, uy, uz = crystal.transform(u, v, w, A)
        
        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)
                
        data = functions.pairs3d(rx, ry, rz, atms, nu, nv, nw, A, fract=1.0)
        dx, dy, dz, atm_pair, counts, search, coordinate, N = data
        
        arr = np.random.random(atm_pair.shape)
        
        symm_data = functions.symmetrize(arr, dx, dy, dz, atm_pair, A, 'm-3m')
        symm_arr, dx_symm, dy_symm, dz_symm, ion_symm = symm_data
        
        symops = symmetry.laue('m-3m')
                                            
        total = []
        for symop in symops:
            
            transformed = symmetry.evaluate([symop], 
                                            [dx_symm,dy_symm,dz_symm], 
                                            translate=False)
            total.append(transformed)
    
        total = np.vstack(total)
    
        for i in range(ion_symm.shape[0]):
            
            total[:,:,i] = total[np.lexsort(total[:,:,i].T),:,i]
            
        total = np.vstack(total)
            
        total, ind, inv = np.unique(total, axis=1, 
                                    return_index=True, 
                                    return_inverse=True)
        
        np.testing.assert_array_almost_equal(symm_arr[ind][inv], symm_arr)
                
if __name__ == '__main__':
    unittest.main()