#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.diffuse import space, simulation
from disorder.material import crystal

class test_simulation(unittest.TestCase):

    def test_energy(self):

        nu, nv, nw = 3, 4, 5

        n_atm = 2

        atm = np.array(['Fe3+','Fe3+'])

        u, v, w = np.array([0,0.5]), np.array([0,0.5]), np.array([0,0.5])

        a, b, c, alpha, beta, gamma = 5, 5, 5, np.pi/2, np.pi/2, np.pi/2

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)

        ux, uy, uz = crystal.transform(u, v, w, A)

        pair_dict = crystal.pairs(u, v, w, atm, A, extend=True)

        img_ind_i, img_ind_j, img_ind_k, atm_ind = [], [], [], []

        du, dv, dw = [], [], []

        for i in pair_dict.keys():
            for pair in pair_dict[i].keys():
                pair_no, atm_j = pair
                ind_i, ind_j, ind_k, atm_a = [], [], [], []
                for j, img in zip(*pair_dict[i][pair]):
                    ind_i.append(img[0])
                    ind_j.append(img[1])
                    ind_k.append(img[2])
                    atm_a.append(j)
                    du.append(u[j]-u[i]+img[0])
                    dv.append(v[j]-v[i]+img[1])
                    dw.append(w[j]-w[i]+img[2])
                img_ind_i.append(ind_i)
                img_ind_j.append(ind_j)
                img_ind_k.append(ind_k)
                atm_ind.append(atm_a)

        img_ind_i = np.array(img_ind_i, dtype=np.int32)
        img_ind_j = np.array(img_ind_j, dtype=np.int32)
        img_ind_k = np.array(img_ind_k, dtype=np.int32)
        atm_ind = np.array(atm_ind, dtype=np.int32)

        du, dv, dw = np.array(du), np.array(dv), np.array(dw)

        dx, dy, dz = crystal.transform(du, dv, dw, A)

        dx = dx.reshape(atm_ind.shape)
        dy = dy.reshape(atm_ind.shape)
        dz = dz.reshape(atm_ind.shape)

        d = np.sqrt(dx**2+dy**2+dz**2)

        n_pair = atm_ind.shape[1]

        dist = np.round(d/1e-2).astype(int)

        _, inv_ind = np.unique(dist, return_inverse=True)

        pair_ind = np.arange(n_pair+1,
                             dtype=np.int32)[inv_ind].reshape(n_atm,n_pair)

        mask = d < 0.99*np.sqrt(2*a**2)

        n_pair = mask.sum() // n_atm

        dx = dx[mask].reshape(n_atm,n_pair)
        dy = dy[mask].reshape(n_atm,n_pair)
        dz = dz[mask].reshape(n_atm,n_pair)

        pair_ind = pair_ind[mask].reshape(n_atm,n_pair)

        img_ind_i = img_ind_i[mask].reshape(n_atm,n_pair)
        img_ind_j = img_ind_j[mask].reshape(n_atm,n_pair)
        img_ind_k = img_ind_k[mask].reshape(n_atm,n_pair)
        atm_ind = atm_ind[mask].reshape(n_atm,n_pair)

        d_xyz = np.stack((dx,dy,dz))

        inv_d_xyz = -d_xyz

        pair_inv = np.zeros((n_atm,n_pair), dtype=np.int32)

        for i in range(n_atm):
            for p in range(n_pair):
                for q in range(n_pair):
                    if (np.allclose(d_xyz[:,i,p],inv_d_xyz[:,atm_ind[i,p],q])):
                        pair_inv[i,p] = q

        pair_ij = np.zeros((n_atm,n_pair), dtype=np.int32)

        J = np.zeros((n_pair,3,3))
        K = np.zeros((n_atm,3,3))
        g = np.zeros((n_atm,3,3))

        Jx = -1
        Jz = -1.

        J[:n_pair,:,:] = np.array([[Jx, 0, 0],\
                                   [0, Jx, 0],\
                                   [0, 0, Jz]])

        K[:n_atm,:,:] = np.array([[0, 0, 0],\
                                  [0, 0, 0],\
                                  [0, 0, 0.]])

        g[:n_atm,:,:] = np.array([[2, 0, 0],\
                                  [0, 2, 0],\
                                  [0, 0, 2.]])

        B = np.array([0,0,0.])

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, ion = space.real(ux, uy, uz, ix, iy, iz, atm)
        
        np.random.seed(13)
        
        M = 1
        
        theta = 2*np.pi*np.random.rand(nu,nv,nw,n_atm,M)
        phi = np.arccos(1-2*np.random.rand(nu,nv,nw,n_atm,M))
        
        Sx = np.sin(phi)*np.cos(theta)
        Sy = np.sin(phi)*np.sin(theta)
        Sz = np.cos(phi)
        
        E = simulation.energy(Sx, 
                              Sy, 
                              Sz, 
                              J, 
                              K, 
                              g,
                              B,
                              atm_ind,
                              img_ind_i,
                              img_ind_j,
                              img_ind_k,
                              pair_ind,
                              pair_ij)

        print(E)

if __name__ == '__main__':
    unittest.main()