#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import interaction, space

class test_interaction(unittest.TestCase):

    def test_charge_charge_matrix(self):

        a = b = c = 4.1
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 4, 4, 4, 2

        n = nu*nv*nw*n_atm

        atm = np.array(['Cs','Cl'])

        u = np.array([0.0,0.5])
        v = np.array([0.0,0.5])
        w = np.array([0.0,0.5])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        z = np.zeros(n)

        i, j = np.triu_indices(n)

        Qij = np.zeros((n,n))

        Qij[i,j] = interaction.charge_charge_matrix(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, B, R)

        Qij[j,i] = Qij[i,j]

        z[atms == 'Cs'] = +1
        z[atms == 'Cl'] = -1

        np.testing.assert_array_almost_equal(Qij, Qij.T)

        phi = np.dot(Qij,z)*a*np.sqrt(3)/2
        np.testing.assert_array_almost_equal(phi[atms == 'Cs'], -1.762675, 2)
        np.testing.assert_array_almost_equal(phi[atms == 'Cl'], +1.762675, 2)

        a = b = c = 5.7
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 4, 4, 4, 8

        n = nu*nv*nw*n_atm

        atm = np.array(['Na','Na','Na','Na','Cl','Cl','Cl','Cl'])

        u = np.array([0.0,0.0,0.5,0.5,0.5,0.5,0.0,0.0])
        v = np.array([0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5])
        w = np.array([0.0,0.5,0.5,0.0,0.0,0.5,0.5,0.0])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        z = np.zeros(n)

        i, j = np.triu_indices(n)

        Qij = np.zeros((n,n))

        Qij[i,j] = interaction.charge_charge_matrix(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, B, R)

        Qij[j,i] = Qij[i,j]

        z[atms == 'Na'] = +1
        z[atms == 'Cl'] = -1

        phi = np.dot(Qij,z)*a/2
        np.testing.assert_array_almost_equal(phi[atms == 'Na'], -1.747565, 2)
        np.testing.assert_array_almost_equal(phi[atms == 'Cl'], +1.747565, 2)

        a = b = 3.819
        c = 6.246
        alpha = beta = np.pi/2
        gamma = 2*np.pi/3

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 6, 6, 6, 4

        n = nu*nv*nw*n_atm

        atm = np.array(['Zn','Zn','S','S'])

        u = np.array([2/3,1/3,2/3,1/3])
        v = np.array([1/3,2/3,1/3,2/3])
        w = np.array([0.0,0.5,0.625,0.125])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        z = np.zeros(n)

        i, j = np.triu_indices(n)

        Qij = np.zeros((n,n))

        Qij[i,j] = interaction.charge_charge_matrix(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, B, R)

        Qij[j,i] = Qij[i,j]

        z[atms == 'Zn'] = +2
        z[atms == 'S'] = -2

        phi = np.dot(Qij,z)*c*0.375
        np.testing.assert_array_almost_equal(phi[atms == 'Zn'], -3.28146, 2)
        np.testing.assert_array_almost_equal(phi[atms == 'S'], +3.28146, 2)

        a = b = c = 5.52
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 5, 5, 5, 12

        n = nu*nv*nw*n_atm

        atm = np.array(['Ca','Ca','Ca','Ca','F','F','F','F','F','F','F','F'])

        u = np.array([0,0,0.5,0.5,0.25,0.25,0.25,0.25,0.75,0.75,0.75,0.75])
        v = np.array([0,0.5,0,0.5,0.75,0.25,0.25,0.75,0.75,0.25,0.25,0.75])
        w = np.array([0,0.5,0.5,0,0.75,0.75,0.25,0.25,0.25,0.25,0.75,0.75])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        z = np.zeros(n)

        i, j = np.triu_indices(n)

        Qij = np.zeros((n,n))

        Qij[i,j] = interaction.charge_charge_matrix(rx, ry, rz,
                                                    nu, nv, nw,
                                                    n_atm, A, B, R)

        Qij[j,i] = Qij[i,j]

        z[atms == 'Ca'] = +2
        z[atms == 'F'] = -1

        phi = np.dot(Qij,z)*a*0.25*np.sqrt(3)
        np.testing.assert_array_almost_equal(phi[atms == 'Ca'], -3.276110, 2)
        np.testing.assert_array_almost_equal(phi[atms == 'F'], +1.762675, 2)

    def test_charge_dipole_matrix(self):

        a = b = c = 5.4187
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 5, 5, 5, 12

        n = nu*nv*nw*n_atm

        atm = np.array(['S','S','S','S','S','S','S','S','Fe','Fe','Fe','Fe'])

        x = 0.385

        u = np.array([x,1-x,0.5-x,0.5+x,x,1-x,0.5+x,0.5-x,0.0,0.0,0.5,0.5])
        v = np.array([x,1-x,0.5+x,0.5-x,0.5-x,0.5+x,x,1-x,0.0,0.5,0.0,0.5])
        w = np.array([x,1-x,x,1-x,0.5+x,0.5-x,0.5-x,0.5+x,0.0,0.5,0.5,0.0])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        z = np.zeros(n)

        px = np.zeros(n)
        py = np.zeros(n)
        pz = np.zeros(n)

        i, j = np.triu_indices(n)

        Qijk = np.zeros((n,n,3))

        Qijk[i,j,:] = interaction.charge_dipole_matrix(rx, ry, rz,
                                                       nu, nv, nw,
                                                       n_atm, A, B, R)

        Qijk[j,i,:] = Qijk[i,j,:]

        z[atms == 'S'] = -1
        z[atms == 'Fe'] = +2

        px.reshape(nu,nv,nw,n_atm)[:,:,:,[0,3,4,6]] = +np.sqrt(3)/3
        px.reshape(nu,nv,nw,n_atm)[:,:,:,[1,2,5,7]] = -np.sqrt(3)/3

        py.reshape(nu,nv,nw,n_atm)[:,:,:,[0,2,5,6]] = +np.sqrt(3)/3
        py.reshape(nu,nv,nw,n_atm)[:,:,:,[1,3,4,7]] = -np.sqrt(3)/3

        pz.reshape(nu,nv,nw,n_atm)[:,:,:,[0,2,4,7]] = +np.sqrt(3)/3
        pz.reshape(nu,nv,nw,n_atm)[:,:,:,[1,3,5,6]] = -np.sqrt(3)/3

        p = np.column_stack((px,py,pz))

        am = np.array([-1.957,-7.458])
        ad = np.array([-1.184,-2.898])
        bm = 2.632
        bd = -2.561

        np.testing.assert_array_almost_equal(Qijk, np.swapaxes(Qijk,0,1))

        E = -np.einsum('i,i->...',z,np.einsum('ijk,jk->i',Qijk,p))/n
        self.assertAlmostEqual(E, -2*(bm/a**2+bd/a**3)/a, places=1)

        Qij = np.zeros((n,n))

        Qij[i,j] = interaction.charge_charge_matrix(rx, ry, rz, nu, nv, nw,
                                                    n_atm, A, B, R)

        Qij[j,i] = Qij[i,j]

        E = np.dot(z,np.dot(Qij,z))/n
        self.assertAlmostEqual(E, (2*am[0]+am[1])/a*2/3, places=2)

        Qijm = np.zeros((n,n,6))
        Qijkl = np.zeros((n,n,3,3))

        Qijm[i,j,:] = interaction.dipole_dipole_matrix(rx, ry, rz,
                                                       nu, nv, nw,
                                                       n_atm, A, B, R)

        Qijm[j,i,:] = Qijm[i,j,:]

        Qijkl[:,:,0,0] = Qijm[:,:,0]
        Qijkl[:,:,1,1] = Qijm[:,:,1]
        Qijkl[:,:,2,2] = Qijm[:,:,2]
        Qijkl[:,:,1,2] = Qijkl[:,:,2,1] = Qijm[:,:,3]
        Qijkl[:,:,0,2] = Qijkl[:,:,2,0] = Qijm[:,:,4]
        Qijkl[:,:,0,1] = Qijkl[:,:,1,0] = Qijm[:,:,5]

        E = np.einsum('ik,ik->...',p,np.einsum('ijkl,jl->ik',Qijkl,p))/n
        self.assertAlmostEqual(E, -(2*ad[0]+ad[1])/a**3*2/3, places=2)

    def test_dipole_dipole_matrix(self):

        a = b = c = 4.04
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 8, 8, 16, 1

        n = nu*nv*nw*n_atm

        atm = np.array(['Ti'])

        u = np.array([0.5])
        v = np.array([0.5])
        w = np.array([0.5])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        px = np.zeros(n)
        py = np.zeros(n)
        pz = np.zeros(n)

        i, j = np.triu_indices(n)

        Qijm = np.zeros((n,n,6))
        Qijkl = np.zeros((n,n,3,3))

        Qijm[i,j,:] = interaction.dipole_dipole_matrix(rx, ry, rz,
                                                       nu, nv, nw,
                                                       n_atm, A, B, R)

        Qijm[j,i,:] = Qijm[i,j,:]

        Qijkl[:,:,0,0] = Qijm[:,:,0]
        Qijkl[:,:,1,1] = Qijm[:,:,1]
        Qijkl[:,:,2,2] = Qijm[:,:,2]
        Qijkl[:,:,1,2] = Qijkl[:,:,2,1] = Qijm[:,:,3]
        Qijkl[:,:,0,2] = Qijkl[:,:,2,0] = Qijm[:,:,4]
        Qijkl[:,:,0,1] = Qijkl[:,:,1,0] = Qijm[:,:,5]

        np.testing.assert_array_almost_equal(Qijkl, np.swapaxes(Qijkl,0,1))
        np.testing.assert_array_almost_equal(Qijkl, np.swapaxes(Qijkl,2,3))

        px.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 1

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)*a**3/2
        np.testing.assert_array_almost_equal(E[...,2], -2.09440, 2)

        px.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[:,:,0::2,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[:,:,1::2,:] = -1

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)[...,2]*a**3/2
        np.testing.assert_array_almost_equal(E[...,0::2], +4.84372, 2)
        np.testing.assert_array_almost_equal(E[...,1::2], -4.84372, 2)

        px.reshape(nu,nv,nw,n_atm)[:,:,0::2,:] = +1
        px.reshape(nu,nv,nw,n_atm)[:,:,1::2,:] = -1
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)[...,0]*a**3/2
        np.testing.assert_array_almost_equal(E[0::2], -2.42186, 2)
        np.testing.assert_array_almost_equal(E[1::2], +2.42186, 2)

        px.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[0::2,0::2,:,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[1::2,1::2,:,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[0::2,1::2,:,:] = -1
        pz.reshape(nu,nv,nw,n_atm)[1::2,0::2,:,:] = -1

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)[...,2]*a**3/2
        E = E.reshape(nu,nv,nw,n_atm)
        np.testing.assert_array_almost_equal(E[0::2,0::2,:,:], -2.67679, 2)
        np.testing.assert_array_almost_equal(E[1::2,1::2,:,:], -2.67679, 2)
        np.testing.assert_array_almost_equal(E[0::2,1::2,:,:], +2.67679, 2)
        np.testing.assert_array_almost_equal(E[1::2,0::2,:,:], +2.67679, 2)

        px.reshape(nu,nv,nw,n_atm)[0::2,0::2,:,:] = +1
        px.reshape(nu,nv,nw,n_atm)[1::2,1::2,:,:] = +1
        px.reshape(nu,nv,nw,n_atm)[0::2,1::2,:,:] = -1
        px.reshape(nu,nv,nw,n_atm)[1::2,0::2,:,:] = -1
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)[...,0]*a**3/2
        E = E.reshape(nu,nv,nw,n_atm)
        np.testing.assert_array_almost_equal(E[0::2,0::2,:,:], +1.33839, 2)
        np.testing.assert_array_almost_equal(E[1::2,1::2,:,:], +1.33839, 2)
        np.testing.assert_array_almost_equal(E[0::2,1::2,:,:], -1.33839, 2)
        np.testing.assert_array_almost_equal(E[1::2,0::2,:,:], -1.33839, 2)

        px.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[0::2,0::2,0::2,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[1::2,1::2,0::2,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[0::2,1::2,1::2,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[1::2,0::2,1::2,:] = +1
        pz.reshape(nu,nv,nw,n_atm)[1::2,1::2,1::2,:] = -1
        pz.reshape(nu,nv,nw,n_atm)[1::2,0::2,0::2,:] = -1
        pz.reshape(nu,nv,nw,n_atm)[0::2,1::2,0::2,:] = -1
        pz.reshape(nu,nv,nw,n_atm)[0::2,0::2,1::2,:] = -1

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)*a**3/2
        np.testing.assert_array_almost_equal(E, 0, decimal=2)

        a = c = 2*4.04
        b = 4.04
        alpha = beta = gamma = np.pi/2

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        R = crystal.cartesian_rotation(a, b, c, alpha, beta, gamma)
        B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

        nu, nv, nw, n_atm = 8, 8, 8, 2

        n = nu*nv*nw*n_atm

        atm = np.array(['Ti','Ti'])

        u = np.array([0.0,0.5])
        v = np.array([0.0,0.0])
        w = np.array([0.0,0.5])

        Rx, Ry, Rz = space.cell(nu, nv, nw, A)

        ux, uy, uz = crystal.transform(u, v, w, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, Rx, Ry, Rz, atm)

        px = np.zeros(n)
        py = np.zeros(n)
        pz = np.zeros(n)

        i, j = np.triu_indices(n)

        Qijm = np.zeros((n,n,6))
        Qijkl = np.zeros((n,n,3,3))

        Qijm[i,j,:] = interaction.dipole_dipole_matrix(rx, ry, rz,
                                                       nu, nv, nw,
                                                       n_atm, A, B, R)

        Qijm[j,i,:] = Qijm[i,j,:]

        Qijkl[:,:,0,0] = Qijm[:,:,0]
        Qijkl[:,:,1,1] = Qijm[:,:,1]
        Qijkl[:,:,2,2] = Qijm[:,:,2]
        Qijkl[:,:,1,2] = Qijkl[:,:,2,1] = Qijm[:,:,3]
        Qijkl[:,:,0,2] = Qijkl[:,:,2,0] = Qijm[:,:,4]
        Qijkl[:,:,0,1] = Qijkl[:,:,1,0] = Qijm[:,:,5]

        px.reshape(nu,nv,nw,n_atm)[0::2,:,0::2,:] = -np.sqrt(2)/2
        px.reshape(nu,nv,nw,n_atm)[1::2,:,1::2,:] = -np.sqrt(2)/2
        px.reshape(nu,nv,nw,n_atm)[0::2,:,1::2,:] = +np.sqrt(2)/2
        px.reshape(nu,nv,nw,n_atm)[1::2,:,0::2,:] = +np.sqrt(2)/2
        py.reshape(nu,nv,nw,n_atm)[:,:,:,:] = 0
        pz.reshape(nu,nv,nw,n_atm)[0::2,:,0::2,:] = +np.sqrt(2)/2
        pz.reshape(nu,nv,nw,n_atm)[1::2,:,1::2,:] = +np.sqrt(2)/2
        pz.reshape(nu,nv,nw,n_atm)[0::2,:,1::2,:] = -np.sqrt(2)/2
        pz.reshape(nu,nv,nw,n_atm)[1::2,:,0::2,:] = -np.sqrt(2)/2

        p = np.column_stack((px,py,pz))
        E = np.einsum('ijkl,jl->ik',Qijkl,p)*a**3/16
        E = np.sqrt(np.sum(E**2, axis=1))
        np.testing.assert_array_almost_equal(E, 2.93226, 1)

if __name__ == '__main__':
    unittest.main()