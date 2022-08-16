#!/usr/bin/env python3

import unittest
import numpy as np

from disorder.material import crystal
from disorder.diffuse import space, scattering, powder
from disorder.diffuse import magnetic, occupational, displacive

class test_powder(unittest.TestCase):

    def test_magnetic(self):

        np.random.seed(13)

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        nQ = 41

        Q = 2*np.pi*np.linspace(0.1,1,nQ)

        nu, nv, nw, n_atm = 8, 8, 8, 2

        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])

        atm = np.array(['Fe3+','Mn3+'])
        occupancy = np.array([0.25,0.5])
        g = np.array([2.,2.])

        U11 = np.array([0.05,0.03])
        U22 = np.array([0.06,0.04])
        U33 = np.array([0.04,0.06])
        U23 = np.array([0.005,-0.003])
        U13 = np.array([-0.004,0.002])
        U12 = np.array([0.003,-0.002])

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        Uiso = displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

        Sx, Sy, Sz = magnetic.spin(nu, nv, nw, n_atm)

        ux, uy, uz = crystal.transform(u, v, w, A)

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)

        I = powder.magnetic(Sx, Sy, Sz, occupancy,
                            U11, U22, U33, U23, U13, U12,
                            rx, ry, rz, atms, Q, A, D, nu, nv, nw, g)

        N = nu*nv*nw*n_atm

        i, j = np.triu_indices(N,1)
        k, l = i % n_atm, j % n_atm

        m = np.arange(N)
        n = m % n_atm

        rx_ij, ry_ij, rz_ij = rx[j]-rx[i], ry[j]-ry[i], rz[j]-rz[i]

        r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)

        S_i_dot_S_j = Sx[i]*Sx[j]+Sy[i]*Sy[j]+Sz[i]*Sz[j]

        S_i_dot_r_hat_ij = (Sx[i]*rx_ij+Sy[i]*ry_ij+Sz[i]*rz_ij)/r_ij
        S_j_dot_r_hat_ij = (Sx[j]*rx_ij+Sy[j]*ry_ij+Sz[j]*rz_ij)/r_ij

        Aij = S_i_dot_S_j-S_i_dot_r_hat_ij*S_j_dot_r_hat_ij
        Bij = 3*S_i_dot_r_hat_ij*S_j_dot_r_hat_ij-S_i_dot_S_j

        f = magnetic.form(Q, atm, g).reshape(nQ,n_atm)

        Tk = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[k])
        Tl = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[l])

        fi, fj = occupancy[k]*f[:,k]*Tk, occupancy[l]*f[:,l]*Tl

        Qr_ij = Q[:,np.newaxis]*r_ij

        Tn = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[n])

        fm = occupancy[n]*f[:,n]*Tn

        I_ref = 2*np.sum(fm**2, axis=1)/3/N\
              + 2*np.sum(fi*fj*(Aij*np.sin(Qr_ij)/Qr_ij
                                +Bij*(np.sin(Qr_ij)/Qr_ij**3
                                    -np.cos(Qr_ij)/Qr_ij**2)), axis=1)/N

        self.assertLess(np.sqrt(np.mean((I/I_ref-1)**2)), 0.05)

    def test_occupational(self):

        np.random.seed(13)

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        nQ = 41

        Q = 2*np.pi*np.linspace(0.1,1,nQ)

        nu, nv, nw, n_atm = 8, 8, 8, 2

        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])

        atm = np.array(['Fe','Mn'])
        occupancy = np.array([0.25,0.5])

        U11 = np.array([0.05,0.03])
        U22 = np.array([0.06,0.04])
        U33 = np.array([0.04,0.06])
        U23 = np.array([0.005,-0.003])
        U13 = np.array([-0.004,0.002])
        U12 = np.array([0.003,-0.002])

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        Uiso = displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

        A_r = occupational.composition(nu, nv, nw, n_atm, value=occupancy)

        ux, uy, uz = crystal.transform(u, v, w, A)

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)

        I = powder.occupational(A_r, occupancy, U11, U22, U33, U23, U13, U12,
                                rx, ry, rz, atms, Q, A, D, nu, nv, nw)

        I_ave = powder.structural(occupancy, U11, U22, U33, U23, U13, U12,
                                  rx, ry, rz, atms, Q, A, D, nu, nv, nw)

        I_diff = I-I_ave

        N = nu*nv*nw*n_atm

        i, j = np.triu_indices(N,1)
        k, l = i % n_atm, j % n_atm

        m = np.arange(N)
        n = m % n_atm

        rx_ij, ry_ij, rz_ij = rx[j]-rx[i], ry[j]-ry[i], rz[j]-rz[i]

        r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)

        delta_ij = (1+A_r[i])*(1+A_r[j])

        b = scattering.length(atm, Q.size).reshape(nQ,n_atm)

        Tk = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[k])
        Tl = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[l])

        fi, fj = occupancy[k]*b[:,k]*Tk, occupancy[l]*b[:,l]*Tl

        fij = (fi*np.conj(fj)).real

        Qr_ij = Q[:,np.newaxis]*r_ij

        delta_mm = (1+A_r[m])**2

        Tn = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[n])

        fm = occupancy[n]*b[:,n]*Tn

        fmm = (fm*np.conj(fm)).real

        I_ref =   np.sum(fmm*(delta_mm-1), axis=1)/N\
              + 2*np.sum(fij*(delta_ij-1)*np.sin(Qr_ij)/Qr_ij, axis=1)/N

        self.assertLess(np.sqrt(np.mean((I_diff/I_ref-1)**2)), 0.05)

    def test_displacive(self):

        np.random.seed(13)

        a, b, c, alpha, beta, gamma = 5, 6, 7, np.pi/2, np.pi/3, np.pi/4

        inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

        a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

        nQ = 41

        Q = 2*np.pi*np.linspace(0.1,1,nQ)

        nu, nv, nw, n_atm = 8, 8, 8, 2

        u = np.array([0.2,0.1])
        v = np.array([0.3,0.4])
        w = np.array([0.4,0.5])

        atm = np.array(['Fe','Mn'])
        occupancy = np.array([0.25,0.5])

        U11 = np.array([0.05,0.03])
        U22 = np.array([0.06,0.04])
        U33 = np.array([0.04,0.06])
        U23 = np.array([0.005,-0.003])
        U13 = np.array([-0.004,0.002])
        U12 = np.array([0.003,-0.002])

        A = crystal.cartesian(a, b, c, alpha, beta, gamma)
        D = crystal.cartesian_displacement(a, b, c, alpha, beta, gamma)

        Uxx, Uyy, Uzz, \
        Uyz, Uxz, Uxy = displacive.cartesian(U11, U22, U33,
                                             U23, U13, U12, D)

        U = np.row_stack((Uxx, Uyy, Uzz, Uyz, Uxz, Uxy))
        Ux, Uy, Uz = displacive.expansion(nu, nv, nw, n_atm, value=U)

        Uiso = displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

        ux, uy, uz = crystal.transform(u, v, w, A)

        ix, iy, iz = space.cell(nu, nv, nw, A)

        rx, ry, rz, atms = space.real(ux, uy, uz, ix, iy, iz, atm)

        I = powder.displacive(Ux, Uy, Uz, occupancy,
                              rx, ry, rz, atms, Q, A, D, nu, nv, nw, 3)

        I_ave = powder.structural(occupancy, U11, U22, U33, U23, U13, U12,
                                  rx, ry, rz, atms, Q, A, D, nu, nv, nw)

        I_diff = I-I_ave

        N = nu*nv*nw*n_atm

        i, j = np.triu_indices(N,1)
        k, l = i % n_atm, j % n_atm

        m = np.arange(N)
        n = m % n_atm

        rx_ij, ry_ij, rz_ij = rx[j]-rx[i], ry[j]-ry[i], rz[j]-rz[i]

        r_ij = np.sqrt(rx_ij**2+ry_ij**2+rz_ij**2)

        Ux_ij, Uy_ij, Uz_ij = Ux[j]-Ux[i], Uy[j]-Uy[i], Uz[j]-Uz[i]

        U_ij = np.sqrt(Ux_ij**2+Uy_ij**2+Uz_ij**2)

        U_ij_mul_r_ij = U_ij*r_ij

        cos_theta = (Ux_ij*rx_ij+Uy_ij*ry_ij+Uz_ij*rz_ij)/U_ij_mul_r_ij

        Qu_ij = Q[:,np.newaxis]*U_ij

        b = scattering.length(atm, Q.size).reshape(nQ,n_atm)

        Tk = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[k])
        Tl = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[l])

        bi, bj = occupancy[k]*b[:,k], occupancy[l]*b[:,l]

        bij = (bi*np.conj(bj)).real

        Qr_ij = Q[:,np.newaxis]*r_ij

        Aij =  1
        Bij = -Qu_ij*(cos_theta+Qu_ij/Qr_ij/2)
        Cij =  Qu_ij**2/2*cos_theta*(cos_theta+0.5*Qu_ij/Qr_ij)
        Dij = -Qu_ij**2/3*cos_theta**3

        Tn = np.exp(-0.5*Q[:,np.newaxis]**2*Uiso[n])

        bm = occupancy[n]*b[:,n]

        bmm = (bm*np.conj(bm)).real

        I_ref =   np.sum(bmm*(1-Tn**2), axis=1)/N\
              + 2*np.sum(bij*((Aij-Tk*Tl)*np.sin(Qr_ij)/Qr_ij
                              +Bij*(np.sin(Qr_ij)/Qr_ij-np.cos(Qr_ij))/Qr_ij
                              +Cij*((3-Qr_ij**2)*np.sin(Qr_ij)/Qr_ij
                                              -3*np.cos(Qr_ij))/Qr_ij**2
                              +Dij*((15-6*Qr_ij**2)*np.sin(Qr_ij)/Qr_ij
                                    -(15-Qr_ij**2)*np.cos(Qr_ij))/Qr_ij**3),
                         axis=1)/N

        self.assertLess(np.sqrt(np.mean((I_diff/I_ref-1)**2)), 0.1)

if __name__ == '__main__':
    unittest.main()
