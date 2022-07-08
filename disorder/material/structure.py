#!/usr/bin/env/python3

import os
import numpy as np

from disorder.diffuse import scattering, space
from disorder.diffuse import displacive, magnetic
from disorder.material import crystal, symmetry

def factor(u, v, w, atms, occupancy, U11, U22, U33, U23, U13, U12,
           a, b, c, alpha, beta, gamma, symops, dmin=0.3, source='neutron'):

    n_atm = atms.shape[0]

    inv_constants = crystal.reciprocal(a, b, c, alpha, beta, gamma)

    a_, b_, c_, alpha_, beta_, gamma_ = inv_constants

    B = crystal.cartesian(a_, b_, c_, alpha_, beta_, gamma_)

    hmax, kmax, lmax = np.floor(np.array([a,b,c])/dmin).astype(int)

    h, k, l = np.meshgrid(np.arange(-hmax, hmax+1),
                          np.arange(-kmax, kmax+1),
                          np.arange(-lmax, lmax+1), indexing='ij')

    h = np.delete(h, lmax+(2*lmax+1)*(kmax+(2*kmax+1)*hmax))
    k = np.delete(k, lmax+(2*lmax+1)*(kmax+(2*kmax+1)*hmax))
    l = np.delete(l, lmax+(2*lmax+1)*(kmax+(2*kmax+1)*hmax))

    h, k, l = h[::-1], k[::-1], l[::-1]

    Qh, Qk, Ql = crystal.vector(h, k, l, B)

    Q = np.sqrt(Qh**2+Qk**2+Ql**2)

    d = 2*np.pi/Q

    ind = d >= dmin

    h, k, l, d, Q = h[ind], k[ind], l[ind], d[ind], Q[ind]

    ind = ~symmetry.absence(symops, h, k, l)

    h, k, l, d, Q = h[ind], k[ind], l[ind], d[ind], Q[ind]

    n_hkl = Q.size

    phase_factor = np.exp(2j*np.pi*(h[:,np.newaxis]*u+
                                    k[:,np.newaxis]*v+
                                    l[:,np.newaxis]*w))

    if (source == 'neutron'):
        scattering_power = scattering.length(atms, n_hkl).reshape(n_hkl,n_atm)
    else:
        scattering_power = scattering.form(atms, Q).reshape(n_hkl,n_atm)

    T = np.exp(-2*np.pi**2*(U11*(h*a_)[:,np.newaxis]**2+
                            U22*(k*b_)[:,np.newaxis]**2+
                            U33*(l*c_)[:,np.newaxis]**2+
                            U23*(k*l*b_*c_*2)[:,np.newaxis]+
                            U13*(h*l*a_*c_*2)[:,np.newaxis]+
                            U12*(h*k*a_*b_*2)[:,np.newaxis]))

    factors = scattering_power*occupancy*T*phase_factor

    F = factors.sum(axis=1)

    coordinate = [h,k,l]

    symops = np.unique(symmetry.inverse(symops))

    total = []
    for symop in symops:

        transformed = symmetry.evaluate([symop], coordinate, translate=False)
        total.append(transformed)

    total = np.vstack(total)

    for i in range(n_hkl):

        total[:,:,i] = total[np.lexsort(total[:,:,i].T),:,i]

    total = np.vstack(total)

    total, ind, mult = np.unique(total, axis=1,
                                 return_index=True,
                                 return_counts=True)

    h, k, l, d, F = h[ind], k[ind], l[ind], d[ind], F[ind]

    ind = np.lexsort((h,k,l,d),axis=0)[::-1]

    h, k, l, d, F, mult = h[ind], k[ind], l[ind], d[ind], F[ind], mult[ind]

    return h, k, l, d, F, mult

class UnitCell:

    def __init__(self, filename, tol=1e-2):

        filename = os.path.abspath(filename)

        folder = os.path.dirname(filename)
        filename = os.path.basename(filename)

        self.__folder = folder
        self.__filename = filename

        self. __load_unit_cell(tol)

    def __load_unit_cell(self, tol):

        folder = self.get_filepath()
        filename = self.get_filename()

        constants = crystal.parameters(folder=folder, filename=filename)

        a, b, c, alpha, beta, gamma = constants

        self.__a, self.__b, self.__c = a, b, c
        self.__alpha, self.__beta, self.__gamma = alpha, beta, gamma

        uc_dict = crystal.unitcell(folder=folder, filename=filename, tol=tol)

        n_atm = uc_dict['n_atom']

        self.__atm = uc_dict['atom']
        self.__site = uc_dict['site']

        uni, ind = np.unique(self.__site, return_index=True)

        self.__act = np.full(uni.size, True)
        self.__ind = ind

        self.__mask = self.__ind[self.__act]
        self.__index = self.__act[self.__site]
        self.__inverse = np.arange(self.__act.size)[self.__site][self.__index]

        self.__op = uc_dict['operator']
        self.__mag_op = uc_dict['magnetic_operator']

        self.__u = uc_dict['u']
        self.__v = uc_dict['v']
        self.__w = uc_dict['w']

        self.__occ = uc_dict['occupancy']

        displacement = uc_dict['displacement']

        self.__U11 = np.zeros(n_atm)
        self.__U22 = np.zeros(n_atm)
        self.__U33 = np.zeros(n_atm)
        self.__U23 = np.zeros(n_atm)
        self.__U13 = np.zeros(n_atm)
        self.__U12 = np.zeros(n_atm)

        if (displacement.shape[1] == 1):
            self.set_isotropic_displacement_parameter(displacement.flatten())
        else:
            self.__U11 = displacement.T[0]
            self.__U22 = displacement.T[1]
            self.__U33 = displacement.T[2]
            self.__U23 = displacement.T[3]
            self.__U13 = displacement.T[4]
            self.__U12 = displacement.T[5]

        self.__mu1, self.__mu2, self.__mu3 = uc_dict['moment'].T

        self.__g = np.full(n_atm, 2.0)

        hm, sg = crystal.group(folder=folder, filename=filename)

        self.__hm = hm
        self.__sg = sg

        lat = crystal.lattice(*constants)

        self.__lat = lat

        laue =  crystal.laue(folder=folder, filename=filename)

        self.__laue = laue

        self.__pg = np.empty(n_atm, dtype=object)
        self.__mult = np.empty(n_atm, dtype=int)
        self.__sp_pos = np.empty(n_atm, dtype=object)

        A = self.get_fractional_cartesian_transform()

        for i, (u, v, w) in enumerate(zip(self.__u,self.__v,self.__w)):

            pg, mult, sp_pos = symmetry.site(self.__op, [u,v,w], A, tol=1e-2)

            self.__pg[i], self.__mult[i], self.__sp_pos[i] = pg, mult, sp_pos

    def __get_all_lattice_constants(self):

        constants = self.__a, self.__b, self.__c, \
                    self.__alpha, self.__beta, self.__gamma

        return constants

    def get_filepath(self):

        return self.__folder

    def get_filename(self):

        return self.__filename

    def get_sites(self):

        return self.__site

    def get_active_sites(self):

        return self.__act

    def set_active_sites(self, act):

        self.__act = act

        self.__mask = self.__ind[self.__act]
        self.__index = self.__act[self.__site]
        self.__inverse = np.arange(self.__act.size)[self.__site][self.__index]

    def get_number_atoms_per_unit_cell(self):

        return self.__act[self.__site].sum()

    def get_fractional_coordinates(self):

        mask = self.__mask

        return self.__u[mask], self.__v[mask], self.__w[mask]

    def set_fractional_coordinates(self, u, v, w):

        mask = self.__mask

        ind = self.__index
        inv = self.__inverse

        operators = symmetry.binary(self.__op[ind], self.__op[mask][inv])

        up, vp, wp = u[inv], v[inv], w[inv]

        for i, operator in enumerate(operators):
            uvw = symmetry.evaluate([operator], [up[i], vp[i], wp[i]])
            up[i], vp[i], wp[i] = np.mod(uvw, 1).flatten()

        self.__u[ind], self.__v[ind], self.__w[ind] = up, vp, wp

    def get_unit_cell_cartesian_atomic_coordinates(self):

        A = self.get_fractional_cartesian_transform()
        u, v, w = self.get_fractional_coordinates()

        return crystal.transform(u, v, w, A)

    def get_unit_cell_atoms(self):

        mask = self.__mask

        return self.__atm[mask]

    def set_unit_cell_atoms(self, atm):

        ind = self.__index
        inv = self.__inverse

        self.__atm[ind] = atm[inv]

    def get_occupancies(self):

        mask = self.__mask

        return self.__occ[mask]

    def set_occupancies(self, occ):

        ind = self.__index
        inv = self.__inverse

        self.__occ[ind] = occ[inv]

    def get_anisotropic_displacement_parameters(self):

        mask = self.__mask

        U11 = self.__U11[mask]
        U22 = self.__U22[mask]
        U33 = self.__U33[mask]
        U23 = self.__U23[mask]
        U13 = self.__U13[mask]
        U12 = self.__U12[mask]

        return U11, U22, U33, U23, U13, U12

    def set_anisotropic_displacement_parameters(self, U11, U22, U33,
                                                      U23, U13, U12):

        mask = self.__mask

        ind = self.__index
        inv = self.__inverse

        operators = symmetry.binary(self.__op[ind], self.__op[mask][inv])

        U11p = U11[inv]
        U22p = U22[inv]
        U33p = U33[inv]
        U23p = U23[inv]
        U13p = U13[inv]
        U12p = U12[inv]

        for i, operator in enumerate(operators):
            disp = [U11p[i], U22p[i], U33p[i], U23p[i], U13p[i], U12p[i]]
            disp = symmetry.evaluate_disp([operator], disp)
            U11p[i], U22p[i], U33p[i], U23p[i], U13p[i], U12p[i] = disp

        self.__U11[ind] = U11p
        self.__U22[ind] = U22p
        self.__U33[ind] = U33p
        self.__U23[ind] = U23p
        self.__U13[ind] = U13p
        self.__U12[ind] = U12p

    def set_isotropic_displacement_parameter(self, Uiso):

        ind = self.__index
        inv = self.__inverse

        D = self.get_atomic_displacement_cartesian_transform()

        uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))

        U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
        U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]

        self.__U11[ind] = U11[inv]
        self.__U22[ind] = U22[inv]
        self.__U33[ind] = U33[inv]
        self.__U23[ind] = U23[inv]
        self.__U13[ind] = U13[inv]
        self.__U12[ind] = U12[inv]

    def get_isotropic_displacement_parameter(self):

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

    def get_principal_displacement_parameters(self):

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.principal(U11, U22, U33, U23, U13, U12, D)

    def get_cartesian_anistropic_displacement_parameters(self):

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.cartesian(U11, U22, U33, U23, U13, U12, D)

    def get_crystal_axis_magnetic_moments(self):

        mask = self.__mask

        mu1 = self.__mu1[mask]
        mu2 = self.__mu2[mask]
        mu3 = self.__mu3[mask]

        return mu1, mu2, mu3

    def set_crystal_axis_magnetic_moments(self, mu1, mu2, mu3):

        mask = self.__mask

        ind = self.__index
        inv = self.__inverse

        operators = symmetry.binary_mag(self.__mag_op[ind],
                                        self.__mag_op[mask][inv])

        mu1p = mu1[inv]
        mu2p = mu2[inv]
        mu3p = mu3[inv]

        for i, operator in enumerate(operators):
            mag = [mu1p[i], mu2p[i], mu3p[i]]
            mag = symmetry.evaluate_mag([operator], mag)
            mu1p[i], mu2p[i], mu3p[i] = np.array(mag).flatten()

        self.__mu1[ind] = mu1p
        self.__mu2[ind] = mu2p
        self.__mu3[ind] = mu3p

    def get_magnetic_moment_magnitude(self):

        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()

        return magnetic.magnitude(mu1, mu2, mu3, C)

    def get_cartesian_magnetic_moments(self):

        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()

        return magnetic.cartesian(mu1, mu2, mu3, C)

    def get_g_factors(self):

        mask = self.__mask

        return self.__g[mask]

    def set_g_factors(self, g):

        ind = self.__index
        inv = self.__inverse

        self.__g[ind] = g[inv]

    def get_lattice_constants(self):

        lat = self.get_lattice_system()

        a = self.__a
        b = self.__b
        c = self.__c
        alpha = self.__alpha
        beta = self.__beta
        gamma = self.__gamma

        if (lat == 'Cubic'):
            constants = a
        elif (lat == 'Hexagonal' or lat == 'Tetragonal'):
            constants = a, c
        elif (lat == 'Rhobmohedral'):
            constants = a, alpha
        elif (lat == 'Orthorhombic'):
            constants = a, b, c
        elif (lat == 'Monoclinic'):
            if (not np.isclose(beta, np.pi/2)):
                constants = a, b, c, alpha, gamma
            else:
                constants = a, b, c, alpha, beta
        else:
            constants = a, b, c, alpha, beta, gamma

        return constants

    def set_lattice_constants(self, *constants):

        lat = self.get_lattice_system()

        a = self.__a
        b = self.__b
        c = self.__c
        alpha = self.__alpha
        beta = self.__beta
        gamma = self.__gamma

        if (lat == 'Cubic'):
            a = constants
            b = c = a
        elif (lat == 'Hexagonal' or lat == 'Tetragonal'):
            a, c = constants
            b = a
        elif (lat == 'Rhobmohedral'):
            a, alpha = constants
            b = c = a
            beta = gamma = alpha
        elif (lat == 'Orthorhombic'):
            a, b, c = constants
            alpha = beta = gamma = np.pi/2
        elif (lat == 'Monoclinic'):
            if (not np.isclose(beta, np.pi/2)):
                a, b, c, alpha, gamma = constants
            else:
                a, b, c, alpha, beta = constants
        else:
            a, b, c, alpha, beta, gamma = constants

        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    def get_reciprocal_lattice_constants(self):

        constants = self.__get_all_lattice_constants()

        return crystal.reciprocal(*constants)

    def get_symmetry_operators(self):

        mask = self.__mask

        return self.__op[mask]

    def get_magnetic_symmetry_operators(self):

        mask = self.__mask

        return self.__mag_op[mask]

    def get_lattice_system(self):

        return self.__lat

    def get_lattice_volume(self):

        constants = self.__get_all_lattice_constants()

        return crystal.volume(*constants)

    def get_reciprocal_lattice_volume(self):

        constants = self.__get_all_lattice_constants()

        return crystal.volume(*constants)

    def get_metric_tensor(self):

        constants = self.__get_all_lattice_constants()

        return crystal.metric(*constants)

    def get_reciprocal_metric_tensor(self):

        constants = self.__get_all_lattice_constants()

        return crystal.metric(*constants)

    def get_fractional_cartesian_transform(self):

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian(*constants)

    def get_miller_cartesian_transform(self):

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian(*constants)

    def get_cartesian_rotation(self):

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_rotation(*constants)

    def get_moment_cartesian_transform(self):

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_moment(*constants)

    def get_atomic_displacement_cartesian_transform(self):

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_displacement(*constants)

    def get_space_group_symbol(self):

        return self.__hm

    def get_space_group_number(self):

        return self.__sg

    def get_laue(self):

        return self.__laue

    def get_site_symmetries(self):

        mask = self.__mask

        return self.__pg[mask]

    def get_wyckoff_special_positions(self):

        mask = self.__mask

        return self.__sp_pos[mask]

    def get_site_multiplicities(self):

        mask = self.__mask

        return self.__mult[mask]

class SuperCell(UnitCell):

    def __init__(self, filename, nu=1, nv=1, nw=1, tol=1e-2):

        super(SuperCell, self).__init__(filename, tol)

        self.set_super_cell_dimensions(nu, nv, nw)

    def get_super_cell_dimensions(self):

        return self.__nu, self.__nv, self.__nw

    def set_super_cell_dimensions(self, nu, nv, nw):

        self.__nu = nu
        self.__nv = nv
        self.__nw = nw

    def get_super_cell_size(self):

        nu, nv, nw = self.get_super_cell_dimensions()

        return nu*nv*nw

    def get_number_atoms_per_super_cell(self):

        n_uvw = self.get_super_cell_size()
        n_atm = self.get_number_atoms_per_unit_cell()

        return n_uvw*n_atm

    def get_cartesian_lattice_points(self):

        A = self.get_fractional_cartesian_transform()
        nu, nv, nw = self.get_super_cell_dimensions()

        return space.cell(nu, nv, nw, A)

    def get_super_cell_cartesian_atomic_coordinates(self):

        ux, uy, uz = self.get_unit_cell_cartesian_atomic_coordinates()
        ix, iy, iz = self.get_cartesian_lattice_points()
        atm = self.get_unit_cell_atoms()

        return space.real(ux, uy, uz, ix, iy, iz, atm)