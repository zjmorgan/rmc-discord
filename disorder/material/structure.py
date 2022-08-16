#!/usr/bin/env/python3

import os
import numpy as np

from disorder.diffuse import scattering, space
from disorder.diffuse import displacive, magnetic
from disorder.material import crystal, symmetry

def factor(u, v, w, atms, occupancy, U11, U22, U33, U23, U13, U12,
           a, b, c, alpha, beta, gamma, symops, dmin=0.3, source='neutron'):
    """
    Structure factor :math:`F(h,k,l)`.

    Parameters
    ----------
    u, v, w : 1d array
        Fractional coordinates for each atom site.
    atms : 1d array, str
        Ion or isotope for each atom site.
    occupancy : 1d array
        Site occupancies for each atom site.
    U11, U22, U33, U23, U13, U12 : 1d array
        Atomic displacement parameters in crystal axis system.
    a, b, c, alpha, beta, gamma : float
        Lattice constants :math:`a`, :math:`b`, :math:`c`, :math:`\\alpha`,
        :math:`\\beta`, and :math:`\\gamma`. Angles are in radians.
    symops : 1d array, str
        Space group symmetry operations.
    dmin : float, optional
        Minimum d-spacing. Default ``0.3``
    source : str, optional
        Radiation source ``'neutron'``, ``'x-ray'``, or ``'electron'``.
        Default ``'neutron'``.

    Returns
    -------
    h, k, l : 1d array, int
        Miller indices.
    d : 1d array
        d-spacing distance between planes of atoms.
    F : 1d array, complex
        Structure factor.
    mult : 1d array, int
        Multiplicity.

    """

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
    """
    Unit cell.

    Parameters
    ----------
    filename : str
        Name of CIF file.
    tol : float, optional
        Tolerance of unique atom coordinates.

    Methods
    -------
    get_filepath()
        Path of CIF file.
    get_filename()
        Name of CIF file.
    get_sites()
        Atom sites in the unit cell.
    get_active_sites()
        Active atom sites in the unit cell.
    set_active_sites()
        Update active atom sites in the unit cell.
    get_number_atoms_per_unit_cell()
        Total number of atoms in the unit cell.
    get_fractional_coordinates()
        Fractional coordiantes.
    set_fractional_coordinates()
        Update fractional coordiantes of active atoms.
    get_unit_cell_cartesian_atomic_coordinates()
        Cartesian coordiantes.
    get_unit_cell_atoms()
        Atom symbols of active atoms.
    set_unit_cell_atoms()
        Update atom symbols.
    get_occupancies()
        Occupancies.
    set_occupancies()
        Update occupancies.
    get_anisotropic_displacement_parameters()
        Anisotropic displacement parameters in crystal coordinates.
    set_anisotropic_displacement_parameters()
        Update anisotropic displacement parameters in crystal coordinates.
    get_isotropic_displacement_parameter()
        Isotropic displacement parameters.
    set_isotropic_displacement_parameter()
        Update isotropic displacement parameters.
    get_principal_displacement_parameters()
        Principal displacement parameters in Cartesian coordinates.
    get_cartesian_anistropic_displacement_parameters()
        Anisotropic displacement parameters in Cartesian coordinates.
    get_crystal_axis_magnetic_moments()
        Magnetic moments in crystal coordinates.
    set_crystal_axis_magnetic_moments()
        Update magnetic moments in crystal coordinates.
    get_magnetic_moment_magnitude()
        Magnitude of magnetic moments.
    get_cartesian_magnetic_moments()
        Magnetic moments in Cartesian coordinates.
    get_g_factors()
        g-factors.
    set_g_factors()
        Update g-factors.
    get_lattice_constants()
        Lattice parameters.
    set_lattice_constants()
        Update lattice parameters.
    get_reciprocal_lattice_constants()
        Reciprocal lattice parameters.
    get_symmetry_operators()
        Symmetry operators.
    get_magnetic_symmetry_operators()
        Magnetic symmetry operators.
    get_lattice_system()
        Lattice system of unit cell.
    get_lattice_volume()
        Lattice volume of unit cell.
    get_reciprocal_lattice_volume()
        Reciprocal lattice volume of reciprocal cell.
    get_metric_tensor()
        Unit cell metric tensor.
    get_reciprocal_metric_tensor()
        Reciprocal cell metric tensor.
    get_fractional_cartesian_transform()
        Fractional to Cartesian coordinates transform matrix.
    get_miller_cartesian_transform()
        Miller to Cartesian coordinates transform matrix.
    get_cartesian_rotation()
        Transform matrix between Cartesian axes of real and reciprocal lattice.
    get_moment_cartesian_transform()
        Magnetic moment components crystal to Cartesian transfomrmation matrix.
    get_atomic_displacement_cartesian_transform()
        Atomic displacement parameters crystal to Cartesian transfomrmation
        matrix.
    get_space_group_symbol()
        Space group symbol.
    get_space_group_number()
        Space group number.
    get_laue()
        Laue class.
    get_site_symmetries()
        Site symmetry operators.
    get_wyckoff_special_positions()
        Wyckoff special positions.
    get_site_multiplicities()
        Site multiplicites.

    """

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
        """
        Path of CIF file.

        Returns
        -------
        filepath : str
            Name of path excluding filename.

        """

        return self.__folder

    def get_filename(self):
        """
        Name of CIF file.

        Returns
        -------
        filename : str
            Name of filename excluding path.

        """

        return self.__filename

    def get_sites(self):
        """
        Atom sites in the unit cell.

        Returns
        -------
        sites : 1d array, int
            All site numbers.

        """

        return self.__site

    def get_active_sites(self):
        """
        Active atom sites in the unit cell.

        Returns
        -------
        sites : 1d array, int
            All active site numbers.

        """

        return self.__act

    def set_active_sites(self, act):
        """
        Update active atom sites in the unit cell.

        Parameters
        ----------
        act : 1d array, int
            All active site numbers.

        """

        self.__act = act

        self.__mask = self.__ind[self.__act]
        self.__index = self.__act[self.__site]
        self.__inverse = np.arange(self.__act.size)[self.__site][self.__index]

    def get_number_atoms_per_unit_cell(self):
        """
        Total number of atoms in the unit cell.

        Returns
        -------
        n_atm : int
            All active atoms.

        """

        return self.__act[self.__site].sum()

    def get_fractional_coordinates(self):
        """
        Fractional coordiantes of active atoms

        Returns
        -------
        u, v, w : 1d array
           Fractional coordiantes :math:`u`, :math:`v`, and :math:`w`.

        """

        mask = self.__mask

        return self.__u[mask], self.__v[mask], self.__w[mask]

    def set_fractional_coordinates(self, u, v, w):
        """
        Update fractional coordiantes of active atoms.

        Parameters
        ----------
        u, v, w : 1d array
           Fractional coordiantes :math:`u`, :math:`v`, and :math:`w`.

        """

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
        """
        Cartesian coordiantes of active atoms.

        Returns
        -------
        rx, ry, rz : 1d array
           Cartesian coordiantes :math:`r_x`, :math:`r_y`, and :math:`r_z`.

        """

        A = self.get_fractional_cartesian_transform()
        u, v, w = self.get_fractional_coordinates()

        return crystal.transform(u, v, w, A)

    def get_unit_cell_atoms(self):
        """
        Atom symbols of active atoms.

        Returns
        -------
        atm : 1d array
            Atom symbols.

        """

        mask = self.__mask

        return self.__atm[mask]

    def set_unit_cell_atoms(self, atm):
        """
        Update atom symbols of active atoms.

        Parameters
        ----------
        atm : 1d array
            Atom symbols.

        """

        ind = self.__index
        inv = self.__inverse

        self.__atm[ind] = atm[inv]

    def get_occupancies(self):
        """
        Occupancies of active atoms.

        Returns
        -------
        occ : 1d array
            Site occupancies.

        """

        mask = self.__mask

        return self.__occ[mask]

    def set_occupancies(self, occ):
        """
        Update occupancies of active atoms.

        Parameters
        ----------
        occ : 1d array
            Site occupancies.

        """

        ind = self.__index
        inv = self.__inverse

        self.__occ[ind] = occ[inv]

    def get_anisotropic_displacement_parameters(self):
        """
        Anisotropic displacement parameters in crystal coordinates of active
        atoms.

        Returns
        -------
        U11, U22, U33, U23, U13, U12 : 1d array
            Atomic displacement parameters :math:`U_{11}`, :math:`U_{22}`,
            :math:`U_{33}`, :math:`U_{23}`, :math:`U_{13}`, and :math:`U_{12}`.

        """

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
        """
        Update anisotropic displacement parameters in crystal coordinates of
        active atoms.

        Parameters
        ----------
        U11, U22, U33, U23, U13, U12 : 1d array
            Atomic displacement parameters :math:`U_{11}`, :math:`U_{22}`,
            :math:`U_{33}`, :math:`U_{23}`, :math:`U_{13}`, and :math:`U_{12}`.

        """

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

    def get_isotropic_displacement_parameter(self):
        """
        Isotropic displacement parameters of active atoms.

        Returns
        -------
        Uiso : 1d array
            Isotropic atomic displacement parameters :math:`U_\mathrm{iso}`.

        """

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.isotropic(U11, U22, U33, U23, U13, U12, D)

    def set_isotropic_displacement_parameter(self, Uiso):
        """
        Update isotropic displacement parameters of active atoms.

        Parameters
        ----------
        Uiso : 1d array
            Isotropic atomic displacement parameters :math:`U_\mathrm{iso}`.

        """

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

    def get_principal_displacement_parameters(self):
        """
        Principal displacement parameters in Cartesian coordinates of active
        atoms.

        Returns
        -------
        U1, U2, U3 : 1d array
            Atomic displacement parameters :math:`U_1`, :math:`U_2`, and
            :math:`U_3`.

        """

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.principal(U11, U22, U33, U23, U13, U12, D)

    def get_cartesian_anistropic_displacement_parameters(self):
        """
        Anisotropic displacement parameters in Cartesian coordinates of active
        atoms.

        Returns
        -------
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy : 1d array
            Atomic displacement parameters :math:`U_{xx}`, :math:`U_{yy}`,
            :math:`U_{zz}`, :math:`U_{yz}`, :math:`U_{xz}`, and :math:`U_{xy}`.

        """

        D = self.get_atomic_displacement_cartesian_transform()
        adps = self.get_anisotropic_displacement_parameters()

        U11, U22, U33, U23, U13, U12 = adps

        return displacive.cartesian(U11, U22, U33, U23, U13, U12, D)

    def get_crystal_axis_magnetic_moments(self):
        """
        Magnetic moments in crystal coordinates of active atoms.

        Returns
        -------
        mu1, mu2, mu3 : 1d array
            Magnetic moments :math:`\mu_1`, :math:`\mu_2`, and :math:`\mu_3`.

        """

        mask = self.__mask

        mu1 = self.__mu1[mask]
        mu2 = self.__mu2[mask]
        mu3 = self.__mu3[mask]

        return mu1, mu2, mu3

    def set_crystal_axis_magnetic_moments(self, mu1, mu2, mu3):
        """
        Update magnetic moments in crystal coordinates of active atoms.

        Parameters
        ----------
        mu1, mu2, mu3 : 1d array
            Magnetic moments :math:`\mu_1`, :math:`\mu_2`, and :math:`\mu_3`.

        """

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
        """
        Magnitude of magnetic moments of active atoms.

        Returns
        -------
        mu : 1d array
            Moment of magnetic moments :math:`\mu`.

        """

        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()

        return magnetic.magnitude(mu1, mu2, mu3, C)

    def get_cartesian_magnetic_moments(self):
        """
        Magnetic moments in Cartesian coordinates of active atoms.

        Returns
        -------
        mu_x, mu_y, mu_z : 1d array
            Magnetic moments :math:`\mu_x`, :math:`\mu_y`, and :math:`\mu_z`.

        """

        C = self.get_moment_cartesian_transform()
        mu1, mu2, mu3 = self.get_crystal_axis_magnetic_moments()

        return magnetic.cartesian(mu1, mu2, mu3, C)

    def get_g_factors(self):
        """
        g-factors of active ions.

        Returns
        -------
        g : 1d array
           Magnetic :math:`g`-factors.

        """

        mask = self.__mask

        return self.__g[mask]

    def set_g_factors(self, g):
        """
        Update g-factors of active ions.

        Paramters
        ---------
        g : 1d array
           Magnetic :math:`g`-factors.

        """

        ind = self.__index
        inv = self.__inverse

        self.__g[ind] = g[inv]

    def get_lattice_constants(self):
        """
        Lattice parameters.

        ============ =============================
        Cell         Parameters
        ============ =============================
        Cubic        a
        Hexagonal    a, c
        Rhombohedral a, alpha
        Tetragonal   a, c
        Orthorhombic a, b, c
        Monoclinic   a, b, c, alpha, beta or gamma
        Triclinic    a, b, c, alpha, beta, gamma
        ============ =============================

        Returns
        -------
        constants : tuple
            Non-constrained lattice constants and angles. Angles in radians.

        """

        lat = self.get_lattice_system()

        a = self.__a
        b = self.__b
        c = self.__c
        alpha = self.__alpha
        beta = self.__beta
        gamma = self.__gamma

        if (lat == 'Cubic'):
            constants = a,
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
        """
        Update lattice parameters.

        ============ =============================
        Cell         Parameters
        ============ =============================
        Cubic        a
        Hexagonal    a, c
        Rhombohedral a, alpha
        Tetragonal   a, c
        Orthorhombic a, b, c
        Monoclinic   a, b, c, alpha, beta or gamma
        Triclinic    a, b, c, alpha, beta, gamma
        ============ =============================

        Parameters
        ----------
        constants : tuple
            Non-constrained lattice constants and angles. Angles in radians.

        """

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
        """
        Reciprocal lattice parameters.

        Returns
        -------
        a_, b_, c_, alpha_, beta_, gamma_ : float
            Reciprocal lattice constants and angles :math:`a^*`, :math:`b^*`,
            :math:`c^*`, :math:`\\alpha^*`, :math:`\\beta^*` and
            :math:`\\gamma^*`. Angles in radians.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.reciprocal(*constants)

    def get_symmetry_operators(self):
        """
        Symmetry operators of active atoms.

        Returns
        -------
        op : 1d array, str
            Symmetry operator of each site.

        """

        mask = self.__mask

        return self.__op[mask]

    def get_magnetic_symmetry_operators(self):
        """
        Magnetic symmetry operators of active atoms.

        Returns
        -------
        mag_op : 1d array, str
            Magnetic symmetry operator of each site.

        """

        mask = self.__mask

        return self.__mag_op[mask]

    def get_lattice_system(self):
        """
        Lattice system of unit cell.

        Returns
        -------
        system : str
           One of ``'Cubic'``, ``'Hexagonal'``, ``'Rhombohedral'``,
           ``'Tetragonal'``, ``'Orthorhombic'``, ``'Monoclinic'``, or
           ``'Triclinic'``.

        """

        return self.__lat

    def get_lattice_volume(self):
        """
        Lattice volume of unit cell.

        .. math:: V = abc\sqrt{1-\cos^2{\\alpha}-\cos^2{\\beta}-\cos^2{\\gamma}
                               +2\cos{\\alpha}\cos{\\beta}\cos{\\gamma}}

        Returns
        -------
        V : float
           Unit cell volume :math:`V`.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.volume(*constants)

    def get_reciprocal_lattice_volume(self):
        """
        Reciprocal lattice volume of reciprocal cell.

        .. math:: V^* = a^*b^*c^*\sqrt{1-\cos^2{\\alpha^*}-\cos^2{\\beta^*}
                                        -\cos^2{\\gamma^*}+2\cos{\\alpha^*}
                                         \cos{\\beta^*}\cos{\\gamma^*}}

        Returns
        -------
        V_ : float
           Reciprocal unit cell volume :math:`V^*`.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.volume(*constants)

    def get_metric_tensor(self):
        """
        Unit cell metric tensor.

        .. math:: G = \\begin{bmatrix}
            a^2 & ab\cos{\\gamma} & ac\cos{\\beta}  \\\\
            ba\cos{\\gamma} & b^2 & bc\cos{\\alpha} \\\\
            ca\cos{\\beta} & cb\cos{\\alpha} & c^2
            \\end{bmatrix}

        Returns
        -------
        G : 2d array
           Components of the :math:`G` metric tensor.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.metric(*constants)

    def get_reciprocal_metric_tensor(self):
        """
        Reciprocal cell metric tensor.

        .. math:: G^* = \\begin{bmatrix}
            (a^*)^2 & a^*b^*\cos{\\gamma} & a^*c^*\cos{\\beta} \\\\
            b^*a^*\cos{\\gamma} & (b^*)^2 & b^*c^*\cos{\\alpha} \\\\
            c^*a^*\cos{\\beta} & c^*b^*\cos{\\alpha} & (c^*)^2
            \\end{bmatrix}

        Returns
        -------
        G_ : 2d array
           Components of the :math:`G^*` metric tensor.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.metric(*constants)

    def get_fractional_cartesian_transform(self):
        """
        Trasform matrix from fractional to Cartesian coordinates.

        .. math:: \\begin{bmatrix} r_x \\\\ r_y \\\\ r_z \\end{bmatrix} =
            \\begin{bmatrix}
            A_{11} & A_{12} & A_{13} \\\\
            0      & A_{22} & A_{23} \\\\
            0      & 0      & A_{33}
            \\end{bmatrix} \\begin{bmatrix} u \\\\ v \\\\ w \\end{bmatrix}

        Returns
        -------
        A : 2d array
           Components of the :math:`A` matrix.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian(*constants)

    def get_miller_cartesian_transform(self):
        """
        Trasform matrix from Miller to Cartesian coordinates.

        .. math:: \\begin{bmatrix} Q_x \\\\ Q_y \\\\ Q_z \\end{bmatrix} = 2\\pi
            \\begin{bmatrix}
            R_{11} & R_{12} & R_{13} \\\\
            R_{21} & R_{22} & R_{23} \\\\
            R_{31} & R_{32} & R_{33}
            \\end{bmatrix}
            \\begin{bmatrix}
            B_{11} & B_{12} & B_{13} \\\\
            0      & B_{22} & B_{23} \\\\
            0      & 0      & B_{33}
            \\end{bmatrix} \\begin{bmatrix} h \\\\ k \\\\ l \\end{bmatrix}

        Returns
        -------
        B : 2d array
           Components of the :math:`B` matrix.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian(*constants)

    def get_cartesian_rotation(self):
        """
        Transform matrix between Cartesian axes of real and reciprocal lattice.

        .. math:: \\begin{bmatrix} Q_x \\\\ Q_y \\\\ Q_z \\end{bmatrix} = 2\\pi
            \\begin{bmatrix}
            R_{11} & R_{12} & R_{13} \\\\
            R_{21} & R_{22} & R_{23} \\\\
            R_{31} & R_{32} & R_{33}
            \\end{bmatrix}
            \\begin{bmatrix}
            B_{11} & B_{12} & B_{13} \\\\
            0      & B_{22} & B_{23} \\\\
            0      & 0      & B_{33}
            \\end{bmatrix} \\begin{bmatrix} h \\\\ k \\\\ l \\end{bmatrix}

        Returns
        -------
        R : 2d array
           Components of the :math:`R` matrix.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_rotation(*constants)

    def get_moment_cartesian_transform(self):
        """
        Transform matrix between crystal and Cartesian coordinates for magnetic
        moments.

        .. math::
            \\begin{bmatrix} \\mu_x \\\\ \\mu_y \\\\ \\mu_z \\end{bmatrix} =
            \\begin{bmatrix}
            C_{11} & C_{12} & C_{13} \\\\
            C_{21} & C_{22} & C_{23} \\\\
            C_{31} & C_{32} & C_{33}
            \\end{bmatrix}
            \\begin{bmatrix} \\mu_1 \\\\ \\mu_2 \\\\ \\mu_3 \\end{bmatrix}

        Returns
        -------
        C : 2d array
           Components of the :math:`C` matrix.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_moment(*constants)

    def get_atomic_displacement_cartesian_transform(self):
        """
        Transform matrix between crystal and Cartesian coordinates for atomic
        displacement parameters.

        .. math::
            \\begin{bmatrix}
            U_{xx} & U_{xy} & U_{xz} \\\\
            U_{yx} & U_{yy} & U_{yz} \\\\
            U_{zx} & U_{zy} & U_{zz}
            \\end{bmatrix} =
            \\begin{bmatrix}
            D_{11} & D_{12} & D_{13} \\\\
            D_{21} & D_{22} & D_{23} \\\\
            D_{31} & D_{32} & D_{33}
            \\end{bmatrix}
            \\begin{bmatrix}
            U_{11} & U_{12} & U_{13} \\\\
            U_{21} & U_{22} & U_{23} \\\\
            U_{31} & U_{32} & U_{33}
            \\end{bmatrix}
            \\begin{bmatrix}
            D_{11} & D_{21} & D_{31} \\\\
            D_{12} & D_{22} & D_{32} \\\\
            D_{13} & D_{23} & D_{33}
            \\end{bmatrix}

        Returns
        -------
        D : 2d array
           Components of the :math:`D` matrix.

        """

        constants = self.__get_all_lattice_constants()

        return crystal.cartesian_displacement(*constants)

    def get_space_group_symbol(self):
        """
        Space group symbol.

        Returns
        -------
        hm : str
           Symbol in Hermann–Mauguin notation.

        """

        return self.__hm

    def get_space_group_number(self):
        """
        Space group number.

        Returns
        -------
        group : int
           Number between 1 and 230.

        """

        return self.__sg

    def get_laue(self):
        """
        Laue class.

        Returns
        -------
        laue : str
           One of ``'-1'``, ``'2/m'``, ``'mmm'``, ``'4/m'``, ``'4/mmm'``,
           ``'-3'``, ``'-3m'``, ``'6/m'``, ``'6/mmm'``, ``'m-3'``, or
           ``'m-3m'``.

        """

        return self.__laue

    def get_site_symmetries(self):
        """
        Site symmetry operators.

        Returns
        -------
        pg : 1d array, str
            Point group symmetry of each site.

        """

        mask = self.__mask

        return self.__pg[mask]

    def get_wyckoff_special_positions(self):
        """
        Wyckoff special positions of active atoms.

        Returns
        -------
        sp_pos : 1d array, str
            Special position of eac site

        """

        mask = self.__mask

        return self.__sp_pos[mask]

    def get_site_multiplicities(self):
        """
        Site multiplicites of active atoms.

        Returns
        -------
        mult : 1d array, int
            Multiplicity of each site.

        """

        mask = self.__mask

        return self.__mult[mask]

class SuperCell(UnitCell):
    """
    Supercell.

    Parameters
    ----------
    filename : str
        Name of CIF file.
    nu, nv, nw : int
        Extents :math:`N_1`, :math:`N_2`, :math:`N_3` along the :math:`a`,
        :math:`b`, and :math:`c`-axis of the supercell.
    tol : float, optional
        Tolerance of unique atom coordinates.

    Methods
    -------
    get_super_cell_extents()
        Number of cells along each dimension.
    set_super_cell_extents()
        Update number of cells along each dimension.
    get_super_cell_size()
        Total number of cells.
    get_number_atoms_per_super_cell()
        Total number of atoms.
    get_cartesian_lattice_points()
        Position of lattice points in Cartesian coordinates,
    get_super_cell_cartesian_atomic_coordinates()
        Atom positions in Cartesian coordinates.

    """

    def __init__(self, filename, nu=1, nv=1, nw=1, tol=1e-2):

        super(SuperCell, self).__init__(filename, tol)

        self.set_super_cell_extents(nu, nv, nw)

    def get_super_cell_extents(self):
        """
        Number of cells along each dimension.

        Returns
        -------
        nu, nv, nw : int
            Extents :math:`N_1`, :math:`N_2`, :math:`N_3` along the :math:`a`,
            :math:`b`, and :math:`c`-axis of the supercell.

        """

        return self.__nu, self.__nv, self.__nw

    def set_super_cell_extents(self, nu, nv, nw):
        """
        Update number of cells along each dimension.

        Parameters
        ----------
        nu, nv, nw : int
            Extents :math:`N_1`, :math:`N_2`, :math:`N_3` along the :math:`a`,
            :math:`b`, and :math:`c`-axis of the supercell.

        """

        self.__nu = nu
        self.__nv = nv
        self.__nw = nw

    def get_super_cell_size(self):
        """
        Total number of cells.

        Returns
        -------
        n_uvw : int
            Supercell size :math:`N_1N_2N_3`.

        """

        nu, nv, nw = self.get_super_cell_dimensions()

        return nu*nv*nw

    def get_number_atoms_per_super_cell(self):
        """
        Total number of atoms.

        Returns
        -------
        n : int
            Number of atoms :math:`n` in the supercell.

        """

        n_uvw = self.get_super_cell_size()
        n_atm = self.get_number_atoms_per_unit_cell()

        return n_uvw*n_atm

    def get_cartesian_lattice_points(self):
        """
        Position of lattice points in Cartesian coordinates,

        Returns
        -------
        Rx, Ry, Rz : 1d array
            Lattice vectors :math:`R_x`, :math:`R_y`, and :math:`R_z`.

        """

        A = self.get_fractional_cartesian_transform()
        nu, nv, nw = self.get_super_cell_dimensions()

        return space.cell(nu, nv, nw, A)

    def get_super_cell_cartesian_atomic_coordinates(self):
        """
        Atom positions in Cartesian coordinates.

        Returns
        -------
        rx, ry, rz : 1d array
            Spatial vectors :math:`r_x`, :math:`r_y`, and :math:`r_z`.
        atms : 1d array, str
            Atoms, ions, or isotopes.

        """

        ux, uy, uz = self.get_unit_cell_cartesian_atomic_coordinates()
        ix, iy, iz = self.get_cartesian_lattice_points()
        atm = self.get_unit_cell_atoms()

        return space.real(ux, uy, uz, ix, iy, iz, atm)
