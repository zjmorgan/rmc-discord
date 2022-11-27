#!/usr/bin/env/python3

import os
import h5py
import numpy as np

from disorder.diffuse import scattering, space, powder, monocrystal, filters
from disorder.diffuse import experimental
from disorder.diffuse import displacive, magnetic, occupational
from disorder.correlation import functions
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

    coordinate = [h,k,l]

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

    h, k, l, d, Q = h[ind], k[ind], l[ind], d[ind], Q[ind]

    ind = np.lexsort((h,k,l,d),axis=0)[::-1]

    h, k, l, d, Q, mult = h[ind], k[ind], l[ind], d[ind], Q[ind], mult[ind]

    n_hkl = Q.size

    phase_factor = np.exp(2j*np.pi*(h[:,np.newaxis]*u+
                                    k[:,np.newaxis]*v+
                                    l[:,np.newaxis]*w))

    if source == 'neutron':
        scattering_power = scattering.length(atms, n_hkl).reshape(n_hkl,n_atm)
    else:
        scattering_power = scattering.form(atms, Q).reshape(n_hkl,n_atm)

    dw_factors = np.exp(-2*np.pi**2*(U11*(h*a_)[:,np.newaxis]**2+
                                     U22*(k*b_)[:,np.newaxis]**2+
                                     U33*(l*c_)[:,np.newaxis]**2+
                                     U23*(k*l*b_*c_*2)[:,np.newaxis]+
                                     U13*(h*l*a_*c_*2)[:,np.newaxis]+
                                     U12*(h*k*a_*b_*2)[:,np.newaxis]))

    factors = scattering_power*occupancy*dw_factors*phase_factor

    F = factors.sum(axis=1)

    return h, k, l, d, F, mult

def factor_magnetic(mu1, mu2, mu3, kvecs, u, v, w, atms, occupancy,
                    U11, U22, U33, U23, U13, U12,
                    a, b, c, alpha, beta, gamma, symops, dmin=1.0):
    """
    Magnetic tructure factor :math:`F(h,k,l)`.

    """

    p = 0.2695*10

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

    Qh, Qk, Ql = crystal.vector(h, k, l, B)

    n_hkl = Q.size

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

    h, k, l, d, Q = h[ind], k[ind], l[ind], d[ind], Q[ind]

    ind = np.lexsort((h,k,l,d),axis=0)[::-1]

    h, k, l, d, Q, mult = h[ind], k[ind], l[ind], d[ind], Q[ind], mult[ind]

    if len(kvecs) > 0:

        kvecs = np.array(kvecs)
        kvecs = np.vstack([kvecs,-kvecs])

        h, k, l = np.rollaxis(np.stack([h,k,l])+kvecs[...,np.newaxis], axis=1)

        h, k, l = h.flatten(), k.flatten(), l.flatten()

    Qh, Qk, Ql = crystal.vector(h, k, l, B)

    eh, ek, el, Q = space.unit(Qh, Qk, Ql)

    n_hkl = Q.size

    phase_factor = np.exp(2j*np.pi*(h[:,np.newaxis]*u+
                                    k[:,np.newaxis]*v+
                                    l[:,np.newaxis]*w))

    magnetic_form = magnetic.form(Q, atms).reshape(n_hkl,n_atm)


    e_dot_mu = eh[:,np.newaxis]*mu1+ek[:,np.newaxis]*mu2+el[:,np.newaxis]*mu3

    mu1_perp = mu1-e_dot_mu*eh[:,np.newaxis]
    mu2_perp = mu2-e_dot_mu*ek[:,np.newaxis]
    mu3_perp = mu3-e_dot_mu*el[:,np.newaxis]

    dw_factors = np.exp(-2*np.pi**2*(U11*(h*a_)[:,np.newaxis]**2+
                                     U22*(k*b_)[:,np.newaxis]**2+
                                     U33*(l*c_)[:,np.newaxis]**2+
                                     U23*(k*l*b_*c_*2)[:,np.newaxis]+
                                     U13*(h*l*a_*c_*2)[:,np.newaxis]+
                                     U12*(h*k*a_*b_*2)[:,np.newaxis]))

    factors = magnetic_form*occupancy*dw_factors*phase_factor

    M1 = p*(mu1_perp*factors).sum(axis=1)
    M2 = p*(mu2_perp*factors).sum(axis=1)
    M3 = p*(mu3_perp*factors).sum(axis=1)

    return h, k, l, d, M1, M2, M3, mult

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
    save()
        Save unit cell data.
    load()
        Load unit cell data.
    get_filepath()
        Path of CIF file.
    get_filename()
        Name of CIF file.
    get_sites()
        Atom sites in the unit cell.
    get_atom_sites()
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
    get_unit_cell_charge_numbers()
        Charge numbers of active ions.
    set_unit_cell_charge_numbers()
        Update charge numbers of active ions.
    get_unit_cell_isotope_numbers()
        Mass numbers of active isotopes.
    set_unit_cell_isotope_numbers()
        Update mass numbers of active isotopes.
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
    get_all_lattice_constants()
        All lattice parameters.
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
    get_twins()
        Twin transformation matrices and mass fractions.
    set_twins()
        Update twin transformation matrices and mass fractions.
    get_space_group_symmetry_operators()
        All symmetry operators.

    """

    def __init__(self, filename, tol=1e-2):

        name, ext = os.path.splitext(filename)

        if '.cif' == ext or '.mcif' == ext:

            filename = os.path.abspath(filename)

            folder = os.path.dirname(filename)
            filename = os.path.basename(filename)

            self.__folder = folder
            self.__filename = filename

            self. __load_unit_cell(tol)

        else:

            self.__load_unitcell(filename)

    def __repr__(self):

        lat = self.get_lattice_system()

        sym = self.get_space_group_symbol()
        no = self.get_space_group_number()

        a, b, c, alpha, beta, gamma = self.get_all_lattice_constants()

        standard = np.isclose(gamma, np.pi/2)

        a = 'a = {:.4}'.format(a)
        b = 'b = {:.4}'.format(b)
        c = 'c = {:.4}'.format(c)
        alpha = 'alpha = {:.2}'.format(np.rad2deg(alpha))
        beta = 'beta = {:.2}'.format(np.rad2deg(beta))
        gamma = 'gamma = {:.2}'.format(np.rad2deg(gamma))

        if lat == 'Cubic':
            constants = a,
        elif lat == 'Hexagonal' or lat == 'Tetragonal':
            constants = a, c
        elif lat == 'Rhobmohedral':
            constants = a, alpha
        elif lat == 'Orthorhombic':
            constants = a, b, c
        elif lat == 'Monoclinic':
            if not standard:
                constants = a, b, c, gamma
            else:
                constants = a, b, c, beta
        else:
            constants = a, b, c, alpha, beta, gamma

        title = '{} {}, {}\n'.format(sym,no,', '.join(constants))

        atms = self.get_unit_cell_atoms()
        occ = self.get_occupancies()
        u, v, w = self.get_fractional_coordinates()
        Uiso = self.get_isotropic_displacement_parameter()

        header = 'atm    occ     u     v     w  Uiso\n'
        divide = '==================================\n'
        coords = '{:4}{:6.2}{:6.2}{:6.2}{:6.2}{:6.2}\n'

        sites = ''.join([coords.format(*x) for x in zip(atms,occ,u,v,w,Uiso)])

        return title+divide+header+divide+sites+divide

    def __load_unit_cell(self, tol):

        folder = self.get_filepath()
        filename = self.get_filename()

        constants = crystal.parameters(folder, filename)

        a, b, c, alpha, beta, gamma = constants

        self.__a, self.__b, self.__c = a, b, c
        self.__alpha, self.__beta, self.__gamma = alpha, beta, gamma

        uc_dict = crystal.unitcell(folder, filename, tol=tol)

        n_atm = uc_dict['n_atom']

        self.__ion = uc_dict['ion'].astype('<U3')
        self.__iso = uc_dict['isotope'].astype('<U3')
        self.__atm = uc_dict['atom'].astype('<U3')
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

        if displacement.shape[1] == 1:
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

        hm, sg = crystal.group(folder, filename)

        self.__hm = hm
        self.__sg = sg

        lat = crystal.lattice(*constants)

        self.__lat = lat

        laue = crystal.laue(folder, filename)

        self.__laue = laue

        self.__pg = np.empty(n_atm, dtype=object)
        self.__mult = np.empty(n_atm, dtype=int)
        self.__sp_pos = np.empty(n_atm, dtype=object)

        A = self.get_fractional_cartesian_transform()

        operators = crystal.operators(folder, filename)

        for i, (u, v, w) in enumerate(zip(self.__u,self.__v,self.__w)):

            pg, mult, sp_pos = symmetry.site(operators, [u,v,w], A, tol=1e-2)

            self.__pg[i] = pg
            self.__mult[i] = mult
            self.__sp_pos[i] = sp_pos[0]

        T, weights = crystal.twins(folder, filename)

        self.__T = T
        self.__weights = weights

        self.__symops = crystal.operators(folder, filename)

    def __save_unitcell(self, filename):

        with h5py.File(filename, 'w') as f:

            uc = f.create_group('disorder/unitcell')

            uc.create_dataset('ions', data=self.__ion.astype('S'))
            uc.create_dataset('isotopes', data=self.__iso.astype('S'))
            uc.create_dataset('atoms', data=self.__atm.astype('S'))
            uc.create_dataset('sites', data=self.__site)

            uc.create_dataset('active', data=self.__act)
            uc.create_dataset('unique', data=self.__ind)
            uc.create_dataset('index', data=self.__index)
            uc.create_dataset('inverse', data=self.__inverse)
            uc.create_dataset('mask', data=self.__mask)

            uc.create_dataset('operators', data=self.__op.astype('S'))
            uc.create_dataset('moments', data=self.__mag_op.astype('S'))

            uc.create_dataset('u', data=self.__u)
            uc.create_dataset('v', data=self.__v)
            uc.create_dataset('w', data=self.__w)

            uc.create_dataset('occupancy', data=self.__occ)

            uc.create_dataset('U11', data=self.__U11)
            uc.create_dataset('U22', data=self.__U22)
            uc.create_dataset('U33', data=self.__U33)
            uc.create_dataset('U23', data=self.__U23)
            uc.create_dataset('U13', data=self.__U13)
            uc.create_dataset('U12', data=self.__U12)

            uc.create_dataset('mu1', data=self.__mu1)
            uc.create_dataset('mu2', data=self.__mu2)
            uc.create_dataset('mu3', data=self.__mu3)

            uc.create_dataset('g', data=self.__g)

            uc.create_dataset('point_group', data=self.__pg)
            uc.create_dataset('multiplicity', data=self.__mult)
            uc.create_dataset('positions', data=self.__sp_pos)

            uc.create_dataset('T', data=self.__T)
            uc.create_dataset('weights', data=self.__weights)

            uc.create_dataset('symops', data=self.__symops)

            uc.attrs['folder'] = self.__folder
            uc.attrs['filename'] = self.__filename

            uc.attrs['a'] = self.__a
            uc.attrs['b'] = self.__b
            uc.attrs['c'] = self.__c
            uc.attrs['alpha'] = self.__alpha
            uc.attrs['beta'] = self.__beta
            uc.attrs['gamma'] = self.__gamma

            uc.attrs['hermann_mauguin'] = self.__hm
            uc.attrs['space_group'] = self.__sg
            uc.attrs['laue'] = self.__laue

            uc.attrs['lattice'] = self.__lat

    def __load_unitcell(self, filename):

        with h5py.File(filename, 'r') as f:

            uc = f['disorder/unitcell']

            self.__ion = uc['ions'][...].astype('<U3')
            self.__iso = uc['isotopes'][...].astype('<U3')
            self.__atm = uc['atoms'][...].astype('<U3')
            self.__site = uc['sites'][...]

            self.__act = uc['active'][...]
            self.__ind = uc['unique'][...]
            self.__index = uc['index'][...]
            self.__inverse = uc['inverse'][...]
            self.__mask = uc['mask'][...]

            self.__op = uc['operators'][...].astype('U')
            self.__mag_op = uc['moments'][...].astype('U')

            self.__u = uc['u'][...]
            self.__v = uc['v'][...]
            self.__w = uc['w'][...]

            self.__occ = uc['occupancy'][...]

            self.__U11 = uc['U11'][...]
            self.__U22 = uc['U22'][...]
            self.__U33 = uc['U33'][...]
            self.__U23 = uc['U23'][...]
            self.__U13 = uc['U13'][...]
            self.__U12 = uc['U12'][...]

            self.__mu1 = uc['mu1'][...]
            self.__mu2 = uc['mu2'][...]
            self.__mu3 = uc['mu3'][...]

            self.__g = uc['g'][...]

            self.__pg  = uc['point_group'][...]
            self.__mult = uc['multiplicity'][...]
            self.__sp_pos = uc['positions'][...]

            self.__T = uc['T'][...]
            self.__weights = uc['weights'][...]

            self.__symops = uc['symops'][...]

            self.__folder = uc.attrs['folder']
            self.__filename = uc.attrs['filename']

            self.__a = uc.attrs['a']
            self.__b = uc.attrs['b']
            self.__c = uc.attrs['c']
            self.__alpha = uc.attrs['alpha']
            self.__beta = uc.attrs['beta']
            self.__gamma = uc.attrs['gamma']

            self.__hm = uc.attrs['hermann_mauguin']
            self.__sg = uc.attrs['space_group']
            self.__laue = uc.attrs['laue']

            self.__lat = uc.attrs['lattice']

    def save(self, filename):
        """
        Save unit cell data.

        Parameters
        ----------
        filename : str
            Name of HDF5 file.

        """

        self.__save_unitcell(filename)

    def load(self, filename):
        """
        Load unit cell data.

        Parameters
        ----------
        filename : str
            Name of HDF5 file.

        """

        self.__load_unitcell(filename)

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
            Name of file excluding path.

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

    def get_atom_sites(self):
        """
        Atom site symbols in the unit cell.

        Returns
        -------
        atms : 1d array, int
            All atom sites.

        """

        return self.__atm[self.__mask]

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

        ind = self.__index

        return self.__u[ind], self.__v[ind], self.__w[ind]

    def set_fractional_coordinates(self, u, v, w):
        """
        Update fractional coordiantes of active atoms.

        Parameters
        ----------
        u, v, w : 1d array
           Fractional coordiantes :math:`u`, :math:`v`, and :math:`w`.

        """

        ind = self.__index

        operators = self.__op[ind]

        for i, operator in enumerate(operators):
            uvw = symmetry.evaluate([operator], [u[i], v[i], w[i]])
            u[i], v[i], w[i] = np.mod(uvw, 1).flatten()

        self.__u[ind], self.__v[ind], self.__w[ind] = u, v, w

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
        atm : 1d array, str
            Atom symbols.

        """

        ind = self.__index

        return self.__atm[ind]

    def set_unit_cell_atoms(self, atm):
        """
        Update atom symbols of active atoms.

        Parameters
        ----------
        atm : 1d array, str
            Atom symbols.

        """

        ind = self.__index
        inv = self.__inverse

        self.__atm[ind] = atm[inv]

    def get_unit_cell_charge_numbers(self):
        """
        Charge numbers of active ions.

        Returns
        -------
        ion : 1d array, str
            Valence charge number of active ions.

        """

        ind = self.__index

        return self.__ion[ind]

    def set_unit_cell_charge_numbers(self, ion):
        """
        Update charge numbers of active ions.

        Parameters
        ----------
        ion : 1d array, str
            Valence charge number of active ions.

        """

        ind = self.__index
        inv = self.__inverse

        self.__ion[ind] = ion[inv]

    def get_unit_cell_mass_numbers(self):
        """
        Mass numbers of active isotopes.

        Returns
        -------
        iso : 1d array, str
            Nuclide charge number of active ions.

        """

        ind = self.__index

        return self.__iso[ind]

    def set_unit_cell_mass_numbers(self, iso):
        """
        Update mass numbers of active isotopes.

        Parameters
        ----------
        iso : 1d array, str
            Nuclide mass number of active ions.

        """

        ind = self.__index
        inv = self.__inverse

        self.__iso[ind] = iso[inv]

    def get_unit_cell_ions(self):
        """
        Ion symbols of active ions.

        Returns
        -------
        ion : 1d array, str
            Ion symbols.

        """

        ind = self.__index

        atm = self.__atm[ind]
        ion = self.__ion[ind]

        return np.array([a+c for a, c in zip(atm, ion)])

    def get_unit_cell_isotopes(self):
        """
        Isotopes symbols of active isotopes.

        Returns
        -------
        iso : 1d array, str
            Isotopes symbols.

        """

        ind = self.__index

        atm = self.__atm[ind]
        iso = self.__iso[ind]

        return np.array([A+a for a, A in zip(atm, iso)])

    def get_occupancies(self):
        """
        Occupancies of active atoms.

        Returns
        -------
        occ : 1d array
            Site occupancies.

        """

        ind = self.__index

        return self.__occ[ind]

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

        ind = self.__index

        U11 = self.__U11[ind]
        U22 = self.__U22[ind]
        U33 = self.__U33[ind]
        U23 = self.__U23[ind]
        U13 = self.__U13[ind]
        U12 = self.__U12[ind]

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

        ind = self.__index

        operators = self.__op[ind]

        for i, operator in enumerate(operators):
            disp = [U11[i], U22[i], U33[i], U23[i], U13[i], U12[i]]
            disp = symmetry.evaluate_disp([operator], disp)
            U11[i], U22[i], U33[i], U23[i], U13[i], U12[i] = disp

        self.__U11[ind] = U11
        self.__U22[ind] = U22
        self.__U33[ind] = U33
        self.__U23[ind] = U23
        self.__U13[ind] = U13
        self.__U12[ind] = U12

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

        ind = self.__index

        mu1 = self.__mu1[ind]
        mu2 = self.__mu2[ind]
        mu3 = self.__mu3[ind]

        return mu1, mu2, mu3

    def set_crystal_axis_magnetic_moments(self, mu1, mu2, mu3):
        """
        Update magnetic moments in crystal coordinates of active atoms.

        Parameters
        ----------
        mu1, mu2, mu3 : 1d array
            Magnetic moments :math:`\mu_1`, :math:`\mu_2`, and :math:`\mu_3`.

        """

        ind = self.__index

        operators = self.__mag_op[ind]

        for i, operator in enumerate(operators):
            mag = [mu1[i], mu2[i], mu3[i]]
            mag = symmetry.evaluate_mag([operator], mag)
            mu1[i], mu2[i], mu3[i] = np.array(mag).flatten()

        self.__mu1[ind] = mu1
        self.__mu2[ind] = mu2
        self.__mu3[ind] = mu3

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

        ind = self.__index

        return self.__g[ind]

    def set_g_factors(self, g):
        """
        Update g-factors of active ions.

        Parameters
        ----------
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

        if lat == 'Cubic':
            constants = a,
        elif lat == 'Hexagonal' or lat == 'Tetragonal':
            constants = a, c
        elif lat == 'Rhobmohedral':
            constants = a, alpha
        elif lat == 'Orthorhombic':
            constants = a, b, c
        elif lat == 'Monoclinic':
            if np.isclose(beta, np.pi/2):
                constants = a, b, c, gamma
            else:
                constants = a, b, c, beta
        else:
            constants = a, b, c, alpha, beta, gamma

        return constants

    def set_lattice_constants(self, constants):
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

        if lat == 'Cubic':
            a = constants
            b = c = a
        elif lat == 'Hexagonal' or lat == 'Tetragonal':
            a, c = constants
            b = a
        elif lat == 'Rhobmohedral':
            a, alpha = constants
            b = c = a
            beta = gamma = alpha
        elif lat == 'Orthorhombic':
            a, b, c = constants
            alpha = beta = gamma = np.pi/2
        elif lat == 'Monoclinic':
            if np.isclose(beta, np.pi/2):
                a, b, c, gamma = constants
            else:
                a, b, c, beta = constants
        else:
            a, b, c, alpha, beta, gamma = constants

        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    def get_all_lattice_constants(self):
        """
        All lattice parameters.

        Returns
        -------
        constants : tuple
           Lattice constants and angles. Angles in radians.

        """

        constants = self.__a, self.__b, self.__c, \
                    self.__alpha, self.__beta, self.__gamma

        return constants

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

        constants = self.get_all_lattice_constants()

        return crystal.reciprocal(*constants)

    def get_symmetry_operators(self):
        """
        Symmetry operators of active atoms.

        Returns
        -------
        op : 1d array, str
            Symmetry operator of each site.

        """

        ind = self.__index

        return self.__op[ind]

    def get_magnetic_symmetry_operators(self):
        """
        Magnetic symmetry operators of active atoms.

        Returns
        -------
        mag_op : 1d array, str
            Magnetic symmetry operator of each site.

        """

        ind = self.__index

        return self.__mag_op[ind]

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

        constants = self.get_all_lattice_constants()

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

        constants = self.get_reciprocal_lattice_constants()

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

        constants = self.get_all_lattice_constants()

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

        constants = self.get_reciprocal_lattice_constants()

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

        constants = self.get_all_lattice_constants()

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

        constants = self.get_reciprocal_lattice_constants()

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

        constants = self.get_all_lattice_constants()

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

        constants = self.get_all_lattice_constants()

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

        constants = self.get_all_lattice_constants()

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


    def get_space_group_symmetry_operators(self):
        """
        All symmetry operators.

        Returns
        -------
        sym_ops : 1d array, str
            Space group symmetry operators.

        """

        return self.__symops

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

        ind = self.__index

        return self.__pg[ind]

    def get_wyckoff_special_positions(self):
        """
        Wyckoff special positions of active atoms.

        Returns
        -------
        sp_pos : 1d array, str
            Special position of eac site

        """

        ind = self.__index

        return self.__sp_pos[ind]

    def get_site_multiplicities(self):
        """
        Site multiplicites of active atoms.

        Returns
        -------
        mult : 1d array, int
            Multiplicity of each site.

        """

        ind = self.__index

        return self.__mult[ind]

    def get_twins(self):
        """
        Twin transformation matrices and mass fractions.

        Returns
        -------
        T : 3d array
            Twin transformation matrices.
        weights : 1d array
            Twin mass fractions.

        """

        return self.__T, self.__weights

    def set_twins(self, T, weights):
        """
        Update twin transformation matrices and mass fractions.

        Parameters
        ----------
        T : 3d array
            Twin transformation matrices.
        weights : 1d array
            Twin mass fractions.

        """

        self.__T = T
        self.__weights = weights

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
    save()
        Save supercell data.
    load()
        Load supercell data.
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

        name, ext = os.path.splitext(filename)

        if 'cif' in ext:

            self.set_super_cell_extents(nu, nv, nw)

        else:

            self.__load_supercell(filename)

    def __repr__(self):

        dims = self.get_super_cell_extents()
        n_atm = self.get_number_atoms_per_unit_cell()

        return '{}x{}x{} supercell {} atom unit cell'.format(*dims,n_atm)

    def __save_supercell(self, filename):

        super().save(filename)

        with h5py.File(filename, 'a') as f:

            sc = f.create_group('disorder/supercell')

            sc.create_dataset('Sx', data=self._Sx)
            sc.create_dataset('Sy', data=self._Sy)
            sc.create_dataset('Sz', data=self._Sz)

            sc.create_dataset('A_r', data=self._A_r)

            sc.create_dataset('Ux', data=self._Ux)
            sc.create_dataset('Uy', data=self._Uy)
            sc.create_dataset('Uz', data=self._Uz)

            sc.attrs['nu'] = self.__nu
            sc.attrs['nv'] = self.__nv
            sc.attrs['nw'] = self.__nw

    def __load_supercell(self, filename):

        with h5py.File(filename, 'r') as f:

            sc = f['disorder/supercell']

            self._Sx = sc['Sx'][...]
            self._Sy = sc['Sy'][...]
            self._Sz = sc['Sz'][...]

            self._A_r = sc['A_r'][...]

            self._Ux = sc['Ux'][...]
            self._Uy = sc['Uy'][...]
            self._Uz = sc['Uz'][...]

            self.__nu = sc.attrs['nu']
            self.__nv = sc.attrs['nv']
            self.__nw = sc.attrs['nw']

    def save(self, filename):
        """
        Save supercell data.

        Parameters
        ----------
        filename : str
            Name of HDF5 file.

        """

        self.__save_supercell(filename)

    def load(self, filename):
        """
        Load supercell data.

        Parameters
        ----------
        filename : str
            Name of HDF5 file.

        """

        self.__load_supercell(filename)

    def reset_disorder(self):
        """
        Reset data arrays.

        """

        self._Sx, self._Sy, self._Sz = [], [], []
        self._Ux, self._Uy, self._Uz = [], [], []
        self._A_r = []

        self.randomize_magnetic_moments()
        self.randomize_site_occupancies()
        self.randomize_atomic_displacements()

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

        self.reset_disorder()

    def get_super_cell_size(self):
        """
        Total number of cells.

        Returns
        -------
        n_uvw : int
            Supercell size :math:`N_1N_2N_3`.

        """

        nu, nv, nw = self.get_super_cell_extents()

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
        nu, nv, nw = self.get_super_cell_extents()

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

    def randomize_magnetic_moments(self):
        """
        Generate random spin vectors.

        """

        dims = self.get_super_cell_extents()
        n_atm = self.get_number_atoms_per_unit_cell()

        mu = self.get_magnetic_moment_magnitude()

        Sx, Sy, Sz = magnetic.spin(*dims, n_atm, mu)

        self._Sx.append(Sx)
        self._Sy.append(Sy)
        self._Sz.append(Sz)

    def randomize_site_occupancies(self):
        """
        Generate random site occupancies.

        """

        dims = self.get_super_cell_extents()
        n_atm = self.get_number_atoms_per_unit_cell()

        occ = self.get_occupancies()

        A_r = occupational.composition(*dims, n_atm, occ)

        self._A_r.append(A_r)

    def randomize_atomic_displacements(self):
        """
        Generate random atomic displacements.

        """

        dims = self.get_super_cell_extents()
        n_atm = self.get_number_atoms_per_unit_cell()

        U = self.get_cartesian_anistropic_displacement_parameters()

        disp = np.row_stack(U)

        Ux, Uy, Uz = displacive.expansion(*dims, n_atm, disp)

        self._Ux.append(Ux)
        self._Uy.append(Uy)
        self._Uz.append(Uz)

    def __statistics(self, data):

        return np.mean(data, axis=0), np.std(data, axis=0)**2

    def spin_correlations_1d(self, fract, tol, average=False):
        """
        Spherically-averaged spin correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        coll : 1d array
            Collinearity.
        d : 1d array
            Separation distance magnitude.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        _corr, _coll = [], []

        for Sx, Sy, Sz in zip(self._Sx, self._Sy, self._Sz):

            args = Sx, Sy, Sz, *coords, atms, *dims, A, fract, tol

            corr, coll, _, _, d, pairs = functions.vector1d(*args)

            _corr.append(corr)
            _coll.append(coll)

        data = *self.__statistics(_corr), *self.__statistics(_coll)

        if average:

            *data, d, pairs = functions.average1d(data, d, pairs, tol)

        return (*data, d, pairs)

    def spin_correlations_3d(self, fract, tol, average=False, laue=None):
        """
        Three-dimensional spin correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        coll : 1d array
            Collinearity.
        dx, dy, dz : 1d array
            Separation distance vector.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        _corr, _coll = [], []

        for Sx, Sy, Sz in zip(self._Sx, self._Sy, self._Sz):

            args = Sx, Sy, Sz, *coords, atms, *dims, A, fract, tol

            corr, coll, _, _, dx, dy, dz, pairs = functions.vector3d(*args)

            _corr.append(corr)
            _coll.append(coll)

        data = *self.__statistics(_corr), *self.__statistics(_coll)

        if laue is not None:

            args = data, dx, dy, dz, pairs, A, laue, tol

            *data, dx, dy, dz, pairs = functions.symmetrize(*args)

        if average:

            args = data, dx, dy, dz, pairs, tol

            *data, dx, dy, dz, pairs = functions.average3d(*args)

        return (*data, dx, dy, dz, pairs)

    def occupancy_correlations_1d(self, fract, tol, average=False):
        """
        Spherically-averaged occupancy correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        d : 1d array
            Separation distance magnitude.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        _corr = []

        for A_r in self._A_r:

            args = A_r, *coords, atms, *dims, A, fract, tol

            corr, _, d, pairs = functions.scalar1d(*args)

            _corr.append(corr)

        data = self.__statistics(_corr)

        if average:

            *data, d, pairs = functions.average1d(data, d, pairs, tol)

        return (*data, d, pairs)

    def occupancy_correlations_3d(self, fract, tol, average=False, laue=None):
        """
        Three-dimensional occupancy correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        dx, dy, dz : 1d array
            Separation distance vector.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        args = self._A_r, *coords, atms, *dims, A, fract, tol

        corr, coll, _, _, dx, dy, dz, pairs = functions.vector3d(*args)

        return corr, coll, dx, dy, dz, pairs

        _corr = []

        for A_r in self._A_r:

            args = A_r, *coords, atms, *dims, A, fract, tol

            corr, _, _, dx, dy, dz, pairs = functions.scalar3d(*args)

            _corr.append(corr)

        data = self.__statistics(_corr)

        if laue is not None:

            args = data, dx, dy, dz, pairs, A, laue, tol

            *data, dx, dy, dz, pairs = functions.symmetrize(*args)

        if average:

            args = data, dx, dy, dz, pairs, tol

            *data, dx, dy, dz, pairs = functions.average1d(*args)

        return (*data, dx, dy, dz, pairs)

    def displacement_correlations_1d(self, fract, tol, average=False):
        """
        Spherically-averaged displacement correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        coll : 1d array
            Collinearity.
        d : 1d array
            Separation distance magnitude.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        _corr, _coll = [], []

        for Ux, Uy, Uz in zip(self._Ux, self._Uy, self._Uz):

            args = Ux, Uy, Uz, *coords, atms, *dims, A, fract, tol

            corr, coll, _, _, d, pairs = functions.vector1d(*args)

            _corr.append(corr)
            _coll.append(coll)

        data = *self.__statistics(_corr), *self.__statistics(_coll)

        if average:

            *data, d, pairs = functions.average1d(data, d, pairs, tol)

        return (*data, d, pairs)

    def displacement_correlations_3d(self, fract, tol, average=False, laue=None):
        """
        Three-dimensional displacement correlations.

        Parameters
        ----------
        fract : float
            Fraction of longest distance for radial cutoff.
        tol : float
            Tolerance of distances for unique pairs.

        Returns
        -------
        corr : 1d array
            Correlation.
        coll : 1d array
            Collinearity.
        dx, dy, dz : 1d array
            Separation distance vector.
        pairs : 1d array, str
            Atom, ion, or isotope-pairs.

        """

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()

        *coords, atms = self.get_super_cell_cartesian_atomic_coordinates()

        _corr, _coll = [], []

        for Ux, Uy, Uz in zip(self._Ux, self._Uy, self._Uz):

            args = Ux, Uy, Uz, *coords, atms, *dims, A, fract, tol

            corr, coll, _, _, dx, dy, dz, pairs = functions.vector3d(*args)

            _corr.append(corr)
            _coll.append(coll)

        data = *self.__statistics(_corr), *self.__statistics(_coll)

        if laue is not None:

            args = data, dx, dy, dz, pairs, A, laue, tol

            *data, dx, dy, dz, pairs = functions.symmetrize(*args)

        if average:

            args = data, dx, dy, dz, pairs, tol

            *data, dx, dy, dz, pairs = functions.average3d(*args)

        return (*data, dx, dy, dz, pairs)

    def magnetic_powder_intensity(self, extents, bins):
        """
        Calculate magnetic powder intensity.

        Parameters
        ----------
        extents : list, float
            Reciprocal space extents.
        bins : int
            Number of bins.

        Returns
        -------
        I : 1d array
            Magnetic scattering intensity.

        """

        Q = np.linspace(*extents, bins)

        dims = self.get_super_cell_extents()

        A = self.get_fractional_cartesian_transform()
        D = self.get_atomic_displacement_cartesian_transform()

        *coords, _ = self.get_super_cell_cartesian_atomic_coordinates()
        ions = self.get_unit_cell_ions()

        occ = self.get_occupancies()
        U = self.get_cartesian_anistropic_displacement_parameters()

        g = self.get_g_factors()

        intensity = []

        for Sx, Sy, Sz in zip(self._Sx, self._Sy, self._Sz):

            args = Sx, Sy, Sz, occ, *U, *coords, ions, Q, A, D, *dims, g

            intensity.append(powder.magnetic(*args))

        return self.__statistics(intensity)

    def magnetic_single_crystal_intensity(self, extents, bins, W, laue=None):
        """
        Calculate magnetic _single_crystal intensity.

        Parameters
        ----------
        extents : list of lists, float
            Reciprocal space extents.
        bins : list, int
            Number of bins.
        W : 2d array
            Projection matrix.
        laue : str, optional
            Laue symmetry.

        Returns
        -------
        I : 1d array
            Magnetic scattering intensity.

        """

        dims = self.get_super_cell_extents()

        B = self.get_miller_cartesian_transform()
        R = self.get_cartesian_rotation()
        D = self.get_atomic_displacement_cartesian_transform()

        T, wgts = self.get_twins()

        coords = self.get_unit_cell_cartesian_atomic_coordinates()
        ions = self.get_unit_cell_ions()

        occ = self.get_occupancies()
        U = self.get_cartesian_anistropic_displacement_parameters()

        g = self.get_g_factors()

        W = np.array(W).astype(float)

        args = *extents, *bins, *dims, W, laue

        indices, inverses, ops, *points = space.reduced(*args)

        symop = symmetry.laue_id(ops)

        trans = *extents, indices, symop, W, B, R, D, T, wgts, *bins

        intensity = []

        for Sx, Sy, Sz in zip(self._Sx, self._Sy, self._Sz):

            spins = Sx, Sy, Sz

            args = *spins, occ, *U, *coords, ions, *trans, *dims, *points, g

            intensity.append(monocrystal.magnetic(*args))

        I, sigma_sq = self.__statistics(intensity)

        return I[inverses].reshape(*bins), sigma_sq[inverses].reshape(*bins)

    def single_crystal_intensity_blur(self, I, sigma):

        return filters.blurring(I, sigma)

    def save_intensity_3d(self, filename, I, extents, bins, W=np.eye(3)):

        B = self.get_miller_cartesian_transform()

        h_range, k_range, l_range = extents
        nh, nk, nl = bins

        h, k, l = np.meshgrid(np.linspace(*h_range,nh),
                              np.linspace(*k_range,nk),
                              np.linspace(*l_range,nl), indexing='ij')


        experimental.intensity(filename, h, k, l, I, np.dot(B,W))

    def save_correlations_3d(self, filename, data, dx, dy, dz, pairs):

        label = 'vector-pair'

        corr, _, coll, _ = data

        outdata = (dx, dy, dz, corr, coll, pairs)

        experimental.correlations(filename, outdata, label)

    def save_data(self, filename, signal, sigma_sq, extents, bins):

        self.__save_supercell(filename)

        with h5py.File(filename, 'a') as f:

            data = f.create_group('disorder/data')

            data.create_dataset('extents', data=extents)
            data.create_dataset('bins', data=bins)

            data.create_dataset('signal', data=signal)
            data.create_dataset('sigma_sq', data=sigma_sq)
