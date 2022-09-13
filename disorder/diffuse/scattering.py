#!/usr/bin/env python3

import re

import numpy as np

from disorder.diffuse import space, filters, magnetic
from disorder.diffuse import interaction, simulation
from disorder.diffuse import experimental, refinement
from disorder.material import crystal, tables

def length(atms, n_hkl):
    """
    Scattering length :math:`b` of neutrons.

    Parameters
    ----------
    atms : 1d array, str
        Atoms or isotopes.
    n_hkl : int
        Number of reciprocal space points.

    Returns
    -------
    b : 1d array
        Has the shape as the number of reciprocal space points.

    """
    n_atm = len(atms)

    b = np.zeros(n_hkl*n_atm, dtype=complex)

    for i, atm in enumerate(atms):

        bc = tables.bc.get(atm)

        for i_hkl in range(n_hkl):

            b[i::n_atm] = bc

    return b

def form(ions, Q, source='x-ray'):
    """
    Scattering form factor :math:`f(Q)`.

    Parameters
    ----------
    ions : 1d array, str
        Ions.
    Q : 1d array
        Magnitude of wavevector.
    source : str, optional
       Radiation source. Either ``'x-ray'`` or ``'electron'``.
       Defualt is ``source='x-ray'``.

    Returns
    -------
    f : 1d array
        Has the same shape as the input wavevector.

    """

    n_hkl = Q.shape[0]
    n_atm = len(ions)

    factor = np.zeros(n_hkl*n_atm)

    s = Q/(4*np.pi)

    for i, ion in enumerate(ions):

        if (source == 'x-ray'):

            a1, b1, a2, b2, a3, b3, a4, b4, c = tables.X.get(ion)

            factor[i::n_atm] = a1*np.exp(-b1*s**2)\
                             + a2*np.exp(-b2*s**2)\
                             + a3*np.exp(-b3*s**2)\
                             + a4*np.exp(-b4*s**2)\
                             + c

        else:

            a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = tables.E.get(ion)

            factor[i::n_atm] = a1*np.exp(-b1*s**2)\
                             + a2*np.exp(-b2*s**2)\
                             + a3*np.exp(-b3*s**2)\
                             + a4*np.exp(-b4*s**2)\
                             + a5*np.exp(-b5*s**2)

    factor[factor < 0] = 0

    if (source == 'electron'):

        for i, ion in enumerate(ions):

            delta_Z = re.sub(r'[a-zA-Z]', '', ion)[::-1]
            delta_Z = int(delta_Z) if delta_Z != '' else 0
            factor[i::n_atm] += 0.023934*delta_Z/s**2

    return factor

def phase(Qx, Qy, Qz, rx, ry, rz):
    """
    Phase factor :math:`e^{i\\boldsymbol{Q}\cdot\\boldsymbol{r}}`.

    Parameters
    ----------
    rx, ry, rz : 1d array
        Components of spatial vector in Cartesian coordinates.
    Qx, Qy, Qz : 1d array
        Components of wavevector in Cartesian coordinates.

    Returns
    -------
    factor : 1d array
        Has the same shape as the input wavevector and spatial vector
        components.

    """

    Q_dot_r = Qx[:,np.newaxis]*rx+Qy[:,np.newaxis]*ry+Qz[:,np.newaxis]*rz

    factor = np.exp(1j*Q_dot_r)

    return factor.flatten()

class Simulation:
    """
    Simulation.

    Parameters
    ----------
    sc : supercell
        Supercell for simulation.
    extend : bool, optional
        Extend beyond one unit cell for neighbor bonds. Defualt is ``False``.

    """

    def __init__(self, sc, extend=False):

        self.sc = sc

        dims = sc.get_super_cell_extents()
        n_atm = sc.get_number_atoms_per_unit_cell()

        *coords, atms = sc.get_super_cell_cartesian_atomic_coordinates()

        A = sc.get_fractional_cartesian_transform()
        B = sc.get_miller_cartesian_transform()
        R = sc.get_cartesian_rotation()

        args = *coords, *dims, n_atm, A, B, R

        self.__Qij = interaction.charge_charge_matrix(*args)
        self.__Qijk = interaction.charge_dipole_matrix(*args)
        self.__Qijkl = interaction.dipole_dipole_matrix(*args)

        self.__const_cc = 0.0
        self.__const_cd = 0.0
        self.__const_dd = 0.0

        coords = sc.get_fractional_coordinates()
        atms = sc.get_unit_cell_atoms()

        pair_info = interaction.pairs(*coords, atms, A, extend=extend)
        bond_info = interaction.bonds(pair_info, *coords, A)

        dx, dy, dz, img_i, img_j, img_k, *indices = bond_info

        self.__dx, self.__dy, self.__dz = dx, dy, dz

        self.__img_i, self.__img_j, self.__img_k = img_i, img_j, img_k

        atm_ind, pair_inv, pair_ind, pair_trans = indices

        self.__atm_ind, self.__pair_ind = atm_ind, pair_ind
        self.__pair_inv, self.__pair_trans = pair_inv, pair_trans

        uxx, uyy, uzz, uyz, uxz, uxy = interaction.anisotropy(dx, dy, dz)

        n_pair = 1+np.max(pair_ind)

        self.__J = np.zeros((n_pair,3,3))
        self.__K = np.zeros((n_atm,3,3))
        self.__B = np.zeros(3)

        g = sc.get_g_factors()

        self.__g = np.einsum('i,jk->ijk', g, np.eye(3))

        self.__active = np.ones(n_pair, dtype=bool)

        self.__n_atm = n_atm

    def __mask(self):

        indices = np.arange(self.__active.size)[self.__active]

        return (self.__pair_ind[...,np.newaxis] == indices).any(axis=2)

    def ___get_indices(self):

        mask = self.__mask()

        atm_ind = self.__atm_ind[mask].reshape(self.__n_atm,-1)
        img_i = self.__img_i[mask].reshape(self.__n_atm,-1)
        img_j = self.__img_j[mask].reshape(self.__n_atm,-1)
        img_k = self.__img_k[mask].reshape(self.__n_atm,-1)
        pair_ind = self.__pair_ind[mask].reshape(self.__n_atm,-1)
        pair_trans = self.__pair_trans[mask].reshape(self.__n_atm,-1)

        return atm_ind, img_i, img_j, img_k, pair_ind, pair_trans

    def get_active_bonds(self):
        """
        Active bonds.

        Returns
        -------
        active : 1d array, bool
            The bonds that are active.

        """

        return self.__active

    def set_active_bonds(self, active):
        """
        Update active bonds.

        Parameters
        ----------
        active : 1d array, bool
            The bonds that are active.

        """

        self.__active = active

    def get_bond_lengths(self):
        """
        Bond lengths.

        Returns
        -------
        d : 1d array
            The length of the bonds.

        """

        mask = self.__mask()

        dx, dy, dz = self.__dx, self.__dy, self.__dz

        return np.sqrt(dx**2+dy**2+dz**2)[mask].reshape(self.__n_atm,-1)

    def get_bond_vectors(self):
        """
        Bond vectors.

        Returns
        -------
        dx, dy, dz : 1d array
            The bond separation vectors.

        """

        mask = self.__mask()

        dx, dy, dz = self.__dx, self.__dy, self.__dz

        return dx[mask].reshape(self.__n_atm,-1),\
               dy[mask].reshape(self.__n_atm,-1),\
               dz[mask].reshape(self.__n_atm,-1)

    def get_charge_charge_matrix(self):
        """
        Charge-charge interaction matrix.

        Returns
        -------
        Qij : 2d array
            Matrix for calculating charge-charge interactions between charges.

        """

        return self.__Qij

    def get_charge_dipole_matrix(self):
        """
        Charge-dipole interaction matrix.

        Returns
        -------
        Qijk : 3d array
            Matrix for calculating charge-dipole interactions between charges
            and moments.

        """

        return self.__Qijk

    def get_dipole_dipole_matrix(self):
        """
        Dipole-dipole interaction matrix.

        Returns
        -------
        Qijkl : 4d array
            Matrix for calculating dipole-dipole interactions between moments.

        """

        return self.__Qijkl

    def get_magnetic_exchange_interaction_matrices(self):
        """
        Magnetic exchange interaction matrices.

        Returns
        -------
        J : 3d array
            Interaction matrices.

        """

        return self.__J[self.__active,...]

    def set_magnetic_exchange_interaction_matrices(self, J):
        """
        Update magnetic exchange interaction matrices.

        Parameters
        ----------
        J : 3d array
            Interaction matrices.

        """

        self.__J[self.__active,...] = J

    def get_magnetic_single_ion_anisotropy_matrices(self):
        """
        Magnetocrystalline anisotropy matrices.

        Returns
        -------
        K : 3d array
            Anisotropy matrices.

        """

        return self.__K

    def set_magnetic_single_ion_anisotropy_matrices(self, K):
        """
        Update magnetocrystalline anisotropy matrices.

        Parameters
        ----------
        K : 3d array
            Anisotropy matrices.

        """

        self.__K = K

    def get_magnetic_g_tensor_matrices(self):
        """
        g-tensor matrices.

        Returns
        -------
        g : 3d array
            g-tensors.

        """

        return self.__g

    def set_magnetic_g_tensor_matrices(self, g):
        """
        Update g-tensor matrices.

        Parameters
        ----------
        g : 3d array
            g-tensors.

        """

        self.__g

    def get_magnetic_field(self):
        """
        External magnetic field vector.

        Returns
        -------
        B : 1d array
            Fied vector.

        """

        return self.__B

    def set_magnetic_field(self, B):
        """
        Update external magnetic field vector.

        Parameters
        ----------
        B : 1d array
            Fied vector.

        """

        self.__B = B

    def get_easy_axes_matrices(self):
        """
        Easy axes matrices.

        Returns
        -------
        U : 3d array
            Axes matrices.

        """

        dx, dy, dz = self.get_bond_vectors()

        uxx, uyy, uzz, uyz, uxz, uxy = interaction.anisotropy(dx, dy, dz)

        U = np.zeros((self.__n_atm,3,3))

        U[:,0,0], U[:,1,1], U[:,2,2] = uxx, uyy, uzz

        U[:,0,1] = U[:,1,0] = uxy
        U[:,0,2] = U[:,2,0] = uxz
        U[:,1,2] = U[:,2,1] = uyz

        return U

    def get_magnetic_dipole_dipole_coupling_strength(self):
        """
        Magnetic dipole-dipole coupling strength.

        Returns
        -------
        const : float
            Strength of dipole-dipole interaction.

        """

        return self.__const_dd

    def set_magnetic_dipole_dipole_coupling_strength(self, const):
        """
        Update magnetic dipole-dipole coupling strength.

        Parameters
        ----------
        const : float
            Strength of dipole-dipole interaction.

        """

        self.__const_dd = const

    def initialize_parallel_tempering(self, T0, T1, replicas=1, space='log2'):
        """
        Initialize parallel tempering simulation.

        Parameters
        ----------
        T0 : float
            Start temperature.
        T1 : float
            End temperature.
        replicas : int, optional
            Number of replica simulations. The default is ``1``.
        space : str, optional
            Temperature spacing among ``'log2'`, ``'log10'`, and ``'linear'`.
            The default is ``'log2'``.

        """

        if space == 'log2':
            T = np.logspace(np.log2(T0), np.log2(T1), replicas, base=2)
        elif space == 'log10':
            T = np.logspace(np.log10(T0), np.log10(T1), replicas)
        else:
            T = np.linspace(T0, T1, replicas)

        self.__T = T

        dims = self.sc.get_super_cell_extents()
        n_atm = self.sc.get_number_atoms_per_unit_cell()

        self.__Sx = np.zeros((*dims,n_atm,replicas))
        self.__Sy = np.zeros((*dims,n_atm,replicas))
        self.__Sz = np.zeros((*dims,n_atm,replicas))

        self.__Sx.T[:,...] = self.sc._Sx.reshape(*dims,n_atm).T.copy()
        self.__Sy.T[:,...] = self.sc._Sy.reshape(*dims,n_atm).T.copy()
        self.__Sz.T[:,...] = self.sc._Sz.reshape(*dims,n_atm).T.copy()

    def magnetic_energy(self):
        """
        Magnetic interaction energy.

        Returns
        -------
        E : 6d array
            Magnetic interaction energies.

        """

        spins = self.__Sx, self.__Sy, self.__Sz

        properties = self.__J[self.__active,...], self.__K, self.__g

        field = self.__B

        indices = self.___get_indices()

        args = *spins, *properties, field, *indices

        return simulation.magnetic_energy(*args)

    def magnetic_dipole_dipole_interaction_energy(self):
        """
        Magnetic dipole-dipole interaction energy.

        Returns
        -------
        E : 4d array
            Magnetic dipole-dipole interaction energies.

        """

        args = self.__Sx, self.__Sy, self.__Sz, self.__const_dd*self.__Qijkl

        return simulation.dipole_dipole_interaction_energy(*args)

    def magnetic_simulation(self, N):
        """
        Perform magnetic Heisenberg simulation.

        Parameters
        ----------
        N : int
            Number of Monte Carlo cycles.

        Returns
        -------
        H : 1d array
            Hamiltonian of each replica.
        T : 1d array
            Temperature of each replica.

        """

        kB = 0.08617 # meV/K

        spins = self.__Sx, self.__Sy, self.__Sz

        properties = self.__J[self.__active,...], self.__K, self.__g

        fields = self.__B, self.__Qijkl*self.__const_dd

        indices = self.___get_indices()

        args = *spins, *properties, *fields, *indices, self.__T, kB, N

        H, T = simulation.heisenberg(*args)

        self.__T = T

        ind = np.argmin(H)

        self.sc._Sx = self.__Sx[...,ind].flatten()
        self.sc._Sy = self.__Sy[...,ind].flatten()
        self.sc._Sz = self.__Sz[...,ind].flatten()

        return H, T

class Refinement:
    """
    Refinement.

    Parameters
    ----------
    sc : supercell
        Supercell for refinement.

    """

    def __init__(self, sc):

        self.sc = sc

    def load_intensity(self, filename):

        self.signal, self.sigma_sq, *binning = experimental.data(filename)

        self.extents, self.bins = binning[0:3], binning[3:6]

        self.reset_intensity()

    def reset_intensity(self):

        self.__signal = self.signal.copy()
        self.__sigma_sq = self.sigma_sq.copy()

        self.__extents = self.extents
        self.__bins = self.bins

    def crop(self, slices):

        self.__signal = experimental.crop(self.__signal, *slices)
        self.__sigma_sq = experimental.crop(self.__sigma_sq, *slices)

    def rebin(self, bins):

        self.__signal = experimental.rebin(self.__signal, *bins)
        self.__sigma_sq = experimental.rebin(self.__sigma_sq, *bins)

    def punch(self, radii, centering='P', outlier=1.5, ptype='cuboid'):

        params = *radii, self.__extents, centering, outlier, ptype

        self.__signal = experimental.punch(self.__signal, *params)
        self.__sigma_sq = experimental.punch(self.__sigma_sq, *params)

    def __mask(self):

        return experimental.mask(self.__signal, self.__sigma_sq)

    def __mask_indices(self):

        return space.indices(self.__mask())

    def initialize_refinement(self, temp, const):

        self.__acc_moves, self.__acc_temps = [], []
        self.__rej_moves, self.__rej_temps = [], []

        self.__energy, self.__scale = [], []

        self.__chi_sq, self.__temperature =  [np.inf], [temp]
        self.__constant = const

        self.__A_r = self.sc._A_r
        self.__Ux, self.__Uy, self.__Uz = self.sc._Ux, self.sc._Uy, self.sc._Uz
        self.__Sx, self.__Sy, self.__Sz = self.sc._Sx, self.sc._Sy, self.sc._Sz

        dims = self.sc.get_super_cell_extents()

        rx, ry, rz, _ = self.sc.get_super_cell_cartesian_atomic_coordinates()

        extents = self.__extents
        bins = self.__bins

        mapping = space.mapping(*extents, *bins, *dims)

        h, k, l, self.__H, self.__K, self.__L, \
        self.__indices, self.__inverses, symops = mapping

        B = self.sc.get_miller_cartesian_transform()
        R = self.sc.get_cartesian_rotation()

        Qh, Qk, Ql = crystal.vector(h, k, l, B)

        Qx, Qy, Qz = crystal.transform(Qh, Qk, Ql, R)

        Qx_norm, Qy_norm, Qz_norm, Q = space.unit(Qx, Qy, Qz)

        self.__Qx_norm = Qx_norm
        self.__Qy_norm = Qy_norm
        self.__Qz_norm = Qz_norm
        self.__Q = Q

        ux, uy, uz = self.sc.get_unit_cell_cartesian_atomic_coordinates()

        phase_factor = phase(Qx, Qy, Qz, ux, uy, uz)

        self.__space_factor = space.factor(*dims)

        isotopes = self.sc.get_unit_cell_isotopes()
        ions = self.sc.get_unit_cell_ions()

        occ = self.sc.get_occupancies()
        U = self.sc.get_cartesian_anistropic_displacement_parameters()

        g = self.sc.get_g_factors()

        a_, b_, c_, *_ = self.sc.get_reciprocal_lattice_constants()

        self.__T = space.debye_waller(*extents, *bins, *U, a_, b_, c_)

        form_factor = magnetic.form(Q, ions, g)

        self.__mag_factors = space.prefactors(form_factor, phase_factor, occ)

        scattering_length = length(isotopes, Q.size)

        self.__factors = space.prefactors(scattering_length, phase_factor, occ)

    def __initialize_magnetic(self):

        dims = self.sc.get_super_cell_extents()
        n_atm = self.sc.get_number_atoms_per_unit_cell()

        spins = self.__Sx, self.__Sy, self.__Sz

        points = self.__H, self.__K, self.__L

        args = *spins, *points, *dims, n_atm

        transform = magnetic.transform(*args)

        dirs = self.__Qx_norm, self.__Qy_norm, self.__Qz_norm

        structure = magnetic.structure(*dirs, *transform, self.__factors)

        self.__Sx_k, self.__Sy_k, self.__Sz_k, self.__i_dft = transform

        self.__Fx, self.__Fy, self.__Fz, \
        self.__prod_x, self.__prod_y, self.__prod_z = structure

        n_uvw = self.sc.get_super_cell_size()

        self.__Fx_orig = np.zeros(self.__indices.size, dtype=complex)
        self.__Fy_orig = np.zeros(self.__indices.size, dtype=complex)
        self.__Fz_orig = np.zeros(self.__indices.size, dtype=complex)

        self.__prod_x_orig = np.zeros(self.__indices.size, dtype=complex)
        self.__prod_y_orig = np.zeros(self.__indices.size, dtype=complex)
        self.__prod_z_orig = np.zeros(self.__indices.size, dtype=complex)

        self.__Sx_k_orig = np.zeros(n_uvw, dtype=complex)
        self.__Sy_k_orig = np.zeros(n_uvw, dtype=complex)
        self.__Sz_k_orig = np.zeros(n_uvw, dtype=complex)

        self.__Fx_cand = np.zeros(self.__indices.size, dtype=complex)
        self.__Fy_cand = np.zeros(self.__indices.size, dtype=complex)
        self.__Fz_cand = np.zeros(self.__indices.size, dtype=complex)

        self.__prod_x_cand = np.zeros(self.__indices.size, dtype=complex)
        self.__prod_y_cand = np.zeros(self.__indices.size, dtype=complex)
        self.__prod_z_cand = np.zeros(self.__indices.size, dtype=complex)

        self.__Sx_k_cand = np.zeros(n_uvw, dtype=complex)
        self.__Sy_k_cand = np.zeros(n_uvw, dtype=complex)
        self.__Sz_k_cand = np.zeros(n_uvw, dtype=complex)

    def __intensities(self):

        mask = self.__mask()

        I_obs = np.full(mask.shape, np.nan)
        I_ref = I_obs[~mask]

        I_calc = np.zeros(self.__Q.size, dtype=float)

        I_raw = np.zeros(mask.size, dtype=float)
        I_flat = np.zeros(mask.size, dtype=float)

        I_expt, inv_sigma_sq = self.__signal[~mask], 1/self.__sigma_sq[~mask]

        return I_calc, I_expt, inv_sigma_sq, I_raw, I_flat, I_ref

    def __filters(self, sigma):

        mask = self.__mask()

        filt = np.zeros((9,mask.size), dtype=float)

        v_inv = filters.gaussian(mask, sigma)

        boxes = filters.boxblur(sigma, 3)

        return v_inv, *filt, boxes

    def magnetic_refinement(self, N, sigma):

        fixed, heisenberg = True, True

        mu = self.sc.get_magnetic_moment_magnitude()

        self.__initialize_magnetic()

        dims = self.sc.get_super_cell_extents()
        n_atm = self.sc.get_number_atoms_per_unit_cell()

        n = self.sc.get_number_atoms_per_unit_cell()

        bins = self.__bins

        spins = self.__Sx, self.__Sy, self.__Sz

        dirs = self.__Qx_norm, self.__Qy_norm, self.__Qz_norm

        trans = self.__Sx_k, self.__Sy_k, self.__Sz_k

        trans_orig = self.__Sx_k_orig, self.__Sy_k_orig, self.__Sz_k_orig
        trans_cand = self.__Sx_k_cand, self.__Sy_k_cand, self.__Sz_k_cand

        struct = self.__Fx, self.__Fy, self.__Fz

        struct_orig = self.__Fx_orig, self.__Fy_orig, self.__Fz_orig
        struct_cand = self.__Fx_cand, self.__Fy_cand, self.__Fz_cand

        prod = self.__prod_x, self.__prod_y, self.__prod_z

        prod_orig = self.__prod_x_orig, self.__prod_y_orig, self.__prod_z_orig
        prod_cand = self.__prod_x_cand, self.__prod_y_cand, self.__prod_z_cand

        indices = self.__i_dft, self.__inverses, *self.__mask_indices()

        accepted = self.__acc_moves, self.__acc_temps
        rejected = self.__rej_moves, self.__rej_temps
        statistics = self.__chi_sq, self.__energy
        optimization = self.__temperature, self.__scale, self.__constant

        factors = self.__space_factor, self.__mag_factors

        intensities = self.__intensities()
        filters = self.__filters(sigma)

        opts = fixed, heisenberg

        args = *spins, *dirs, *trans, *trans_orig, *trans_cand, *struct, \
               *struct_orig, *struct_cand, *prod, *prod_orig, *prod_cand, \
               *factors, mu, *intensities, *filters, *indices, \
               *accepted, *rejected, *statistics, *optimization, \
               *opts, *bins, *dims, n_atm, n, N

        refinement.magnetic(*args)

        self.sc._Sx, self.sc._Sy, self.sc._Sz = self.__Sx, self.__Sy, self.__Sz
