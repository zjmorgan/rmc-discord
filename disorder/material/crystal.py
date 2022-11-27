#!/usr/bin/env python3

import CifFile
import numpy as np

import sys, os, io, re

from disorder.material import symmetry
from disorder.material import tables

def unitcell(folder, filename, tol=1e-2):
    """
    Reads atom site information from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.
    tol : float, optional
        Tolerance of unique atom coordinates.

    Returns
    -------
    unit_cell : dict
        Dictionary of unit cell information.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_ops = ['_space_group_symop_operation_xyz',
                '_symmetry_equiv_pos_as_xyz',
                '_space_group_symop_magn_operation_xyz']

    ind_ops = next((i for i, loop_key in enumerate(loop_ops) \
                    if loop_key in cif_dict), None)

    symops = cif_dict[loop_ops[ind_ops]]

    parity = False if len(symops[0].split(',')) == 3 else True

    if ind_ops == 2:
        add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
        combine = []
        for symop in symops:
            for add_symop in add_symops:
                left_op = ','.join(symop.split(',')[:3])
                right_op = ','.join(add_symop.split(',')[:3])
                combine_op = symmetry.binary([left_op],[right_op])
                parity = int(symop.split(',')[-1])*int(add_symop.split(',')[-1])
                combine += [combine_op[0]+','+str(parity)]
        symops = combine

    atomic_sites = cif_dict['_atom_site_label']


    if '_atom_site_moment_label' in cif_dict:
        magnetic_sites = cif_dict['_atom_site_moment_label']
        magnetic_atoms = [asite in magnetic_sites \
                          for asite in atomic_sites]

    if parity:
        parities = [int(symop.split(',')[-1]) for symop in symops]
    else:
        parities = [1 for symop in symops]

    symops = [','.join(symop.split(',')[:3]) for symop in symops]

    atomic_sites = [re.sub(r'[0-9]', '', asite) for asite in atomic_sites]

    if '_atom_site_type_symbol' in cif_dict:
        symbols = cif_dict['_atom_site_type_symbol']
    elif '_atom_site_symbol' in cif_dict:
        symbols = cif_dict['_atom_site_symbol']
    else:
        symbols = atomic_sites

    symbols = [sym if sym != 'D' else '2H' for sym in symbols]

    xs = cif_dict['_atom_site_fract_x']
    ys = cif_dict['_atom_site_fract_y']
    zs = cif_dict['_atom_site_fract_z']

    xs = [re.sub(r'\([^()]*\)', '', x) for x in xs]
    ys = [re.sub(r'\([^()]*\)', '', y) for y in ys]
    zs = [re.sub(r'\([^()]*\)', '', z) for z in zs]

    if '_atom_site_occupancy' in cif_dict:
        occs = cif_dict['_atom_site_occupancy']
        occs = [re.sub(r'\([^()]*\)', '', occ) for occ in occs]
    else:
        occs = [1.0]*len(atomic_sites)

    iso = False
    ani = False

    if '_atom_site_aniso_label' in cif_dict:
        adp_type, ani = 'ani', True
        if '_atom_site_aniso_u_11' in cif_dict:
            adp_U = True
            U11s = cif_dict['_atom_site_aniso_u_11']
            U22s = cif_dict['_atom_site_aniso_u_22']
            U33s = cif_dict['_atom_site_aniso_u_33']
            U23s = cif_dict['_atom_site_aniso_u_23']
            U13s = cif_dict['_atom_site_aniso_u_13']
            U12s = cif_dict['_atom_site_aniso_u_12']
            U11s = [re.sub(r'\([^()]*\)', '', U11) for U11 in U11s]
            U22s = [re.sub(r'\([^()]*\)', '', U22) for U22 in U22s]
            U33s = [re.sub(r'\([^()]*\)', '', U33) for U33 in U33s]
            U23s = [re.sub(r'\([^()]*\)', '', U23) for U23 in U23s]
            U13s = [re.sub(r'\([^()]*\)', '', U13) for U13 in U13s]
            U12s = [re.sub(r'\([^()]*\)', '', U12) for U12 in U12s]
        else:
            adp_U = False
            B11s = cif_dict['_atom_site_aniso_b_11']
            B22s = cif_dict['_atom_site_aniso_b_22']
            B33s = cif_dict['_atom_site_aniso_b_33']
            B23s = cif_dict['_atom_site_aniso_b_23']
            B13s = cif_dict['_atom_site_aniso_b_13']
            B12s = cif_dict['_atom_site_aniso_b_12']
            B11s = [re.sub(r'\([^()]*\)', '', B11) for B11 in B11s]
            B22s = [re.sub(r'\([^()]*\)', '', B22) for B22 in B22s]
            B33s = [re.sub(r'\([^()]*\)', '', B33) for B33 in B33s]
            B23s = [re.sub(r'\([^()]*\)', '', B23) for B23 in B23s]
            B13s = [re.sub(r'\([^()]*\)', '', B13) for B13 in B13s]
            B12s = [re.sub(r'\([^()]*\)', '', B12) for B12 in B12s]

    if '_atom_site_u_iso_or_equiv' in cif_dict:
        adp_type, iso = 'iso', True
        adp_U = True
        Uisos = cif_dict['_atom_site_u_iso_or_equiv']
        Uisos = [re.sub(r'\([^()]*\)', '', Uiso) for Uiso in Uisos]
    elif '_atom_site_b_iso_or_equiv' in cif_dict:
        adp_type, iso = 'iso', True
        adp_U = False
        Bisos = cif_dict['_atom_site_b_iso_or_equiv']
        Bisos = [re.sub(r'\([^()]*\)', '', Biso) for Biso in Bisos]
    elif ani == False:
        adp_type, iso = 'iso', True
        adp_U = True
        Uisos = [0.0]*len(atomic_sites)

    if ani and iso:
        adp_type = 'ani'
        D = cartesian_displacement(*parameters(folder, filename))
        iso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
        iso_labels = cif_dict['_atom_site_label']
        ani_labels = cif_dict['_atom_site_aniso_label']
        adp_types = cif_dict['_atom_site_adp_type']
        U11, U22, U33, U23, U13, U12 = [], [], [], [], [], []
        B11, B22, B33, B23, B13, B12 = [], [], [], [], [], []
        for adp, label in zip(adp_types, iso_labels):
            if adp.lower() == 'uani':
                index = ani_labels.index(label)
                U11.append(U11s[index])
                U22.append(U22s[index])
                U33.append(U33s[index])
                U23.append(U23s[index])
                U13.append(U13s[index])
                U12.append(U12s[index])
            elif adp.lower() == 'bani':
                index = ani_labels.index(label)
                B11.append(B11s[index])
                B22.append(B22s[index])
                B33.append(B33s[index])
                B23.append(B23s[index])
                B13.append(B13s[index])
                B12.append(B12s[index])
            elif adp.lower() == 'uiso':
                index = iso_labels.index(label)
                Uani = iso*float(Uisos[index])
                U11.append(str(Uani[0,0]))
                U22.append(str(Uani[1,1]))
                U33.append(str(Uani[2,2]))
                U23.append(str(Uani[1,2]))
                U13.append(str(Uani[0,2]))
                U12.append(str(Uani[0,1]))
            elif adp.lower() == 'biso':
                index = iso_labels.index(label)
                Bani = iso*float(Bisos[index])
                B11.append(str(Bani[0,0]))
                B22.append(str(Bani[1,1]))
                B33.append(str(Bani[2,2]))
                B23.append(str(Bani[1,2]))
                B13.append(str(Bani[0,2]))
                B12.append(str(Bani[0,1]))
        U11s, U22s, U33s, U23s, U13s, U12s = U11, U22, U33, U23, U13, U12
        B11s, B22s, B33s, B23s, B13s, B12s = B11, B22, B33, B23, B13, B12

    Mxs = [0.0]*len(atomic_sites)
    Mys = [0.0]*len(atomic_sites)
    Mzs = [0.0]*len(atomic_sites)

    mag_symops = ['0,0,0']*len(atomic_sites)

    if '_atom_site_moment_label' in cif_dict:
        mxs = cif_dict['_atom_site_moment_crystalaxis_x']
        mys = cif_dict['_atom_site_moment_crystalaxis_y']
        mzs = cif_dict['_atom_site_moment_crystalaxis_z']
        symmform = cif_dict.get('_atom_site_moment_symmform')
        if symmform is None:
           symmform = ['mx,my,mz']*len(mzs)
        mxs = [re.sub(r'\([^()]*\)', '', mx) for mx in mxs]
        mys = [re.sub(r'\([^()]*\)', '', my) for my in mys]
        mzs = [re.sub(r'\([^()]*\)', '', mz) for mz in mzs]
        if len(magnetic_sites) != len(atomic_sites):
            j = 0
            for i, mag in enumerate(magnetic_atoms):
                if mag:
                    Mxs[i], Mys[i], Mzs[i] = mxs[j], mys[j], mzs[j]
                    mag_symops[i] = symmform[j]
                    j += 1
        else:
            Mxs, Mys, Mzs = mxs, mys, mzs
            mag_symops = symmform

    c, d, m, s = [], [], [], []

    total, types, operators, mag_operators = [], [], [], []

    cf = 8*np.pi**2

    for i, asite in enumerate(atomic_sites):

        x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        Mx, My, Mz = float(Mxs[i]), float(Mys[i]), float(Mzs[i])

        occ = float(occs[i])

        if adp_type == 'ani':
            if adp_U is False:
                B11 = float(B11s[i])
                B22 = float(B22s[i])
                B33 = float(B33s[i])
                B23 = float(B23s[i])
                B13 = float(B13s[i])
                B12 = float(B12s[i])
                U11, U22, U33 = B11/cf, B22/cf, B33/cf
                U23, U13, U12 = B23/cf, B13/cf, B12/cf
            else:
                U11 = float(U11s[i])
                U22 = float(U22s[i])
                U33 = float(U33s[i])
                U23 = float(U23s[i])
                U13 = float(U13s[i])
                U12 = float(U12s[i])

        else:
            if adp_U is False:
                Biso = float(Bisos[i])
                Uiso = Biso/(8*np.pi**2)
            else:
                Uiso = float(Uisos[i])

        symbol = symbols[i]
        mag_symop = mag_symops[i]

        for symop, parity in zip(symops, parities):

            transformed = symmetry.evaluate([symop], [x,y,z]).flatten()

            mag_op = symmetry.generate_mag([symop], mag_symop, parity)

            mom = symmetry.evaluate_mag([mag_op], [Mx,My,Mz]).flatten()

            if adp_type == 'ani':
                disp = symmetry.evaluate_disp([symop], [U11,U22,U33,
                                                        U23,U13,U12])
            else:
                disp = [Uiso]

            transformed = [tf+(tf < 0)-(tf >= 1) for tf in transformed]

            total.append(transformed)
            types.append(symbol)
            operators.append(symop)
            mag_operators.append(mag_op)

            c.append(occ)
            d.append(disp)
            m.append(mom)
            s.append(i)

    total = np.array(total)
    types = np.array(types)
    operators = np.array(operators)
    mag_operators = np.array(mag_operators)

    c = np.array(c)
    d = np.array(d)
    m = np.array(m)
    s = np.array(s)

    metric = np.round(np.round(total/tol, 1)).astype(int)

    _, labels = np.unique(types, return_inverse=True)

    metric = np.column_stack((metric, labels))

    symmetries, indices = np.unique(metric, axis=0, return_index=True)

    total = total[indices]

    u, v, w = total[:,0], total[:,1], total[:,2]

    n_atm = symmetries.shape[0]
    atm = types[indices]

    u = np.round(u, 4)
    v = np.round(v, 4)
    w = np.round(w, 4)

    c = c[indices]
    d = d[indices]
    m = m[indices]
    s = s[indices]

    ops = operators[indices]
    mag_ops = mag_operators[indices]

    sort = np.lexsort(np.column_stack((u, v, w, s)).T)

    u = u[sort]
    v = v[sort]
    w = w[sort]

    c = c[sort]
    d = d[sort]
    m = m[sort]
    s = s[sort]

    ops = ops[sort]
    mag_ops = mag_ops[sort]

    atm = atm[sort]

    ion = np.array([''.join(re.findall(r'[\d.+-]+$', a)) for a in atm])
    iso = np.array([''.join(re.findall(r'^\d+\s*', a)) for a in atm])
    atm = np.array([''.join(re.findall(r'[a-zA-Z]', a)) for a in atm])

    unit_cell = {}

    unit_cell['u'] = u
    unit_cell['v'] = v
    unit_cell['w'] = w
    unit_cell['occupancy'] = c
    unit_cell['displacement'] = d
    unit_cell['moment'] = m
    unit_cell['site'] = s
    unit_cell['operator'] = ops
    unit_cell['magnetic_operator'] = mag_ops
    unit_cell['isotope'] = iso
    unit_cell['ion'] = ion
    unit_cell['atom'] = atm
    unit_cell['n_atom'] = n_atm

    return unit_cell

def supercell(atm, occ, disp, mom, u, v, w, nu, nv, nw,
              name, folder, filename):
    """
    Write average supercell information to a CIF file.

    Parameters
    ----------
    atm : 1d array, str
        Atoms, ions, or isotopes.
    occ : 1d array
        Site occupancies.
    disp : 1d array
        Atomic displacements.
    mom : 1d array
        Magnetic moments.
    u, v, w : 1d array
        Fractional coordinates.
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell.
    folder : str,
        Name of path excluding filename of reference CIF file.
    filename : str
        Name of filename excluding path of reference CIF file.
    tol : float, optional
        Tolerance of unique atom coordinates.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_ops = ['_space_group_symop_operation_xyz',
                '_symmetry_equiv_pos_as_xyz',
                '_space_group_symop_magn_operation_xyz']

    ind_ops = next((i for i, loop_key in enumerate(loop_ops) \
                    if loop_key in cif_dict), None)

    symops = cif_dict[loop_ops[ind_ops]]

    symops = [re.sub(r'[+/][0-9]', '', symop) for symop in symops]

    for symop in symops:
        cb.RemoveLoopItem(symop)

    symops = [re.sub(r'[+/][0-9]', '', symop) for symop in symops]

    atomic_sites = cb.GetLoopNames('_atom_site_label')

    for asite in atomic_sites:
        cb.RemoveLoopItem(asite)

    #n_atm = atm.shape[0]

    a = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_a']))
    b = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_b']))
    c = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_c']))

    cb['_cell_length_a'] = str(a*nu)
    cb['_cell_length_b'] = str(b*nv)
    cb['_cell_length_c'] = str(c*nw)

    alpha = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_alpha']))
    beta = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_beta']))
    gamma = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_gamma']))

    V = volume(a,
               b,
               c,
               np.deg2rad(alpha),
               np.deg2rad(beta),
               np.deg2rad(gamma))

    cb['_cell_volume'] = str(V)

    i, j, k = np.meshgrid(np.arange(nu),
                          np.arange(nv),
                          np.arange(nw), indexing='ij')

    i = i.flatten()
    j = j.flatten()
    k = k.flatten()

    U = (i[:,np.newaxis]+u).flatten()
    V = (j[:,np.newaxis]+v).flatten()
    W = (k[:,np.newaxis]+w).flatten()

    n_uvw = nu*nv*nw

    atom = np.tile(atm, n_uvw)
    occupancy = np.tile(occ, n_uvw)

    if np.shape(disp)[1] == 1:

        D = cartesian_displacement(a,
                                   b,
                                   c,
                                   np.deg2rad(alpha),
                                   np.deg2rad(beta),
                                   np.deg2rad(gamma))

        uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
        u11, u22, u33 = uiso[0,0], uiso[1,1], uiso[2,2]
        u23, u13, u12 = uiso[1,2], uiso[0,2], uiso[0,1]

        ani_disp = []
        for i in range(len(disp)):
            const = disp[i,0]
            ani_disp.append([const*u11, const*u22, const*u33,
                             const*u23, const*u13, const*u12])
        disp = np.array(ani_disp)

    U11 = np.tile(disp[:,0], n_uvw)
    U22 = np.tile(disp[:,1], n_uvw)
    U33 = np.tile(disp[:,2], n_uvw)
    U23 = np.tile(disp[:,3], n_uvw)
    U13 = np.tile(disp[:,4], n_uvw)
    U12 = np.tile(disp[:,5], n_uvw)

    MU = np.tile(mom[:,0], n_uvw)
    MV = np.tile(mom[:,1], n_uvw)
    MW = np.tile(mom[:,2], n_uvw)

    if '_atom_site_aniso_label' in cif_dict:
        cb.RemoveLoopItem('_atom_site_aniso_label')

    if '_atom_site_moment_label' in cif_dict:
        cb.RemoveLoopItem('_atom_site_moment_label')
        cb.RemoveLoopItem('_atom_site_moment.label')
        cb.RemoveLoopItem('_space_group_symop.magn_id')
        cb.RemoveLoopItem('_space_group_symop_magn_id')
        cb.RemoveLoopItem('_space_group_symop.magn_operation_xyz')
        cb.RemoveLoopItem('_space_group_symop_magn_operation_xyz')
        cb.RemoveLoopItem('_space_group_symop.magn_operation_mxmymz')
        cb.RemoveLoopItem('_space_group_symop_magn_operation_mxmymz')
        cb.RemoveLoopItem('_space_group_symop.magn_centering_id')
        cb.RemoveLoopItem('_space_group_symop_magn_centering_id')
        cb.RemoveLoopItem('_space_group_symop.magn_centering_xyz')
        cb.RemoveLoopItem('_space_group_symop_magn_centering_xyz')
        cb.RemoveLoopItem('_space_group_symop.magn_centering_mxmymz')
        cb.RemoveLoopItem('_space_group_symop_magn_centering_mxmymz')

    site, aniso, magn = [], [], []

    for i in range(atom.shape[0]):

        site.append([atom[i]+str(i),
                     U[i]/nu,
                     V[i]/nv,
                     W[i]/nw,
                     atom[i],
                     occupancy[i]])

        aniso.append([atom[i]+str(i),
                      U11[i],
                      U22[i],
                      U33[i],
                      U12[i],
                      U13[i],
                      U23[i]])

        magn.append([atom[i]+str(i),
                     MU[i],
                     MV[i],
                     MW[i],
                     'mx,my,mz'])

    cb.AddLoopItem((['_space_group_symop_operation_xyz'], ['']))

    cb['_space_group_symop_operation_xyz'] = ['x, y, z']

    cb.AddLoopItem((['_atom_site_label',
                     '_atom_site_fract_x',
                     '_atom_site_fract_y',
                     '_atom_site_fract_z',
                     '_atom_site_type_symbol',
                     '_atom_site_occupancy'],
                     map(list, zip(*site))))

    cb.AddLoopItem((['_atom_site_aniso_label',
                     '_atom_site_aniso_U_11',
                     '_atom_site_aniso_U_22',
                     '_atom_site_aniso_U_33',
                     '_atom_site_aniso_U_12',
                     '_atom_site_aniso_U_13',
                     '_atom_site_aniso_U_23'],
                     map(list, zip(*aniso))))

    if not np.allclose((MU,MV,MW), 0):
        cb.AddLoopItem((['_atom_site_moment_label',
                         '_atom_site_moment_crystalaxis_x',
                         '_atom_site_moment_crystalaxis_y',
                         '_atom_site_moment_crystalaxis_z',
                         '_atom_site_moment_symmform'],
                         map(list, zip(*magn))))

        if name.split('.')[-1] == 'cif':
            name = name.replace('cif', 'mcif')

    outfile = open(name, mode='w')

    # supress cf.WriteOut() output
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    outfile.write(cf.WriteOut())

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    outfile.close()

def disordered(delta, Ux, Uy, Uz, Sx, Sy, Sz, rx, ry, rz,
               nu, nv, nw, atm, A, name, folder, filename):
    """
    Write disordered supercell information to a CIF file.

    Parameters
    ----------
    delta : 1d array
        Relative occupancies.
    Ux, Uy, Uz : 1d array
        Atomic displacementsr in Cartesian coordinates.
    Sx, Sy, Sz : 1d array
        Magnetic momentsr in Cartesian coordinates.
    rx, ry, rz : 1d array
        Spatial vector in Cartesian coordinates.
    nu, nv, nw : int
        Number of grid points :math:`N_1`, :math:`N_2`, :math:`N_3` along the
        :math:`a`, :math:`b`, and :math:`c`-axis of the supercell.
    atm : 1d array, str
        Atoms, ions, or isotopes.
    A : 2d array
        Transformation matrix from crystal to Cartesian coordinates.
    folder : str,
        Name of path excluding filename of reference CIF file.
    filename : str
        Name of filename excluding path of reference CIF file.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_ops = ['_space_group_symop_operation_xyz',
                '_symmetry_equiv_pos_as_xyz',
                '_space_group_symop_magn_operation_xyz']

    ind_ops = next((i for i, loop_key in enumerate(loop_ops) \
                    if loop_key in cif_dict), None)

    symops = cif_dict[loop_ops[ind_ops]]

    for symop in symops:
        cb.RemoveLoopItem(symop)

    atomic_sites = cb.GetLoopNames('_atom_site_label')

    for asite in atomic_sites:
        cb.RemoveLoopItem(asite)

    if '_atom_site_aniso_label' in cif_dict:
        atomic_disp = cb.GetLoopNames('_atom_site_aniso_label')
        for adisp in atomic_disp:
            cb.RemoveLoopItem(adisp)

    a = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_a']))
    b = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_b']))
    c = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_c']))

    cb['_cell_length_a'] = str(a*nu)
    cb['_cell_length_b'] = str(b*nv)
    cb['_cell_length_c'] = str(c*nw)

    ux = rx+Ux
    uy = ry+Uy
    uz = rz+Uz

    A_inv = np.linalg.inv(A)

    u, v, w = transform(ux, uy, uz, A_inv)

    mx = np.round(Sx,4)
    my = np.round(Sy,4)
    mz = np.round(Sz,4)

    G = np.dot(A.T,A)

    a, b, c = np.sqrt(G[0,0]), np.sqrt(G[1,1]), np.sqrt(G[2,2])

    alpha = np.arccos(G[1,2]/(b*c))
    beta = np.arccos(G[0,2]/(a*c))
    gamma = np.arccos(G[0,1]/(a*b))

    C = cartesian_moment(a, b, c, alpha, beta, gamma)

    C_inv = np.linalg.inv(C)

    mu, mv, mw = transform(mx, my, mz, C_inv)

    atom = np.tile(atm, nu*nv*nw)

    site = []
    moment = []
    for i in range(atom.shape[0]):

       site.append([atom[i]+str(i),
                    u[i]/nu,
                    v[i]/nv,
                    w[i]/nw,
                    atom[i],
                    delta[i]])

       moment.append([atom[i]+str(i), mu[i], mv[i], mw[i], 'mx, my, mz'])

    cb.AddLoopItem((['_space_group_symop_operation_xyz'],['']))

    cb['_space_group_symop_operation_xyz'] = ['x, y, z']

    cb.AddLoopItem((['_atom_site_label',
                     '_atom_site_fract_x',
                     '_atom_site_fract_y',
                     '_atom_site_fract_z',
                     '_atom_site_type_symbol',
                     '_atom_site_occupancy'],
                     map(list, zip(*site))))

    if not np.allclose((mx,my,mz), 0):

        cb.AddLoopItem((['_atom_site_moment_label',
                         '_atom_site_moment_crystalaxis_x',
                         '_atom_site_moment_crystalaxis_y',
                         '_atom_site_moment_crystalaxis_z',
                         '_atom_site_moment_symmform'],
                         map(list, zip(*moment))))

        if (name.split('.')[-1] == 'cif'):
            name = name.replace('cif', 'mcif')

    outfile = open(name, mode='w')

    # supress cf.WriteOut() output
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    outfile.write(cf.WriteOut())

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    outfile.close()

def laue(folder, filename):
    """
    Determines Laue class from a CIF file.

    ===== ==================
    Laue  Noncentrosymmetric
    ===== ==================
    m-3m  432, -43m
    m-3   23
    6/mmm 622, -62m, 6mmm
    6/m   6, -6
    -3m   32, 3m
    -3    3
    4/mmm 422, -42m, 4mm
    4/m   4, -4
    mmm   222, 22m
    2/m   2, m
    -1    1
    ===== ==================

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    laue : str
        One of ``'-1'``, ``'2/m'``, ``'mmm'``, ``'4/m'``,
        ``'4/mmm'``, ``'-3'``, ``'-3m'``, ``'6/m'``, ``'6/mmm'``, ``'m-3'``, or
        ``'m-3m'``.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_ops = ['_space_group_symop_operation_xyz',
                '_symmetry_equiv_pos_as_xyz',
                '_space_group_symop_magn_operation_xyz']

    ind_ops = next((i for i, loop_key in enumerate(loop_ops) \
                    if loop_key in cif_dict), None)

    symops = cif_dict[loop_ops[ind_ops]]

    symops = symmetry.inverse(symmetry.inverse(symops))

    symops.append(u'-x,-y,-z')

    symops = np.unique(symops)

    laue_sym = symmetry.operators(invert=False)

    symmetries = list(laue_sym.keys())

    for symm in symmetries:
        if set(laue_sym[symm]) == set(symops):
            return symm

def twins(folder, filename):
    """
    Reads twin matrices from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    T : 3d array
        Twin matrices with individual components along last two axes.
    weights : 1d array
        Twin mass fraction.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    if '_twin_individual_id' in cif_dict:
        twin_ids = cif_dict['_twin_individual_id']
        n_var = len(twin_ids)
    else:
        n_var = 1

    if '_twin_individual_mass_fraction_refined' in cif_dict:
        twin_mf = cif_dict['_twin_individual_mass_fraction_refined']
        twin_mf = [float(re.sub(r'\([^()]*\)', '', mf)) for mf in twin_mf]

        T11s = cif_dict['_twin_individual_twin_matrix_11']
        T12s = cif_dict['_twin_individual_twin_matrix_12']
        T13s = cif_dict['_twin_individual_twin_matrix_13']
        T21s = cif_dict['_twin_individual_twin_matrix_21']
        T22s = cif_dict['_twin_individual_twin_matrix_22']
        T23s = cif_dict['_twin_individual_twin_matrix_23']
        T31s = cif_dict['_twin_individual_twin_matrix_31']
        T32s = cif_dict['_twin_individual_twin_matrix_32']
        T33s = cif_dict['_twin_individual_twin_matrix_33']

        T11s = [float(re.sub(r'\([^()]*\)', '', T11)) for T11 in T11s]
        T12s = [float(re.sub(r'\([^()]*\)', '', T12)) for T12 in T12s]
        T13s = [float(re.sub(r'\([^()]*\)', '', T13)) for T13 in T13s]
        T21s = [float(re.sub(r'\([^()]*\)', '', T21)) for T21 in T21s]
        T22s = [float(re.sub(r'\([^()]*\)', '', T22)) for T22 in T22s]
        T23s = [float(re.sub(r'\([^()]*\)', '', T23)) for T23 in T23s]
        T31s = [float(re.sub(r'\([^()]*\)', '', T31)) for T31 in T31s]
        T32s = [float(re.sub(r'\([^()]*\)', '', T32)) for T32 in T32s]
        T33s = [float(re.sub(r'\([^()]*\)', '', T33)) for T33 in T33s]
    else:
        twin_mf = [1.0]

        T11s = [1.0]
        T12s = [0.0]
        T13s = [0.0]
        T21s = [0.0]
        T22s = [1.0]
        T23s = [0.0]
        T31s = [0.0]
        T32s = [0.0]
        T33s = [1.0]

    weights = np.array(twin_mf)/np.sum(twin_mf)

    T = np.stack((T11s,T12s,T13s,
                  T21s,T22s,T23s,
                  T31s,T32s,T33s)).T.reshape(n_var,3,3)

    return T, weights

def parameters(folder=None, filename=None):
    """
    Reads lattice parameters from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())

    a = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_a']))
    b = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_b']))
    c = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_c']))

    alpha = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_alpha']))
    beta = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_beta']))
    gamma = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_angle_gamma']))

    return a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)

def group(folder, filename):
    """
    Reads space group information from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    group : int
        Space group number.
    hm : str
        Space group symbol in Hermann–Mauguin notation.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_hms = ['_space_group_name_h-m_alt',
                '_symmetry_space_group_name_h-m',
                '_parent_space_group_name_h-m']

    ind_hms = next((i for i, loop_key in enumerate(loop_hms) \
                    if loop_key in cif_dict), None)

    if ind_hms is not None:
        hm = cif_dict[loop_hms[ind_hms]]
    else:
        hm = ''

    hm = hm.replace('_', '').replace(' ', '')

    loop_gps = ['_space_group_it_number',
                '_symmetry_int_tables_number',
                '_parent_space_group_it_number']

    ind_gps = next((i for i, loop_key in enumerate(loop_gps) \
                    if loop_key in cif_dict), None)

    if ind_gps is not None:
        group = int(cif_dict[loop_gps[ind_gps]])
    else:
        group = 0

    if group == 0:
        if hm in tables.sg:
            group = tables.sg[hm]

    return group, hm

def operators(folder, filename):
    """
    Reads symmetry operators from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    symops : list
        Symmetry operators in Jones-faithful notation.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_ops = ['_space_group_symop_operation_xyz',
                '_symmetry_equiv_pos_as_xyz',
                '_space_group_symop_magn_operation_xyz']

    ind_ops = next((i for i, loop_key in enumerate(loop_ops) \
                    if loop_key in cif_dict), None)

    symops = cif_dict[loop_ops[ind_ops]]

    if ind_ops == 2:
        add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
        combine = []
        for symop in symops:
            for add_symop in add_symops:
                left_op = ','.join(symop.split(',')[:3])
                right_op = ','.join(add_symop.split(',')[:3])
                combine_op = symmetry.binary([left_op],[right_op])
                parity = int(symop.split(',')[-1])*int(add_symop.split(',')[-1])
                combine += [combine_op[0]+','+str(parity)]
        symops = combine

    symops = [symop.replace(' ', '') for symop in symops]
    symops = [','.join(symop.split(',')[:3]) for symop in symops]

    return symops

def vectors(folder, filename):
    """
    Reads propagation vectors from a CIF file.

    Parameters
    ----------
    folder : str,
        Name of path excluding filename.
    filename : str
        Name of filename excluding path.

    Returns
    -------
    kvecs : list
        Proapagtion operators.

    """

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    loop_kvecs = ['_parent_propagation_vector_kxkykz',
                  '_cell_wave_vector_seq_id']

    ind_kvecs = next((i for i, loop_key in enumerate(loop_kvecs) \
                    if loop_key in cif_dict), None)

    if ind_kvecs == 0:
        kvecs = cif_dict[loop_kvecs[ind_kvecs]]
        kvecs = [np.array([eval(k) for k in kvec]) for kvec in kvecs]
    elif ind_kvecs == 1:
        kxs = cif_dict['_cell_wave_vector_x']
        kys = cif_dict['_cell_wave_vector_y']
        kzs = cif_dict['_cell_wave_vector_z']
        kvecs = [np.array([kx, ky, kz]) for kx, ky, kz in zip(kxs, kys, kzs)]
    else:
        kvecs = []

    return kvecs

def lattice(a, b, c, alpha, beta, gamma):
    """
    Lattice system of unit cell based on lattice parameters.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    system : str
       One of ``'Cubic'``, ``'Hexagonal'``, ``'Rhombohedral'``,
       ``'Tetragonal'``, ``'Orthorhombic'``, ``'Monoclinic'``, or
       ``'Triclinic'``.

    """

    if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], np.pi/2)):
        return 'Cubic'
    elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)):
        return 'Rhombohedral'
    elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], np.pi/2)):
        return 'Tetragonal'
    elif (np.isclose(a, b) and \
          np.allclose([alpha, beta], np.pi/2) and \
          np.isclose(gamma, 2*np.pi/3)):
        return 'Hexagonal'
    elif (np.allclose([alpha, beta, gamma], np.pi/2)):
        return 'Orthorhombic'
    elif (np.allclose([alpha, beta], np.pi/2) or \
          np.allclose([alpha, gamma], np.pi/2)):
        return 'Monoclinic'
    else:
        return 'Triclinic'

def volume(a, b, c, alpha, beta, gamma):
    """
    Volume of unit cell.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    V : float
        Unit cell volume.

    """

    V = a*b*c*np.sqrt(1-np.cos(alpha)**2\
                       -np.cos(beta)**2\
                       -np.cos(gamma)**2\
                       +2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))

    return V

def reciprocal(a, b, c, alpha, beta, gamma):
    """
    Reciprocal lattice parameters.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    a_, b_, c_, alpha_, beta_, gamma_ : float
        Reciprocal lattice constants and angles. Angles in radians.

    """

    V = volume(a, b, c, alpha, beta, gamma)

    a_ = b*c*np.sin(alpha)/V
    b_ = c*a*np.sin(beta)/V
    c_ = a*b*np.sin(gamma)/V

    alpha_ = np.arccos((np.cos(beta)*np.cos(gamma)-np.cos(alpha))\
                       /np.sin(beta)/np.sin(gamma))
    beta_ = np.arccos((np.cos(gamma)*np.cos(alpha)-np.cos(beta))\
                      /np.sin(gamma)/np.sin(alpha))
    gamma_ = np.arccos((np.cos(alpha)*np.cos(beta)-np.cos(gamma))\
                       /np.sin(alpha)/np.sin(beta))

    return a_, b_, c_, alpha_, beta_, gamma_

def metric(a, b, c, alpha, beta, gamma):
    """
    Metric tensor.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    G : 2d array
       Components of the :math:`G` metric tensor.

    """

    G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                  [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                  [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

    return G

def d(a, b, c, alpha, beta, gamma, h, k, l):
    """
    Interplanar d-spacing.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.
    h, k, l : 1d array
        Miller indices

    Returns
    -------
    d : 1d array
        d-spacings.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    G_ = metric(a_, b_, c_, alpha_, beta_, gamma_)

    inv_d_spacing = np.sqrt(G_[0,0]*h*h+G_[0,1]*k*h+G_[0,2]*l*h+\
                            G_[1,0]*h*k+G_[1,1]*k*k+G_[1,2]*l*k+\
                            G_[2,0]*h*l+G_[2,1]*k*l+G_[2,2]*l*l)

    mask = np.isclose(inv_d_spacing, 0)

    if (np.sum(mask) > 0):
        n = np.argwhere(mask)
        inv_d_spacing[n] = 1

        d_spacing = 1/inv_d_spacing
        d_spacing[n] = np.nan
    else:
        d_spacing = 1/inv_d_spacing

    return d_spacing

def interplanar(a, b, c, alpha, beta, gamma, h0, k0, l0, h1, k1, l1):
    """
    Interplanar angle.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.
    h, k, l : 1d array
        Miller indices

    Returns
    -------
    angle : 1d array
        Interplanar angles in radians.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    G_ = metric(a_, b_, c_, alpha_, beta_, gamma_)

    inv_d0_spacing = np.sqrt(G_[0,0]*h0*h0+G_[0,1]*k0*h0+G_[0,2]*l0*h0+\
                             G_[1,0]*h0*k0+G_[1,1]*k0*k0+G_[1,2]*l0*k0+\
                             G_[2,0]*h0*l0+G_[2,1]*k0*l0+G_[2,2]*l0*l0)

    inv_d1_spacing = np.sqrt(G_[0,0]*h1*h1+G_[0,1]*k1*h1+G_[0,2]*l1*h1+\
                             G_[1,0]*h1*k1+G_[1,1]*k1*k1+G_[1,2]*l1*k1+\
                             G_[2,0]*h1*l1+G_[2,1]*k1*l1+G_[2,2]*l1*l1)

    mask0 = np.isclose(inv_d0_spacing, 0)
    mask1 = np.isclose(inv_d1_spacing, 0)

    if (np.sum(mask0) > 0):
        n0 = np.argwhere(mask0)
        inv_d0_spacing[n0] = 1

        d0_spacing = 1/inv_d0_spacing
        d0_spacing[n0] = np.nan
    else:
        d0_spacing = 1/inv_d0_spacing

    if (np.sum(mask1) > 0):
        n1 = np.argwhere(mask1)
        inv_d1_spacing[n1] = 1

        d1_spacing = 1/inv_d1_spacing
        d1_spacing[n1] = np.nan
    else:
        d1_spacing = 1/inv_d1_spacing

    inv_01 = G_[0,0]*h0*h1+G_[0,1]*k0*h1+G_[0,2]*l0*h1+\
             G_[1,0]*h0*k1+G_[1,1]*k0*k1+G_[1,2]*l0*k1+\
             G_[2,0]*h0*l1+G_[2,1]*k0*l1+G_[2,2]*l0*l1

    interplanar_angle = np.arccos(inv_01*d0_spacing*d1_spacing)

    return interplanar_angle

def cartesian(a, b, c, alpha, beta, gamma):
    """
    Transformation matrix from crystal to Cartesian coodinates.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    A : 2d array
        Transformation matrix.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    return np.array([[a, b*np.cos(gamma),  c*np.cos(beta)               ],
                     [0, b*np.sin(gamma), -c*np.sin(beta)*np.cos(alpha_)],
                     [0, 0,                1/c_                         ]])

def cartesian_rotation(a, b, c, alpha, beta, gamma):
    """
    Rotation matrix between reciprocal and real Cartesian coodinates.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    R : 2d array
        Rotation matrix.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    B = cartesian(a_, b_, c_, alpha_, beta_, gamma_)

    R = np.dot(np.linalg.inv(A).T, np.linalg.inv(B))

    return R

def cartesian_moment(a, b, c, alpha, beta, gamma):
    """
    Transformation matrix from crystal to Cartesian coodinates for magnetic
    moments.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    C : 2d array
        Transformation matrix.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    L = np.diag([a,b,c])

    return np.dot(A, np.linalg.inv(L))

def cartesian_displacement(a, b, c, alpha, beta, gamma):
    """
    Transformation matrix from crystal to Cartesian coodinates for atomic
    displacement parameters.

    Parameters
    ----------
    a, b, c, alpha, beta, gamma : float
        Lattice constants and angles. Angles in radians.

    Returns
    -------
    D : 2d array
        Transformation matrix.

    """

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    L_ = np.diag([a_,b_,c_])

    return np.dot(A, L_)

def vector(h, k, l, B):
    """
    Recirprocal lattice vector in Cartesian coodinates.

    Parameters
    ----------
    h, k, l : 1d array
        Miller indices.
    B : 2d array
        Transformation matrix from crystal to Cartesian coodinates in
        reciprocal space.

    Returns
    -------
    Qx, Qy, Qz : 1d array
        Recirprocal lattice vector in Cartesian coodinates.

    """

    kh, kk, kl = transform(h, k, l, B)

    Qh, Qk, Ql = 2*np.pi*kh, 2*np.pi*kk, 2*np.pi*kl

    return Qh, Qk, Ql

def transform(p, q, r, U):
    """
    Transform the components of a vector.

    Parameters
    ----------
    p, q, r : 1d array
        Components of a vector.
    U : 2d array
        Transformation matrix.

    Returns
    -------
    x, y, z : 1d array
        Transformed components of a vector.

    """

    x = U[0,0]*p+U[0,1]*q+U[0,2]*r
    y = U[1,0]*p+U[1,1]*q+U[1,2]*r
    z = U[2,0]*p+U[2,1]*q+U[2,2]*r

    return x, y, z
