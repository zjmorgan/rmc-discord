#!/usr/bin/env python3

import CifFile
import numpy as np

import sys, os, io, re

from disorder.material import symmetry
from disorder.material import tables

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.abspath(os.path.join(directory, '..', 'data'))

def unitcell(folder=folder, filename='copper.cif', tol=1e-2):

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

    if (ind_ops == 2):
        add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
        combine = []
        for symop in symops:
            for add_symop in add_symops:
                left_op = ','.join(symop.split(',')[:3])
                right_op = ','.join(add_symop.split(',')[:3])
                combine_op = symmetry.binary([left_op],[right_op])
                combine += combine_op
        symops = combine

    atomic_sites = cif_dict['_atom_site_label']

    if ('_atom_site_moment_label' in cif_dict):
        magnetic_sites = cif_dict['_atom_site_moment_label']
        magnetic_atoms = [asite in magnetic_sites \
                          for asite in atomic_sites]

    atomic_sites = [re.sub(r'[0-9]', '', asite) for asite in atomic_sites]

    if ('_atom_site_type_symbol' in cif_dict):
        symbols = cif_dict['_atom_site_type_symbol']
    elif ('_atom_site_symbol' in cif_dict):
        symbols = cif_dict['_atom_site_symbol']
    else:
        symbols = atomic_sites

    symbols = [sym if (sym != 'D') else '2H' for sym in symbols]

    xs = cif_dict['_atom_site_fract_x']
    ys = cif_dict['_atom_site_fract_y']
    zs = cif_dict['_atom_site_fract_z']

    xs = [re.sub(r'\([^()]*\)', '', x) for x in xs]
    ys = [re.sub(r'\([^()]*\)', '', y) for y in ys]
    zs = [re.sub(r'\([^()]*\)', '', z) for z in zs]

    if ('_atom_site_occupancy' in cif_dict):
        occs = cif_dict['_atom_site_occupancy']
        occs = [re.sub(r'\([^()]*\)', '', occ) for occ in occs]
    else:
        occs = [1.0]*len(atomic_sites)

    iso = False
    ani = False

    if ('_atom_site_aniso_label' in cif_dict):
        adp_type, ani = 'ani', True
        if ('_atom_site_aniso_u_11' in cif_dict):
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

    if ('_atom_site_u_iso_or_equiv' in cif_dict):
        adp_type, iso = 'iso', True
        adp_U = True
        Uisos = cif_dict['_atom_site_u_iso_or_equiv']
        Uisos = [re.sub(r'\([^()]*\)', '', Uiso) for Uiso in Uisos]
    elif ('_atom_site_b_iso_or_equiv' in cif_dict):
        adp_type, iso = 'iso', True
        adp_U = False
        Bisos = cif_dict['_atom_site_b_iso_or_equiv']
        Bisos = [re.sub(r'\([^()]*\)', '', Biso) for Biso in Bisos]
    elif (ani == False):
        adp_type, iso = 'iso', True
        adp_U = True
        Uisos = [0.0]*len(atomic_sites)

    if (ani and iso):
        adp_type = 'ani'
        D = cartesian_displacement(*parameters(folder, filename))
        iso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
        iso_labels = cif_dict['_atom_site_label']
        ani_labels = cif_dict['_atom_site_aniso_label']
        adp_types = cif_dict['_atom_site_adp_type']
        U11, U22, U33, U23, U13, U12 = [], [], [], [], [], []
        B11, B22, B33, B23, B13, B12 = [], [], [], [], [], []
        for adp, label in zip(adp_types, iso_labels):
            if (adp.lower() == 'uani'):
                index = ani_labels.index(label)
                U11.append(U11s[index])
                U22.append(U22s[index])
                U33.append(U33s[index])
                U23.append(U23s[index])
                U13.append(U13s[index])
                U12.append(U12s[index])
            elif (adp.lower() == 'bani'):
                index = ani_labels.index(label)
                B11.append(B11s[index])
                B22.append(B22s[index])
                B33.append(B33s[index])
                B23.append(B23s[index])
                B13.append(B13s[index])
                B12.append(B12s[index])
            elif (adp.lower() == 'uiso'):
                index = iso_labels.index(label)
                Uani = iso*float(Uisos[index])
                U11.append(str(Uani[0,0]))
                U22.append(str(Uani[1,1]))
                U33.append(str(Uani[2,2]))
                U23.append(str(Uani[1,2]))
                U13.append(str(Uani[0,2]))
                U12.append(str(Uani[0,1]))
            elif (adp.lower() == 'biso'):
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

    if ('_atom_site_moment_label' in cif_dict):
        mxs = cif_dict['_atom_site_moment_crystalaxis_x']
        mys = cif_dict['_atom_site_moment_crystalaxis_y']
        mzs = cif_dict['_atom_site_moment_crystalaxis_z']
        mxs = [re.sub(r'\([^()]*\)', '', mx) for mx in mxs]
        mys = [re.sub(r'\([^()]*\)', '', my) for my in mys]
        mzs = [re.sub(r'\([^()]*\)', '', mz) for mz in mzs]
        if (len(magnetic_sites) != len(atomic_sites)):
            j = 0
            for i, mag in enumerate(magnetic_atoms):
                if mag:
                    Mxs[i], Mys[i], Mzs[i] = mxs[j], mys[j], mzs[j]
                    j += 1
        else:
            Mxs, Mys, Mzs = mxs, mys, mzs

    if ('_space_group_symop_magn_centering_mxmymz' in cif_dict):
        mag_symops = cif_dict['_space_group_symop_magn_centering_mxmymz']
        mag_symops += cif_dict['_space_group_symop_magn_operation_mxmymz']
    else:
        mag_symops = ['mx,my,mz']*len(symops)

    c, d, m, s = [], [], [], []

    total, types, operators, mag_operators = [], [], [], []

    cf = 8*np.pi**2

    for i, asite in enumerate(atomic_sites):

        x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        Mx, My, Mz = float(Mxs[i]), float(Mys[i]), float(Mzs[i])

        occ = float(occs[i])

        if (adp_type == 'ani'):
            if (adp_U is False):
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
            if (adp_U is False):
                Biso = float(Bisos[i])
                Uiso = Biso/(8*np.pi**2)
            else:
                Uiso = float(Uisos[i])

        symbol = symbols[i]

        for symop, mag_symop in zip(symops, mag_symops):

            transformed = symmetry.evaluate([symop], [x,y,z]).flatten()
            
            mom = symmetry.evaluate_mag([mag_symop], [Mx,My,Mz]).flatten()

            if (adp_type == 'ani'):
                disp = symmetry.evaluate_disp([symop], [U11,U22,U33,
                                                        U23,U13,U12])
            else:
                disp = [Uiso]

            transformed = [tf+(tf < 0)-(tf >= 1) for tf in transformed]

            total.append(transformed)
            types.append(symbol)
            operators.append(symop)
            mag_operators.append(mag_symop)

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

    unit_cell_dict = {}

    unit_cell_dict['u'] = u
    unit_cell_dict['v'] = v
    unit_cell_dict['w'] = w
    unit_cell_dict['occupancy'] = c
    unit_cell_dict['displacement'] = d
    unit_cell_dict['moment'] = m
    unit_cell_dict['site'] = s
    unit_cell_dict['operator'] = ops
    unit_cell_dict['magnetic_operator'] = mag_ops
    unit_cell_dict['atom'] = atm
    unit_cell_dict['n_atom'] = n_atm

    return unit_cell_dict

def supercell(atm, occ, disp, mom, u, v, w, nu, nv, nw,
              name, folder=None, filename=None):

    if (filename != None):

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

        if (ind_ops == 2):
            add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
            combine = []
            for symop in symops:
                for add_symop in add_symops:
                    left_op = ','.join(symop.split(',')[:3])
                    right_op = ','.join(add_symop.split(',')[:3])
                    combine_op = symmetry.binary([left_op],[right_op])
                    combine += combine_op
            symops = combine

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

        if (np.shape(disp)[1] == 1):

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

        if ('_atom_site_aniso_label' in cif_dict):
            cb.RemoveLoopItem('_atom_site_aniso_label')

        if ('_atom_site_moment_label' in cif_dict):
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

        if (not np.allclose((MU,MV,MW), 0)):
            cb.AddLoopItem((['_atom_site_moment_label',
                             '_atom_site_moment_crystalaxis_x',
                             '_atom_site_moment_crystalaxis_y',
                             '_atom_site_moment_crystalaxis_z',
                             '_atom_site_moment_symmform'],
                             map(list, zip(*magn))))

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

def disordered(delta, Ux, Uy, Uz, Sx, Sy, Sz, rx, ry, rz,
               nu, nv, nw, atm, A, name, folder=None, filename=None,
               ulim=[0,1], vlim=[0,1], wlim=[0,1]):

    if (filename != None):

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

        if (ind_ops == 2):
            add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
            combine = []
            for symop in symops:
                for add_symop in add_symops:
                    left_op = ','.join(symop.split(',')[:3])
                    right_op = ','.join(add_symop.split(',')[:3])
                    combine_op = symmetry.binary([left_op],[right_op])
                    combine += combine_op
            symops = combine

        for symop in symops:
            cb.RemoveLoopItem(symop)

        atomic_sites = cb.GetLoopNames('_atom_site_label')

        for asite in atomic_sites:
            cb.RemoveLoopItem(asite)

        if ('_atom_site_aniso_label' in cif_dict):
            atomic_disp = cb.GetLoopNames('_atom_site_aniso_label')
            for adisp in atomic_disp:
                cb.RemoveLoopItem(adisp)

        n_atm = atm.shape[0]

        i0, i1 = ulim[0], ulim[1]
        j0, j1 = vlim[0], vlim[1]
        k0, k1 = wlim[0], wlim[1]

        na, nb, nc = i1-i0, j1-j0, k1-k0

        a = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_a']))
        b = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_b']))
        c = float(re.sub(r'\([^()]*\)', '', cif_dict['_cell_length_c']))

        cb['_cell_length_a'] = str(a*na)
        cb['_cell_length_b'] = str(b*nb)
        cb['_cell_length_c'] = str(c*nc)

        mask = np.full((nu,nv,nw,n_atm), fill_value=False)
        mask[i0:i1,j0:j1,k0:k1,:] = True
        mask = mask.flatten()

        ux = rx[mask]+Ux[mask]
        uy = ry[mask]+Uy[mask]
        uz = rz[mask]+Uz[mask]

        A_inv = np.linalg.inv(A)

        u, v, w = transform(ux, uy, uz, A_inv)

        mx = np.round(Sx[mask],4)
        my = np.round(Sy[mask],4)
        mz = np.round(Sz[mask],4)
        
        G = np.dot(A.T,A)
        
        a, b, c = np.sqrt(G[0,0]), np.sqrt(G[1,1]), np.sqrt(G[2,2])
        
        alpha = np.arccos(G[1,2]/(b*c))
        beta = np.arccos(G[0,2]/(a*c))
        gamma = np.arccos(G[0,1]/(a*b))
        
        C = cartesian_moment(a, b, c, alpha, beta, gamma)

        C_inv = np.linalg.inv(C)

        mu, mv, mw = transform(mx, my, mz, C_inv)

        atom = np.tile(atm, na*nb*nc)

        site = []
        moment = []
        for i in range(atom.shape[0]):

           site.append([atom[i]+str(i),
                        u[i]/na,
                        v[i]/nb,
                        w[i]/nc,
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

        if (not np.allclose((mx,my,mz), 0)):

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

    if (ind_ops == 2):
        add_symops = cif_dict['_space_group_symop_magn_centering_xyz']
        combine = []
        for symop in symops:
            for add_symop in add_symops:
                left_op = ','.join(symop.split(',')[:3])
                right_op = ','.join(add_symop.split(',')[:3])
                combine_op = symmetry.binary([left_op],[right_op])
                combine += combine_op
        symops = combine

    symops = symmetry.inverse(symmetry.inverse(symops))

    symops.append(u'-x,-y,-z')

    symops = np.unique(symops)

    laue_sym = symmetry.operators(invert=False)

    symmetries = list(laue_sym.keys())

    for symm in symmetries:
        if (set(laue_sym[symm]) == set(symops)):
            return symm

def twins(folder, filename):

    cf = CifFile.ReadCif(os.path.join(folder, filename))
    cb = cf[[key for key in cf.keys() \
             if cf[key].get('_cell_length_a') is not None][0]]

    cif_dict = dict(cb.items())
    cif_dict = {k.replace('.','_'):v for k,v in cif_dict.items()}

    if ('_twin_individual_id' in cif_dict):
        twin_ids = cif_dict['_twin_individual_id']
        n_var = len(twin_ids)
    else:
        n_var = 1

    if ('_twin_individual_mass_fraction_refined' in cif_dict):
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

    weight = np.array(twin_mf)

    U = np.stack((T11s,T12s,T13s,
                  T21s,T22s,T23s,
                  T31s,T32s,T33s)).T.reshape(n_var,3,3)

    return U, weight

def parameters(folder=None, filename=None):

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

def group(folder=None, filename=None):

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

    if (ind_hms is not None):
        hm = cif_dict[loop_hms[ind_hms]]
    else:
        hm = ''

    hm = hm.replace('_', '').replace(' ', '')

    loop_gps = ['_space_group_it_number',
                '_symmetry_int_tables_number',
                '_parent_space_group_it_number']

    ind_gps = next((i for i, loop_key in enumerate(loop_gps) \
                    if loop_key in cif_dict), None)

    if (ind_gps is not None):
        group = int(cif_dict[loop_gps[ind_gps]])
    else:
        group = 0

    if (group == 0):
        if (hm in tables.sg):
            group = tables.sg[hm]

    return group, hm

def operators(folder=folder, filename='copper.cif'):

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

    symops = [symop.replace(' ', '') for symop in symops]

    return symops

def lattice(a, b, c, alpha, beta, gamma):

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

    V = a*b*c*np.sqrt(1-np.cos(alpha)**2\
                       -np.cos(beta)**2\
                       -np.cos(gamma)**2\
                       +2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))

    return V

def reciprocal(a, b, c, alpha, beta, gamma):

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

    G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                  [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                  [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

    return G

def d(a, b, c, alpha, beta, gamma, h, k, l):

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

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    return np.array([[a, b*np.cos(gamma),  c*np.cos(beta)],
                     [0, b*np.sin(gamma), -c*np.sin(beta)*np.cos(alpha_)],
                     [0, 0,                1/c_]])

def cartesian_rotation(a, b, c, alpha, beta, gamma):

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    B = cartesian(a_, b_, c_, alpha_, beta_, gamma_)

    R = np.dot(np.linalg.inv(A).T, np.linalg.inv(B))

    return R

def cartesian_moment(a, b, c, alpha, beta, gamma):

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    L = np.array([[a,0,0],[0,b,0],[0,0,c]])

    return np.dot(A, np.linalg.inv(L))

def cartesian_displacement(a, b, c, alpha, beta, gamma):

    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, b, c, alpha, beta, gamma)

    A = cartesian(a, b, c, alpha, beta, gamma)

    L_ = np.array([[a_,0,0],[0,b_,0],[0,0,c_]])

    return np.dot(A, L_)

def vector(h, k, l, B):

    qh, qk, ql = transform(h, k, l, B)

    Qh, Qk, Ql = 2*np.pi*qh, 2*np.pi*qk, 2*np.pi*ql

    return Qh, Qk, Ql

def transform(p, q, r, U):

    x = U[0,0]*p+U[0,1]*q+U[0,2]*r
    y = U[1,0]*p+U[1,1]*q+U[1,2]*r
    z = U[2,0]*p+U[2,1]*q+U[2,2]*r

    return x, y, z

def pairs(u, v, w, ion, A, extend=False):

    n_atm = ion.shape[0]

    i, j = np.triu_indices(n_atm, k=1)
    i, j = np.concatenate((i,j)), np.concatenate((j,i))

    du = u[j]-u[i]
    dv = v[j]-v[i]
    dw = w[j]-w[i]

    u_img = -1*(du > 0.5)+(du < -0.5)
    v_img = -1*(dv > 0.5)+(dv < -0.5)
    w_img = -1*(dw > 0.5)+(dw < -0.5)

    du[du < -0.5] += 1
    dv[dv < -0.5] += 1
    dw[dw < -0.5] += 1
    
    du[du > 0.5] -= 1
    dv[dv > 0.5] -= 1
    dw[dw > 0.5] -= 1
    
    if extend:

        U, V, W = np.meshgrid(np.arange(-1,2),
                              np.arange(-1,2),
                              np.arange(-1,2), indexing='ij')

        U = U.flatten()[:,np.newaxis]
        V = V.flatten()[:,np.newaxis]
        W = W.flatten()[:,np.newaxis]

        u_img = (u_img+U).flatten()
        v_img = (v_img+V).flatten()
        w_img = (w_img+W).flatten()

        du, dv, dw = (du+U).flatten(), (dv+V).flatten(), (dw+W).flatten()

        U = np.repeat(np.delete(U.flatten(), 13), n_atm)
        V = np.repeat(np.delete(V.flatten(), 13), n_atm)
        W = np.repeat(np.delete(W.flatten(), 13), n_atm)

        du = np.concatenate((du,U))
        dv = np.concatenate((dv,V))
        dw = np.concatenate((dw,W))

        u_img = np.concatenate((u_img,U))
        v_img = np.concatenate((v_img,V))
        w_img = np.concatenate((w_img,W))

        i = np.tile(i, 27)
        j = np.tile(j, 27)

        i = np.concatenate((i,np.tile(np.arange(n_atm), 26)))
        j = np.concatenate((j,np.tile(np.arange(n_atm), 26)))
        
    dx, dy, dz = transform(du, dv, dw, A)

    d = np.sqrt(dx**2+dy**2+dz**2)

    atms = np.stack((ion[i],ion[j]))

    ion_ion = np.array(['_'.join((a,b)) for a, b in zip(*atms)])
    ions, ion_labels = np.unique(ion_ion, return_inverse=True)

    metric = np.stack((dx,dy,dz,d)).T

    pair_dict = { }

    for k in range(n_atm):

        mask = i == k

        sort = np.lexsort(metric[mask].T)

        label_dict = { }

        l = j[mask][sort]
        ion_lab = ion_labels[mask][sort]
        ion_ref = [ion_pair.split('_')[1] for ion_pair in ion_ion[mask][sort]]

        iu, iv, iw = u_img[mask][sort], v_img[mask][sort], w_img[mask][sort]

        c = 0
        m, c_uvw = [], []
        m.append(l[0])
        c_uvw.append((iu[0],iv[0],iw[0]))
        ind = -1
        for ind in range(ion_lab.shape[0]-1):
            if ion_lab[ind] != ion_lab[ind+1]:
                key = c, ion_ref[ind]
                label_dict[key] = m, c_uvw
                m, c_uvw = [], []
                c += 1
            m.append(l[ind+1])
            c_uvw.append((iu[ind+1],iv[ind+1],iw[ind+1]))
        key = c, ion_ref[ind+1]
        label_dict[key] = m, c_uvw

        pair_dict[k] = label_dict

    return pair_dict