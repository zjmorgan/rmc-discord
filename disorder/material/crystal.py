#!/usr/bin/env python3

import CifFile
import numpy as np

from scipy import spatial

import sys, os, io, re

from disorder.material import symmetry
from disorder.material import tables

def unitcell(folder=None, 
             filename=None, 
             occupancy=False, 
             displacement=False, 
             moment=False,
             site=False, 
             operator=False,
             magnetic_operator=False,
             tol=1e-2):
    
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
                combine.append(symmetry.binary(
                               ','.join(symop.split(',')[:3]),
                               ','.join(add_symop.split(',')[:3])))
        symops = combine  
                            
    atomic_sites = cif_dict['_atom_site_label']
     
    if ('_atom_site_moment_label' in cif_dict):
        magnetic_sites = cif_dict['_atom_site_moment_label']
        magnetic_atoms = [asite in magnetic_sites \
                          for asite in atomic_sites]    
    
    atomic_sites = [re.sub(r'[0-9]', '', asite) for asite in atomic_sites]
    
    if ('_atom_site_symbol' in cif_dict):
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

    if ('_atom_site_aniso_label' in cif_dict):
        adp_type = 'ani'
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
    else:
        adp_type = 'iso'
        if ('_atom_site_u_iso_or_equiv' in cif_dict):
            adp_U = True
            Uisos = cif_dict['_atom_site_u_iso_or_equiv']
            Uisos = [re.sub(r'\([^()]*\)', '', Uiso) for Uiso in Uisos]
        elif ('_atom_site_b_iso_or_equiv' in cif_dict):
            adp_U = False
            Uisos = cif_dict['_atom_site_b_iso_or_equiv']
            Uisos = [re.sub(r'\([^()]*\)', '', Uiso) for Uiso in Uisos]
        if ('_atom_site_u_iso_or_equiv' in cif_dict):
            adp_U = True
            Bisos = cif_dict['_atom_site_u_iso_or_equiv']
            Bisos = [re.sub(r'\([^()]*\)', '', Biso) for Biso in Bisos]
        elif ('_atom_site_b_iso_or_equiv' in cif_dict):
            adp_U = False
            Bisos = cif_dict['_atom_site_b_iso_or_equiv']
            Bisos = [re.sub(r'\([^()]*\)', '', Biso) for Biso in Bisos]
        else:
            adp_U = True
            Uisos = [0.0]*len(atomic_sites) 
        
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
                
                cf = 8*np.pi**2
                U11, U22, U33 = B11/cf, B22/cf, B33/cf
                U23, U13, U12 = B23/cf, B13/cf, B12/cf
            else:
                U11 = float(U11s[i])
                U22 = float(U22s[i])
                U33 = float(U33s[i])
                U23 = float(U23s[i])
                U13 = float(U13s[i])          
                U12 = float(U12s[i])
                
            disp = [U11,U22,U33,U23,U13,U12]
                
        else:
            if (adp_U is False):
                Biso = float(Bisos[i])

                Uiso = Biso/(8*np.pi**2)
            else:
                Uiso = float(Uisos[i])

            disp = [Uiso]

        symbol = symbols[i]
            
        for symop, mag_symop in zip(symops, mag_symops):
                            
            transformed = symmetry.evaluate(symop, [x,y,z])[:3]
                            
            mom = symmetry.evaluate_mag(mag_symop, [Mx,My,Mz])
            
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
    
    output = (u, v, w,)
   
    if occupancy: output = (*output, c)
    if displacement: output = (*output, d)
    if moment: output = (*output, m)
    if site: output = (*output, s)
    if operator: output = (*output, ops)
    if magnetic_operator: output = (*output, mag_ops)
        
    output = (*output, atm, n_atm)
    
    return output

def nuclear(H, K, L, h=None, k=None, l=None, nu=1, nv=1, nw=1, centering=None):
    
    iH = np.mod(H, nu) 
    iK = np.mod(K, nv)
    iL = np.mod(L, nw)
    
    if (h is None and k is None and l is None):
        h = H // nu
        k = K // nv
        l = L // nw
    else:
        h = np.round(h).astype(int)
        k = np.round(k).astype(int)
        l = np.round(l).astype(int)
    
    dft_cond = (iH == 0) & (iK == 0) & (iL == 0)
    
    if (centering is None):
        cond = dft_cond
    elif (centering == 'P'):
        cond = (h % 1 == 0) & (k % 1 == 0) & (l % 1 == 0) & (dft_cond)
    elif (centering == 'I'):
        cond = ((h+k+l) % 2 == 0) & (dft_cond)
    elif (centering == 'F'):
        cond = ((h+k) % 2 == 0) \
             & ((k+l) % 2 == 0) \
             & ((l+h) % 2 == 0) & (dft_cond)
    elif (centering == 'R'):
        cond = ((-h+k+l) % 3 == 0) & (dft_cond)
    elif (centering == 'C'):
        cond = ((h+k) % 2 == 0) & (dft_cond)
    elif (centering == 'A'):
        cond = ((k+l) % 2 == 0) & (dft_cond)
    elif (centering == 'B'):
        cond = ((l+h) % 2 == 0) & (dft_cond)
        
    return H[cond], K[cond], L[cond], cond

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
                combine.append(symmetry.binary(
                               ','.join(symop.split(',')[:3]),
                               ','.join(add_symop.split(',')[:3])))
        symops = combine 
        
    symops = symmetry.inverse(symmetry.inverse(symops)).tolist()
        
    symops.append(u'-x,-y,-z')

    symops = np.unique(symops)
    
    lauesym = symmetry.operators(invert=False)
    
    symmetries = list(lauesym.keys())

    for symm in symmetries:
        if (set(lauesym[symm]) == set(symops)):
            return symm
           
def bragg(h_range, 
          k_range, 
          l_range, 
          nh,
          nk,
          nl,
          nu,
          nv,
          nw,
          T=np.eye(3),
          laue=None):
    
    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                             np.linspace(k_range[0],k_range[1],nk), 
                             np.linspace(l_range[0],l_range[1],nl), 
                             indexing='ij')
     
    h_ = h_.flatten()
    k_ = k_.flatten()
    l_ = l_.flatten()
    
    h = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
    k = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
    l = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
    
    H = (h*nu).astype(int)
    K = (k*nv).astype(int)
    L = (l*nw).astype(int)
    
    iH = np.mod(H, nu)
    iK = np.mod(K, nv)
    iL = np.mod(L, nw)
    
    mask = (iH == 0) & (~np.isclose(np.mod(h*nu,nu),0))
    H[mask] += 1
    
    mask = (iK == 0) & (~np.isclose(np.mod(k*nv,nv),0))
    K[mask] += 1
    
    mask = (iL == 0) & (~np.isclose(np.mod(l*nw,nw),0))
    L[mask] += 1
    
    if (laue == None or laue == 'None'):
        
        index = np.arange(nh*nk*nl)
        
        return h, \
               k, \
               l, \
               H, \
               K, \
               L, \
               index, \
               index, \
               np.array([u'x,y,z'])

    symops = np.array(symmetry.laue(laue))
    
    symops = symmetry.inverse(symops)
    
    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])
                
    total = []
    
    coordinate = np.stack((H,K,L))
        
    cosymmetries, coindices, coinverses = np.unique(coordinate,
                                                    axis=1, 
                                                    return_index=True, 
                                                    return_inverse=True)
        
    for op in symops:
                
        transformed = symmetry.evaluate([op], cosymmetries, translate=False)
                                        
        total.append(transformed.T.tolist())
        
    index = np.arange(coordinate.shape[1])
                               
    total = np.array(total)
        
    for i in range(cosymmetries.shape[1]):
        
        total[:,i,:] = total[np.lexsort(total[:,i,:].T),i,:]
        
    total = np.hstack(total)
     
    _, indices, inverses = np.unique(total, 
                                     axis=0, 
                                     return_index=True, 
                                     return_inverse=True)
        
    reverses = np.arange(indices.shape[0])
                       
    return h[coindices][indices], \
           k[coindices][indices], \
           l[coindices][indices], \
           H[coindices][indices], \
           K[coindices][indices], \
           L[coindices][indices], \
           index[coindices][indices], \
           reverses[inverses][coinverses], \
           symops

def reduced(h_range, 
            k_range, 
            l_range, 
            nh,
            nk,
            nl,
            nu,
            nv,
            nw,
            T=np.eye(3), 
            laue=None):
    
    h_, k_, l_ = np.meshgrid(np.linspace(h_range[0],h_range[1],nh), 
                             np.linspace(k_range[0],k_range[1],nk), 
                             np.linspace(l_range[0],l_range[1],nl), 
                             indexing='ij')
    
    h = T[0,0]*h_+T[0,1]*k_+T[0,2]*l_
    k = T[1,0]*h_+T[1,1]*k_+T[1,2]*l_
    l = T[2,0]*h_+T[2,1]*k_+T[2,2]*l_
        
    h = h.flatten()
    k = k.flatten()
    l = l.flatten()
    
    del h_, k_, l_
        
    if (nh > 1):
        h_max_res = (h_range[1]-h_range[0])/(nh-1)
    else:
        h_max_res = 0
        
    if (nk > 1):
        k_max_res = (k_range[1]-k_range[0])/(nk-1)
    else:
        k_max_res = 0
        
    if (nl > 1):
        l_max_res = (l_range[1]-l_range[0])/(nl-1)
    else:
        l_max_res = 0
        
    hkl_max_res = np.array([[h_max_res,0,0],[0,k_max_res,0],[0,0,l_max_res]])
    hkl_res = np.abs(np.dot(T, hkl_max_res))
    
    h_res, k_res, l_res = np.max(hkl_res, axis=0)
    
    if (h_res > 0 and h_res < 1/nu):
        Nu = int(1/h_res // nu)*nu
    else:
        Nu = nu
        
    if (k_res > 0 and k_res < 1/nv):
        Nv = int(1/k_res // nv)*nv
    else:
        Nv = nv
    
    if (l_res > 0 and l_res < 1/nw):
        Nw = int(1/l_res // nw)*nw
    else:
        Nw = nw
        
    H = (h*Nu).astype(np.int16)
    
    iH = np.mod(H, Nu) # // (Nu // nu)
    mask = (iH < Nu//nu) & (~np.isclose(np.mod(h*Nu,Nu),0))
    # H[mask] += Nu//nu
    del iH, mask
    
    del h
        
    K = (k*Nv).astype(np.int16)
    
    iK = np.mod(K, Nv) # // (Nv // nv)
    mask = (iK < Nv//nv) & (~np.isclose(np.mod(k*Nv,Nv),0))
    # K[mask] += Nv//nv
    del iK, mask
    
    del k
    
    L = (l*Nw).astype(np.int16)
    
    iL = np.mod(L, Nw) # // (Nw // nw)
    mask = (iL < Nw//nw) & (~np.isclose(np.mod(l*Nw,Nw),0))
    # L[mask] += Nw//nw
    del iL, mask  
        
    del l
            
    if (laue == None or laue == 'None'):
        
        index = np.arange(nh*nk*nl)
        
        return index, \
               index, \
               np.array([u'x,y,z']), \
               Nu, \
               Nv, \
               Nw
               
    symops = np.array(symmetry.laue(laue))
    
    symops = symmetry.inverse(symops)
                    
    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])
                    
    coordinate = np.ascontiguousarray(np.stack(((H,-H),(K,-K),(L,-L))).T)
    
    del H, K, L
            
    n = coordinate.shape[0]
        
    pair = np.zeros((n,3), dtype=np.int16)
                    
    symmetry.friedel(pair, coordinate)

    del coordinate
        
    cosymmetries, coindices, coinverses = symmetry.unique(pair)
                        
    del pair
    
    n_symm = cosymmetries.shape[0]
        
    total = np.zeros((n_symm,3), dtype=np.int16)
    
    laue_sym = symmetry.operators(invert=True)
    
    symop = [11,1]
    
    for count, sym in enumerate(list(laue_sym.keys())):
        if (np.array([symops[p] in laue_sym.get(sym) \
                      for p in range(symops.shape[0])]).all() and \
             len(laue_sym.get(sym)) == symops.shape[0]):
            
            symop = [count,len(laue_sym.get(sym))]
        
    index = np.arange(n)
                
    symmetry.sorting(total, cosymmetries, symop)
    
    _, indices, inverses = symmetry.unique(total)
    
    reverses = np.arange(indices.shape[0])
                       
    return index[coindices][indices], \
           reverses[inverses][coinverses], \
           symops, \
           Nu, Nv, Nw
           
def multiplicity(h, k, l, laue):
            
    total = []
    
    coordinate = np.stack((h,k,l))
    
    symops = np.array(symmetry.laue(laue))
    
    symops = symmetry.inverse(symops)
            
    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])
            
    for op in symops:
                
        transformed = symmetry.evaluate([op], coordinate, translate=False)
                                        
        total.append(transformed.T.tolist())
                               
    total = np.hstack(total)
    
    m = np.zeros(coordinate.shape[1])
    
    for i in range(coordinate.shape[1]):
        
        array = total[i,:].reshape(len(symops),3)
        
        symmetries, indices, counts = np.unique(array,\
                                                axis=0, 
                                                return_index=True, 
                                                return_counts=True)
            
        m[i] = counts[0]
        
    return m, symops

def spherical(Q_range, B, laue, tol=0.00001):
    
    Q_min = Q_range[0]
    Q_max = Q_range[1]
    
    inv_d = np.array([Q_min,Q_max])/2/np.pi
    
    h_range = inv_d/np.sqrt(B[0,0]**2+B[1,0]**2+B[2,0]**2)
    k_range = inv_d/np.sqrt(B[0,1]**2+B[1,1]**2+B[2,1]**2)
    l_range = inv_d/np.sqrt(B[0,2]**2+B[1,2]**2+B[2,2]**2)
    
    h, k, l = np.meshgrid(np.arange(h_range[0],h_range[1]+1).astype(int), 
                          np.arange(k_range[0],k_range[1]+1).astype(int), 
                          np.arange(l_range[0],l_range[1]+1).astype(int), 
                          indexing='ij')
    
    h, k, l = h.flatten(), k.flatten(), l.flatten()
        
    symops = np.array(symmetry.laue(laue))
    
    symops = symmetry.inverse(symops)
    
    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])
            
    total = []
    
    coordinate = np.stack((h,k,l))
            
    for op in symops:
                
        transformed = symmetry.evaluate([op], coordinate, translate=False)
                                        
        total.append(transformed.T.tolist())
                                                              
    total = np.array(total)
    sort = np.array(total)
    
    for i in range(coordinate.shape[1]):
        
        sort[:,i,:] = total[np.lexsort(total[:,i,:].T),i,:]
        
    array = np.hstack(sort)
                        
    h, k, l = array[:,:3].T
    
    Q = 2*np.pi*np.sqrt((B[0,0]*h+B[0,1]*k+B[0,2]*l)**2+\
                        (B[1,0]*h+B[1,1]*k+B[1,2]*l)**2+\
                        (B[2,0]*h+B[2,1]*k+B[2,2]*l)**2)
    
    mask = (Q >= Q_min) & (Q <= Q_max)
    
    sort = np.argsort(Q[mask])
    
    Q = Q[mask][sort]
    array = array[mask][sort]

    distance = np.round(np.round(Q/tol,1)).astype(int)
    metric =  np.vstack((distance, array.T)).T
                                
    symmetries, indices, inverses = np.unique(metric,
                                              axis=0, 
                                              return_index=True, 
                                              return_inverse=True)
    
    distance = distance[indices]
    Q = Q[indices]
    
    #length, ind = np.unique(distance, return_index=True)
    
    m = np.zeros(symmetries.shape[0], dtype=np.int)
    
    total = []
    
    for i in range(symmetries.shape[0]):
        
        array = symmetries[i,1:].reshape(symmetries.shape[1] // 3,3)
        
        vectors, indices, inverses = np.unique(array,
                                               axis=0, 
                                               return_index=True, 
                                               return_inverse=True)
        
        m[i] = vectors.shape[0]
        
        total.append(vectors.tolist())
        
    total = np.vstack(total)
    index = np.cumsum(m)
    
    #index = np.append(np.append(0, index[ind]), index[-1])
    h, k, l = total.T

    return Q, h, k, l, index, m

def symmetrize(arrays, 
               dx, 
               dy, 
               dz, 
               ion,
               A, 
               laue,
               tol=1e-4):
        
    arrays = np.hstack(list((arrays,)))
    
    if (arrays.ndim == 1):
        
        arrays = arrays[np.newaxis,:]
            
    M = arrays.shape[0]
        
    symops = np.array(symmetry.laue(laue))
        
    symops = np.roll(symops, -np.argwhere(symops==u'x,y,z')[0][0])     
        
    N = arrays.shape[1]
    
    A_inv = np.linalg.inv(A)
        
    total = []
    
    arr = []
    
    pairs = []
    pair_labels = []
    
    ions, ion_labels = np.unique(ion, return_inverse=True)
                        
    for n in range(N):
        
        x = A_inv[0,0]*dx[n]+A_inv[0,1]*dy[n]+A_inv[0,2]*dz[n]
        y = A_inv[1,0]*dx[n]+A_inv[1,1]*dy[n]+A_inv[1,2]*dz[n]
        z = A_inv[2,0]*dx[n]+A_inv[2,1]*dy[n]+A_inv[2,2]*dz[n]
    
        # x = np.round(x,4)
        # y = np.round(y,4)
        # z = np.round(z,4)        

        displacement = []
        
        for symop in symops:
                        
            transformed = symmetry.evaluate(symop, [x,y,z], translate=False)
                    
            displacement.append(transformed)
        
        # inversion 
        # transformed = symmetry.evaluate([u'-x, -y, -z'],
        #                                 [x,y,z], 
        #                                 translate=False)
                
        displacement.append(transformed)

        symmetries = np.unique(np.array(displacement), axis=0)
                                    
        total.append(symmetries)
        
        arr.append(np.tile(arrays[:,n], symmetries.shape[0]))
        
        pairs.append(np.tile(ion[n], symmetries.shape[0]))
        pair_labels.append(np.tile(ion_labels[n], symmetries.shape[0]))
     
    total = np.vstack(np.array(total, dtype=object)).astype(float)
        
    arr = np.hstack(np.array(arr, dtype=object)).astype(float)
    
    arr = arr.reshape(arr.shape[0] // M, M)
    
    pairs = np.hstack(np.array(pairs, dtype=object))   
    pair_labels = np.hstack(np.array(pair_labels, dtype=object)).astype(int)
                
    metric = np.vstack((np.round(np.round( \
                        total.astype(float).T/tol,1)).astype(int), \
                        pair_labels)).T
                                         
    sort = np.lexsort(np.fliplr(metric).T)
            
    arr = arr[sort]
    
    pairs = pairs[sort]
            
    unique, indices, counts = np.unique(metric[sort],
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
                
    search = np.append(indices,len(arr))
    
    D = unique.shape[0]
    
    u_symm = total[sort][indices,0]
    v_symm = total[sort][indices,1]
    w_symm = total[sort][indices,2]
    ion_symm = pairs[indices]
    
    arrays_ave = np.zeros((M,D))
    
    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):
                arrays_ave[i,r] += arr[s,i]/counts[r]
            
    dx_symm = (A[0,0]*u_symm+A[0,1]*v_symm+A[0,2]*w_symm).astype(float)
    dy_symm = (A[1,0]*u_symm+A[1,1]*v_symm+A[1,2]*w_symm).astype(float)
    dz_symm = (A[2,0]*u_symm+A[2,1]*v_symm+A[2,2]*w_symm).astype(float)

    arrays_ave = arrays_ave.flatten()
        
    output = tuple(np.split(arrays_ave, M))
    output = (*output, dx_symm, dy_symm, dz_symm, ion_symm)
        
    return output

def average(arrays, d, tol=1e-4):
    
    arrays = np.hstack(list((arrays,)))
    
    if (arrays.ndim == 1):
        
        arrays = arrays[np.newaxis,:]
    
    M = arrays.shape[0]
    
    metric = (np.round(np.round(d/tol,1))).astype(int)
    
    sort = np.argsort(metric)

    metric = metric[sort]
    d = d[sort]
    arrays = arrays[:,sort]
    
    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)
    
    search = np.append(indices,len(d))
    
    D = unique.shape[0]
    
    arrays_ave = np.zeros((M,D))
     
    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):
                arrays_ave[i,r] += arrays[i][s]/counts[r]
            
    d_ave = d[indices]
    
    arrays_ave = arrays_ave.flatten()
        
    output = tuple(np.split(arrays_ave, M))
    output = (*output, d_ave)
        
    return output

def average3d(arrays, dx, dy, dz, tol=1e-4):
    
    arrays = np.hstack(list((arrays,)))
    
    if (arrays.ndim == 1):
        
        arrays = arrays[np.newaxis,:]
    
    M = arrays.shape[0]
    
    distance = np.stack((dx,dy,dz)).T
    
    metric = (np.round(np.round(distance.astype(float)/tol,1))).astype(int)
    
    sort = np.lexsort(np.fliplr(metric).T)
    
    metric = metric[sort]
    dx = dx[sort]
    dy = dy[sort]
    dz = dz[sort]
    arrays = arrays[:,sort]
    
    unique, indices, counts = np.unique(metric, 
                                        axis=0,
                                        return_index=True, 
                                        return_counts=True)    
    
    search = np.append(indices,len(distance))
    
    D = unique.shape[0]
    
    arrays_ave = np.zeros((M,D))
    
    for i in range(M):
        for r in range(D):
            for s in range(search[r],search[r+1]):                
                arrays_ave[i,r] += arrays[i][s]/counts[r]
            
    dx_ave = dx[indices].astype(float)
    dy_ave = dy[indices].astype(float)
    dz_ave = dz[indices].astype(float)
        
    arrays_ave = arrays_ave.flatten()
    
    output = tuple(np.split(arrays_ave, M))
    output = (*output, dx_ave, dy_ave, dz_ave)
    
    return output

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

def matrices(a, b, c, alpha, beta, gamma):
    
    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, 
                                                   b, 
                                                   c, 
                                                   alpha, 
                                                   beta, 
                                                   gamma)
    
    A = np.array([[a, b*np.cos(gamma),  c*np.cos(beta)],
                  [0, b*np.sin(gamma), -c*np.sin(beta)*np.cos(alpha_)],
                  [0, 0,                1/c_]])
    
    B = np.array([[a_, b_*np.cos(gamma_),  c_*np.cos(beta_)],
                  [0,  b_*np.sin(gamma_), -c_*np.sin(beta_)*np.cos(alpha)],
                  [0,  0,                  1/c]])
    
    R = np.dot(np.linalg.inv(A).T, np.linalg.inv(B))
                                                      
    return A, B, R

def orthogonalized(a, b, c, alpha, beta, gamma):
    
    a_, b_, c_, alpha_, beta_, gamma_ = reciprocal(a, 
                                                   b, 
                                                   c, 
                                                   alpha, 
                                                   beta, 
                                                   gamma)
    
    A = np.array([[a, b*np.cos(gamma),  c*np.cos(beta)],
                  [0, b*np.sin(gamma), -c*np.sin(beta)*np.cos(alpha_)],
                  [0, 0,                1/c_]])
    
    L = np.array([[a,0,0],[0,b,0],[0,0,c]])
    L_ = np.array([[a_,0,0],[0,b_,0],[0,0,c_]])
        
    return np.dot(A, np.linalg.inv(L)), np.dot(A, L_)

def transform(p, q, r, U):
        
    return U[0,0]*p+U[0,1]*q+U[0,2]*r,\
           U[1,0]*p+U[1,1]*q+U[1,2]*r,\
           U[2,0]*p+U[2,1]*q+U[2,2]*r
            
def supercell(atm,
              occ,
              disp,
              mom,
              u, 
              v, 
              w,
              nu,
              nv,
              nw,
              name,
              folder=None,
              filename=None):
    
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
                    combine.append(symmetry.binary(
                                   ','.join(symop.split(',')[:3]),
                                   ','.join(add_symop.split(',')[:3])))
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
            
            C, D = orthogonalized(a,
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
        
def disordered(delta,
               Ux,
               Uy, 
               Uz, 
               Sx,
               Sy,
               Sz,
               rx, 
               ry, 
               rz,
               nu,
               nv,
               nw,
               atm, 
               A,
               name,
               folder=None,
               filename=None,
               ulim=[0,1], 
               vlim=[0,1], 
               wlim=[0,1]):
    
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
                    combine.append(symmetry.binary(
                                   ','.join(symop.split(',')[:3]),
                                   ','.join(add_symop.split(',')[:3])))
            symops = combine
        
        for symop in symops:
            cb.RemoveLoopItem(symop)     
        
        atomic_sites = cb.GetLoopNames('_atom_site_label')
        
        for asite in atomic_sites:
            cb.RemoveLoopItem(asite)
              
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
        
        u = A_inv[0,0]*ux+A_inv[0,1]*uy+A_inv[0,2]*uz
        v = A_inv[1,0]*ux+A_inv[1,1]*uy+A_inv[1,2]*uz
        w = A_inv[2,0]*ux+A_inv[2,1]*uy+A_inv[2,2]*uz
        
        mx = np.round(Sx[mask],4)
        my = np.round(Sy[mask],4)
        mz = np.round(Sz[mask],4)
        
        mu, mv, mw = transform(mx, my, mz, A_inv)
        
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
            
        cb.AddLoopItem((['_space_group_symop_operation_xyz'],
                         ['']))
        
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
        
def periodic(u, v, w, centers, neighbors, A, nu, nv, nw):
    
    n_atm = u.shape[0]
    
    img = np.array([-1, 0, 1])
    
    img_u, img_v, img_w = np.meshgrid(img, img, img, indexing='ij')
    
    img_u = img_u.flatten()
    img_v = img_v.flatten()
    img_w = img_w.flatten()
    
    U = (u+img_u[:,np.newaxis]).flatten()
    V = (v+img_v[:,np.newaxis]).flatten()
    W = (w+img_w[:,np.newaxis]).flatten()
    
    rx = A[0,0]*U+A[0,1]*V+A[0,2]*W
    ry = A[1,0]*U+A[1,1]*V+A[1,2]*W
    rz = A[2,0]*U+A[2,1]*V+A[2,2]*W
        
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
   
    d, pairs = tree.query(points, 1+neighbors)
    
    mask = np.tile(centers, 27)
        
    indices = np.mod(pairs[mask], n_atm)
    
    sort = np.sort(indices[:,1:], axis=1)
    
    indices[:,1:] = sort
    
    #sort = np.argsort(indices, axis=1)
        
    primitive, ind = np.unique(indices, return_index=True, axis=0)
    
    # ---
    
    bound = np.min([A[0,0],A[1,1],A[2,2]])/2
    
    cutoff = np.all(d[ind].reshape(primitive.shape) < bound, axis=1)
    
    primitive = primitive[cutoff]
    ind = ind[cutoff]
    
    d = d[ind]
   
    while (primitive.shape[0] > centers.sum()):
        
        maximum = np.max(d[:,-1])
        
        mask = np.isclose(d[:,-1],maximum)
        
        primitive = primitive[~mask]
        d = d[~mask]
    
    # ---
    
    n = np.sum(centers)
    
    img_ind_u = np.zeros(primitive.shape, dtype=np.int)
    img_ind_v = np.zeros(primitive.shape, dtype=np.int)
    img_ind_w = np.zeros(primitive.shape, dtype=np.int)
    
    diff_u = (u[primitive].T-u[primitive[:,0]]).T
    diff_v = (v[primitive].T-v[primitive[:,0]]).T
    diff_w = (w[primitive].T-w[primitive[:,0]]).T
    
    img_ind_u[diff_u > 0.5] = -1
    img_ind_u[diff_u < -0.5] = 1
    
    img_ind_v[diff_v > 0.5] = -1
    img_ind_v[diff_v < -0.5] = 1
    
    img_ind_w[diff_w > 0.5] = -1
    img_ind_w[diff_w < -0.5] = 1
    
    i, j, k = np.meshgrid(np.arange(nu), 
                          np.arange(nv), 
                          np.arange(nw), indexing='ij')
    
    i = i.flatten()
    j = j.flatten()
    k = k.flatten()
    
    I = np.mod(img_ind_u[:,:,np.newaxis]+i, nu)
    J = np.mod(img_ind_v[:,:,np.newaxis]+j, nv)
    K = np.mod(img_ind_w[:,:,np.newaxis]+k, nw)
    
    structure = primitive[:,:,np.newaxis]+n_atm*(K+nw*(J+nv*I))
    
    return np.rollaxis(structure, -1).reshape(nu*nv*nw*n,1+neighbors), \
           primitive
           
def pairs(u, v, w, neighbors, A, nu, nv, nw):
    
    n_atm = u.shape[0]
    
    img = np.array([-1, 0, 1])
    
    img_u, img_v, img_w = np.meshgrid(img, img, img, indexing='ij')
    
    img_u = img_u.flatten()
    img_v = img_v.flatten()
    img_w = img_w.flatten()
    
    offset = np.zeros(n_atm, dtype=np.int)
    
    U = (u+img_u[:,np.newaxis]).flatten()
    V = (v+img_v[:,np.newaxis]).flatten()
    W = (w+img_w[:,np.newaxis]).flatten()
    
    I = (offset+img_u[:,np.newaxis]).flatten()
    J = (offset+img_v[:,np.newaxis]).flatten()
    K = (offset+img_w[:,np.newaxis]).flatten()
    
    rx = A[0,0]*U+A[0,1]*V+A[0,2]*W
    ry = A[1,0]*U+A[1,1]*V+A[1,2]*W
    rz = A[2,0]*U+A[2,1]*V+A[2,2]*W
    
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    d, pairs = tree.query(points, 1+neighbors)
    
    indices = pairs[np.arange(n_atm)+n_atm*13,1:]
    
    atm_ind = np.mod(indices, n_atm)
    
    img_ind_i, img_ind_j, img_ind_k = I[indices], J[indices], K[indices]
    
    dist = d[np.arange(n_atm)+n_atm*13,1:]
    
    dx = rx[indices]-rx[np.arange(n_atm)+n_atm*13,np.newaxis]
    dy = ry[indices]-ry[np.arange(n_atm)+n_atm*13,np.newaxis]
    dz = rz[indices]-rz[np.arange(n_atm)+n_atm*13,np.newaxis]
    
    return atm_ind, img_ind_i, img_ind_j, img_ind_k, dist, dx, dy, dz

def nearest(u, v, w, A, nu, nv, nw):
    
    n_atm = u.shape[0]
    
    img = np.array([-1, 0, 1])
    
    img_u, img_v, img_w = np.meshgrid(img, img, img, indexing='ij')
    
    img_u = img_u.flatten()
    img_v = img_v.flatten()
    img_w = img_w.flatten()
        
    U = (u+img_u[:,np.newaxis]).flatten()
    V = (v+img_v[:,np.newaxis]).flatten()
    W = (w+img_w[:,np.newaxis]).flatten()
    
    rx = A[0,0]*U+A[0,1]*V+A[0,2]*W
    ry = A[1,0]*U+A[1,1]*V+A[1,2]*W
    rz = A[2,0]*U+A[2,1]*V+A[2,2]*W
    
    points = np.column_stack((rx, ry, rz))
    tree = spatial.cKDTree(points)
    
    d, pairs = tree.query(points, k=2)
    
    indices = pairs[np.arange(n_atm)+n_atm*13,1:]
        
    dist = d[np.arange(n_atm)+n_atm*13,1:]
    
    dx = rx[indices]-rx[np.arange(n_atm)+n_atm*13,np.newaxis]
    dy = ry[indices]-ry[np.arange(n_atm)+n_atm*13,np.newaxis]
    dz = rz[indices]-rz[np.arange(n_atm)+n_atm*13,np.newaxis]
    
    return dist.flatten(), dx.flatten(), dy.flatten(), dz.flatten()