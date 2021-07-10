#!/usr/bin/env/python3

import sys
import numpy as np

class Presenter:

    def __init__(self, model, view):
        
        self.model = model
        self.view = view
        
        self.fname = ''
        
        self.view.new_triggered(self.new_application)
        self.view.save_as_triggered(self.save_as_application)
        self.view.save_triggered(self.save_application)
        self.view.open_triggered(self.load_application)
        self.view.exit_triggered(self.exit_application)

        self.view.finished_editing_nu(self.supercell_n)
        self.view.finished_editing_nv(self.supercell_n)
        self.view.finished_editing_nw(self.supercell_n)
        
        self.view.index_changed_type(self.change_type)
        self.view.index_changed_parameters(self.change_parameters)
        
        self.view.button_clicked_CIF(self.load_CIF)
        # self.view.button_clicked_NXS(self.load_NXS)
        self.view.select_site(self.select_highlight)
        
    def new_application(self):
        
        self.view.clear_application()
        
    def save_as_application(self):
        
        self.fname = self.view.open_dialog_save()
        self.file_save(self.fname)
        
    def save_application(self):
                
        if self.fname:
            self.file_save(self.fname)
        else:
            self.save_as_application()
            
    def file_save(self, fname):
        
        fname = self.fname
        if fname:
            if not fname.endswith('.ini'): fname += '.ini'
            self.view.save_widgets(fname)
                        
    def load_application(self):
                
        self.fname = self.view.open_dialog_load()
        if self.fname:
            self.view.clear_application()
            self.view.load_widgets(self.fname)
            if (self.view.get_atom_site_table_col_count() > 0):
                atom = self.view.get_atom_combo()
                ion = self.view.get_ion_combo()
                self.connect_table_signals()
                self.view.set_atom_combo(atom)
                self.view.set_ion_combo(ion) 
                
    def exit_application(self):
        
        if self.view.close_application():
            sys.exit()

    def supercell_n(self):
        
        n_atm = self.view.get_n_atm()
        if (n_atm is not None):
            nu = self.view.get_nu()
            nv = self.view.get_nv()
            nw = self.view.get_nw()
            n = self.model.supercell_size(n_atm, nu, nv, nw)
            self.view.set_n(n)
            
    def change_type(self):
        
        n_cols = self.view.get_atom_site_table_col_count()
        
        if (self.view.get_type() == 'Neutron'):
            self.view.add_item_magnetic()
            if (n_cols > 0):
                atm = self.view.get_ion_combo()
                bc_symbols = self.model.get_neutron_scattering_length_keys()
                self.view.add_atom_combo(bc_symbols)
                mag_symbols = self.model.get_magnetic_form_factor_keys()
                self.view.add_ion_combo(mag_symbols)
                self.view.set_atom_combo(atm)
                self.populate_atoms()
        else:
            self.view.remove_item_magnetic()
            if (n_cols > 0):
                atm = self.view.get_atom_combo()
                X_symbols = self.model.get_xray_form_factor_keys()
                self.view.add_ion_combo(X_symbols)
                self.view.set_ion_combo(atm)
                self.populate_ions()
                
        self.change_parameters()
        self.view.index_changed_atom(self.populate_atoms)
        self.view.index_changed_ion(self.populate_ions)

    def populate_atoms(self):
        
        n_rows = self.view.get_atom_site_table_row_count()
        sites = self.view.get_every_site().astype(int)-1
                
        isotope = self.view.get_atom_combo()
        atm = self.model.iso_symbols(isotope)
        nuc = self.model.remove_symbols(isotope)
        
        atoms = self.view.get_every_atom().astype('<U2')
        nuclei = self.view.get_every_isotope().astype('<U3')
        ions = self.view.get_every_ion().astype('<U2')
        
        mag_charges = self.model.get_magnetic_form_factor_keys()
        mag_atoms = self.model.ion_symbols(mag_charges)
                
        for row in range(n_rows):
            rows = np.argwhere(sites == row)
            if (len(rows) > 0):
                row_range = [rows.min(), rows.max()+1]
                atoms[row_range[0]:row_range[1]] = atm[row]
                nuclei[row_range[0]:row_range[1]] = nuc[row]
                mag_symbols = [mag_charges[i] for i, \
                               mag_atom in enumerate(mag_atoms) \
                               if mag_atom in atm[row]]
                self.view.add_mag_ion_combo(row, mag_symbols)
                if (len(mag_symbols) > 0):
                    mag_ions = self.model.remove_symbols(mag_symbols)
                    ions[row_range[0]:row_range[1]] = mag_ions[0]
                    
        self.view.set_unit_cell_atom(atoms)
        self.view.set_unit_cell_isotope(nuclei)
        self.view.set_unit_cell_ion(ions)

        self.view.format_unit_cell_table_col(self.view.unit_table['atom'])
        self.view.format_unit_cell_table_col(self.view.unit_table['isotope'])
        self.view.format_unit_cell_table_col(self.view.unit_table['ion'])
        self.view.index_changed_ion(self.populate_ions)
    
    def populate_ions(self):
        
        n_rows = self.view.get_atom_site_table_row_count()
        sites = self.view.get_every_site().astype(int)-1
                
        charge = self.view.get_ion_combo()
        atm = self.model.iso_symbols(charge)
        ion = self.model.remove_symbols(charge)
        
        atoms = self.view.get_every_atom().astype('<U2')
        ions = self.view.get_every_ion().astype('<U2')
                
        for row in range(n_rows):
            rows = np.argwhere(sites == row)
            if (len(rows) > 0):
                row_range = [rows.min(), rows.max()+1]
                if (atm[row] != ''):
                    atoms[row_range[0]:row_range[1]] = atm[row]
                ions[row_range[0]:row_range[1]+1] = ion[row]
                
        self.view.set_unit_cell_atom(atoms)
        self.view.set_unit_cell_ion(ions)
                
        self.view.format_unit_cell_table_col(self.view.unit_table['atom'])
        self.view.format_unit_cell_table_col(self.view.unit_table['ion'])
        
    def update_atom_site_table(self, item):
        
        row, col, text = item.row(), item.column(), item.text()
        sites = self.view.get_every_site().astype(int)-1
        rows = np.argwhere(sites == row)
        if (len(rows) > 0):
            row_range = [rows.min(), rows.max()+1]
            if (col == self.view.atom_table['occupancy']):
                self.update_occupancy(row_range, text)
            elif (col == self.view.atom_table['U11']):
                self.update_U(row_range, text, 0)
            elif (col == self.view.atom_table['U22']):
                self.update_U(row_range, text, 1)
            elif (col == self.view.atom_table['U33']):
                self.update_U(row_range, text, 2)
            elif (col == self.view.atom_table['U23']):
                self.update_U(row_range, text, 3)
            elif (col == self.view.atom_table['U13']):
                self.update_U(row_range, text, 4)
            elif (col == self.view.atom_table['U12']):
                self.update_U(row_range, text, 5)                
            elif (col == self.view.atom_table['mu1']):
                self.update_mu(row_range, text, 0)
            elif (col == self.view.atom_table['mu2']):
                self.update_mu(row_range, text, 1)
            elif (col == self.view.atom_table['mu3']):
                self.update_mu(row_range, text, 2)
            elif (col == self.view.atom_table['g']):
                self.update_g(row_range, text)
            elif (col == self.view.atom_table['u']):
                self.update_uvw(row_range, text, 0)
            elif (col == self.view.atom_table['v']):
                self.update_uvw(row_range, text, 1)
            elif (col == self.view.atom_table['w']):
                self.update_uvw(row_range, text, 2)
                
    def update_occupancy(self, row_range, occ):
        
        occupancy = self.view.get_every_occupancy().astype('<U6')          
        occupancy[row_range[0]:row_range[1]] = occ
        self.view.set_unit_cell_occupancy(occupancy)  
        self.view.format_unit_cell_table_col(self.view.unit_table['occupancy'])
        
    def update_mu(self, row_range, moment, comp):
        
        mu1 = self.view.get_every_mu1().astype(float)
        mu2 = self.view.get_every_mu2().astype(float)
        mu3 = self.view.get_every_mu3().astype(float)
        mag_op = self.view.get_every_magnetic_operator()
        for row in range(*row_range):
            mu = [mu1[row], mu2[row], mu3[row]]
            mu = self.model.magnetic_symmetry(mag_op[row], mu)
            mu[comp] = float(moment)
            mu = self.model.magnetic_symmetry(mag_op[row], mu)
            mu1[row], mu2[row], mu3[row] = mu
        self.view.set_unit_cell_mu1(mu1)
        self.view.set_unit_cell_mu2(mu2)
        self.view.set_unit_cell_mu3(mu3)
        self.view.format_unit_cell_table_col(self.view.unit_table['mu1'])
        self.view.format_unit_cell_table_col(self.view.unit_table['mu2'])
        self.view.format_unit_cell_table_col(self.view.unit_table['mu3'])
        self.update_moment_magnitudes()
        
    def update_moment_magnitudes(self):
        
        mu1 = self.view.get_every_mu1().astype(float)
        mu2 = self.view.get_every_mu2().astype(float)
        mu3 = self.view.get_every_mu3().astype(float)
        
        a, b, c, alpha, beta, gamma = self.view.get_lattice_parameters()
        A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                    alpha, beta, gamma)
                
        mu = self.model.magnetic_moments(mu1, mu2, mu3, C)
        mu = np.round(mu, 4)

        self.view.set_unit_cell_mu(mu)
        self.view.format_unit_cell_table_col(self.view.unit_table['mu'])
        
    def update_g(self, row_range, constant):
        
        g = self.view.get_every_g().astype('<U6')          
        g[row_range[0]:row_range[1]] = constant
        self.view.set_unit_cell_g(g)  
        self.view.format_unit_cell_table_col(self.view.unit_table['g'])
        
    def update_U(self, row_range, parameter, comp):
 
        U11 = self.view.get_every_U11().astype('<U6')
        U22 = self.view.get_every_U22().astype('<U6')
        U33 = self.view.get_every_U33().astype('<U6')
        U23 = self.view.get_every_U23().astype('<U6')
        U13 = self.view.get_every_U13().astype('<U6')
        U12 = self.view.get_every_U12().astype('<U6')

        if (comp == 0): U11[row_range[0]:row_range[1]] = parameter
        if (comp == 1): U22[row_range[0]:row_range[1]] = parameter
        if (comp == 2): U33[row_range[0]:row_range[1]] = parameter
        if (comp == 3): U23[row_range[0]:row_range[1]] = parameter
        if (comp == 4): U13[row_range[0]:row_range[1]] = parameter
        if (comp == 5): U12[row_range[0]:row_range[1]] = parameter
        
        self.view.set_unit_cell_U11(U11)
        self.view.set_unit_cell_U22(U22)
        self.view.set_unit_cell_U33(U33)
        self.view.set_unit_cell_U23(U23)
        self.view.set_unit_cell_U13(U13)
        self.view.set_unit_cell_U12(U12)
        self.view.format_unit_cell_table_col(self.view.unit_table['U11'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U22'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U33'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U23'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U13'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U12'])
        self.update_Up()
        
    def update_Up(self):
 
        U11 = self.view.get_every_U11().astype(float)
        U22 = self.view.get_every_U22().astype(float)
        U33 = self.view.get_every_U33().astype(float)
        U23 = self.view.get_every_U23().astype(float)
        U13 = self.view.get_every_U13().astype(float)
        U12 = self.view.get_every_U12().astype(float)
        
        a, b, c, alpha, beta, gamma = self.view.get_lattice_parameters()
        A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                    alpha, beta, gamma)
        
        Uiso, U1, U2, U3 = self.model.atomic_displacement_parameters(U11,
                                                                     U22, 
                                                                     U33, 
                                                                     U23, 
                                                                     U13, 
                                                                     U12, 
                                                                     D)
        Uiso = np.round(Uiso, 4)
        U1 = np.round(U1, 4)
        U2 = np.round(U2, 4)
        U3 = np.round(U3, 4)
                    
        self.view.set_unit_cell_Uiso(Uiso)
        self.view.set_unit_cell_U1(U1)
        self.view.set_unit_cell_U2(U2)
        self.view.set_unit_cell_U3(U3)
        self.view.format_unit_cell_table_col(self.view.unit_table['Uiso'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U1'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U2'])
        self.view.format_unit_cell_table_col(self.view.unit_table['U3'])
        
    def update_uvw(self, row_range, coordinate, comp):
        
        u = self.view.get_every_u().astype(float)
        v = self.view.get_every_v().astype(float)
        w = self.view.get_every_w().astype(float)
        op = self.view.get_every_operator()
        for row in range(*row_range):
            coord = [u[row], v[row], w[row]]
            coord = self.model.reverse_symmetry(op[row], coord)
            coord[comp] = float(coordinate)
            coord = self.model.symmetry(op[row], coord)
            u[row], v[row], w[row] = coord
        self.view.set_unit_cell_u(u)
        self.view.set_unit_cell_v(v)
        self.view.set_unit_cell_w(w)
        self.view.format_unit_cell_table_col(self.view.unit_table['u'])
        self.view.format_unit_cell_table_col(self.view.unit_table['v'])
        self.view.format_unit_cell_table_col(self.view.unit_table['w'])
            
    def change_parameters(self):
        
        if (self.view.get_atom_site_table_col_count() > 0):
            self.view.show_atom_site_table_cols()
            self.view.show_unit_cell_table_cols()
            self.view.clear_atom_site_table_selection()
            self.view.clear_unit_cell_table_selection()
    
    def select_highlight(self, item):
        
        row, col = item.row(), item.column()
        sites = self.view.get_every_site().astype(int)-1
        rows = np.argwhere(sites == row)
        n_rows = len(rows)
        if (n_rows > 0):
            row_range = [rows.min(), rows.max()]
            self.view.highlight_atoms(row_range, self.view.unit_site_col(col))
            
    def add_remove_atoms(self):
        
        n_atm = self.view.change_site_check()
        nu = self.view.get_nu()
        nv = self.view.get_nv()
        nw = self.view.get_nw()
        self.view.set_n_atm(n_atm)
        self.view.set_n(self.model.supercell_size(n_atm, nu, nv, nw))
        
    def connect_table_signals(self):
            
        self.change_type()
        
        self.view.format_atom_site_table()
        self.view.format_unit_cell_table()
            
        self.view.check_clicked_site(self.add_remove_atoms)
        self.view.item_changed_atom_site_table(self.update_atom_site_table)
        
    def load_CIF(self):
        
        name = self.view.open_dialog_cif()
        
        if name:
            
            self.view.create_unit_cell_table(0)
            self.view.create_atom_site_table(0)
                        
            folder, filename = name.rsplit('/', 1)
            folder += '/'
            
            parameters = self.model.load_lattice_parameters(folder, filename)
            a, b, c, alpha, beta, gamma = parameters
            
            lat = self.model.find_lattice(a, b, c, alpha, beta, gamma)
            
            self.view.set_lattice(lat)
            
            self.view.set_lattice_parameters(a, b, c, alpha, beta, gamma)
            
            group, hm = self.model.load_space_group(folder, filename)
            
            self.view.set_space_group(group, hm)
                
            u, \
            v, \
            w, \
            occupancy, \
            displacement, \
            moment, \
            site, \
            op, \
            mag_op, \
            atm, \
            n_atm = self.model.load_unit_cell(folder, filename)
            
            A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                        alpha, beta, gamma)
            
            if (displacement.shape[1] == 6):
                U11, U22, U33, U23, U13, U12 = np.round(displacement.T, 4)
            else:
                Uiso = np.round(displacement.flatten(), 4)                
                uiso = np.dot(np.linalg.inv(D), np.linalg.inv(D.T))
                U11, U22, U33 = Uiso*uiso[0,0], Uiso*uiso[1,1], Uiso*uiso[2,2]
                U23, U13, U12 = Uiso*uiso[1,2], Uiso*uiso[0,2], Uiso*uiso[0,1]
            
            mu1, mu2, mu3 = np.round(moment.T, 4)
            
            Uiso, U1, U2, U3 = self.model.atomic_displacement_parameters(U11,
                                                                         U22, 
                                                                         U33, 
                                                                         U23, 
                                                                         U13, 
                                                                         U12, 
                                                                         D)

            mu = self.model.magnetic_moments(mu1, mu2, mu3, C)
            
            Uiso = np.round(Uiso, 4)
            U1 = np.round(U1, 4)
            U2 = np.round(U2, 4)
            U3 = np.round(U3, 4)
            
            mu = np.round(mu, 4)
          
            self.view.set_n_atm(n_atm)
                        
            uni, ind, inv = np.unique(site, 
                                      return_index=True, 
                                      return_inverse=True)
            
            n_sites = ind.size
            
            g = np.full(n_atm, 2.0)
            
            empty = np.full(n_atm, '-')
            
            self.view.create_atom_site_table(n_sites)
            self.view.show_atom_site_table_cols()
            
            self.view.set_atom_site_occupancy(occupancy[ind])            
            self.view.set_atom_site_U11(U11[ind]) 
            self.view.set_atom_site_U22(U22[ind]) 
            self.view.set_atom_site_U33(U33[ind]) 
            self.view.set_atom_site_U23(U23[ind]) 
            self.view.set_atom_site_U13(U13[ind]) 
            self.view.set_atom_site_U12(U12[ind]) 
            self.view.set_atom_site_U11(U11[ind]) 
            self.view.set_atom_site_mu1(mu1[ind]) 
            self.view.set_atom_site_mu2(mu2[ind]) 
            self.view.set_atom_site_mu3(mu3[ind]) 
            self.view.set_atom_site_g(g[ind]) 
            self.view.set_atom_site_u(u[ind]) 
            self.view.set_atom_site_v(v[ind])
            self.view.set_atom_site_w(w[ind])
            self.view.add_site_check()
            self.view.format_atom_site_table()
            
            self.view.create_unit_cell_table(n_atm)
            self.view.show_unit_cell_table_cols()
            
            self.view.set_unit_cell_site(site+1)  
            self.view.set_unit_cell_ion(empty)
            self.view.set_unit_cell_isotope(empty)
            self.view.set_unit_cell_occupancy(occupancy)
            self.view.set_unit_cell_Uiso(Uiso) 
            self.view.set_unit_cell_U11(U11) 
            self.view.set_unit_cell_U22(U22) 
            self.view.set_unit_cell_U33(U33) 
            self.view.set_unit_cell_U23(U23) 
            self.view.set_unit_cell_U13(U13) 
            self.view.set_unit_cell_U12(U12) 
            self.view.set_unit_cell_U1(U1) 
            self.view.set_unit_cell_U2(U2) 
            self.view.set_unit_cell_U3(U3) 
            self.view.set_unit_cell_mu(mu) 
            self.view.set_unit_cell_mu1(mu1) 
            self.view.set_unit_cell_mu2(mu2) 
            self.view.set_unit_cell_mu3(mu3) 
            self.view.set_unit_cell_g(g) 
            self.view.set_unit_cell_u(u) 
            self.view.set_unit_cell_v(v)
            self.view.set_unit_cell_w(w)
            self.view.set_unit_cell_operator(op)
            self.view.set_unit_cell_magnetic_operator(mag_op)
            self.view.format_unit_cell_table()
            
            self.view.set_atom_combo(atm[ind])
            self.view.set_ion_combo(atm[ind])
                
            nu = self.view.get_nu()
            nv = self.view.get_nv()
            nw = self.view.get_nw()
            
            self.view.set_n(self.model.supercell_size(n_atm, nu, nv, nw))
            
            self.connect_table_signals()
        
    def load_NXS(self):

        if (self.view.get_atom_site_table_col_count() > 0):

            name = self.view.open_dialog_nxs()
            
            if name:
                
                signal, \
                error_sq, \
                h_range,\
                k_range, \
                l_range, \
                nh, \
                nk, \
                nl = self.model.data(name)
                