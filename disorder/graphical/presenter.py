#!/usr/bin/env/python3

import sys

import numpy as np

from disorder.graphical import plots

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
        self.view.button_clicked_NXS(self.load_NXS)
        self.view.select_site(self.select_highlight)
        
        self.view.clicked_batch(self.check_batch)
        self.view.clicked_batch_1d(self.check_batch_1d)
        self.view.clicked_batch_3d(self.check_batch_3d)
        self.view.clicked_batch_calc(self.check_batch_calc)
        
        self.view.clicked_disorder_mag(self.disorder_check_mag)
        self.view.clicked_disorder_occ(self.disorder_check_occ)
        self.view.clicked_disorder_dis(self.disorder_check_dis)
        
        self.view.clicked_run(self.run_refinement)
        self.view.clicked_stop(self.stop_refinement)
        self.view.clicked_reset(self.reset_refinement)
        
        self.view.index_changed_correlations_1d(self.change_type_1d)
        self.view.index_changed_correlations_3d(self.change_type_3d)
        
        self.view.finished_editing_min_ref(self.draw_plot_ref)
        self.view.finished_editing_max_ref(self.draw_plot_ref)
        self.view.finished_editing_slice(self.draw_plot_ref)
        
        self.view.index_changed_slice_hkl(self.redraw_plot_ref)
        self.view.index_changed_plot_ref(self.redraw_plot_ref)
        self.view.index_changed_norm_ref(self.redraw_plot_ref)
        
        self.view.index_changed_plot_top_chi_sq(self.draw_plot_chi_sq)
        self.view.index_changed_plot_bottom_chi_sq(self.draw_plot_chi_sq)
        
        self.threadpool = self.view.create_threadpool()
        
        self.allocated = False
        self.iteration = 0
        
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
            self.view.enable_disorder_mag(True)
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
            self.view.enable_disorder_mag(False)
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
        sites = self.view.get_every_site()-1
                
        isotope = self.view.get_atom_combo()
        atm = self.model.iso_symbols(isotope)
        nuc = self.model.remove_symbols(isotope)
        
        atoms = self.view.get_every_atom().astype('<U3')
        nuclei = self.view.get_every_isotope().astype('<U3')
        ions = self.view.get_every_ion().astype('<U3')
        
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
        sites = self.view.get_every_site()-1
                
        charge = self.view.get_ion_combo()
        atm = self.model.ion_symbols(charge)
        ion = self.model.remove_symbols(charge)
            
        atoms = self.view.get_every_atom().astype('<U3')
        ions = self.view.get_every_ion().astype('<U3')
                        
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
        
        row, col, text = self.view.get_table_item_info(item)
        sites = self.view.get_every_site()-1
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
        
        mu1 = self.view.get_every_mu1()
        mu2 = self.view.get_every_mu2()
        mu3 = self.view.get_every_mu3()
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
        
        mu1 = self.view.get_every_mu1()
        mu2 = self.view.get_every_mu2()
        mu3 = self.view.get_every_mu3()
        
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
        
        Uiso, \
        U1, U2, U3 = self.model.atomic_displacement_parameters(U11, U22, U33, 
                                                               U23, U13, U12,
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
            
            self.view.clear_unit_cell_table()
            self.view.clear_atom_site_table()
                        
            folder, filename = name.rsplit('/', 1)
            folder += '/'
            
            parameters = self.model.load_lattice_parameters(folder, filename)
            a, b, c, alpha, beta, gamma = parameters
            
            lat = self.model.find_lattice(a, b, c, alpha, beta, gamma)
            
            self.view.set_lattice(lat)
            
            self.view.set_lattice_parameters(a, b, c, alpha, beta, gamma)
            
            group, hm = self.model.load_space_group(folder, filename)
            
            self.view.set_space_group(group, hm)
                
            u, v, w, occupancy, \
            displacement, moment, \
            site, op, mag_op, \
            atm, n_atm = self.model.load_unit_cell(folder, filename)
            
            A, B, R, C, D = self.model.crystal_matrices(a, b, c, 
                                                        alpha, beta, gamma)
            
            U11, U22, U33, \
            U23, U13, U12 = self.model.anisotropic_parameters(displacement, D)
            
            mu1, mu2, mu3 = np.round(moment.T, 4)
            
            Uiso, \
            U1, \
            U2, \
            U3 = self.model.atomic_displacement_parameters(U11, U22, U33, 
                                                           U23, U13, U12, D)

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
            
            empty = np.full(n_atm, '-', dtype='<U3')
                        
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
            self.view.set_unit_cell_atom(empty)
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
        
    def draw_plot_exp(self):
        
        canvas = self.view.get_plot_exp_canvas()
        data = self.exp_arr_m 
        
        constants = self.view.get_lattice_parameters()        
        A, B, R, C, D = self.model.crystal_matrices(*constants)
                
        matrix_h, scale_h = self.model.matrix_transform(B, 'h')
        matrix_k, scale_k = self.model.matrix_transform(B, 'k')
        matrix_l, scale_l = self.model.matrix_transform(B, 'l')
        
        h = self.view.get_slice_h()
        k = self.view.get_slice_k()
        l = self.view.get_slice_l()
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        ih = self.model.slice_index(min_h, max_h, nh, h)
        ik = self.model.slice_index(min_k, max_k, nk, k)
        il = self.model.slice_index(min_l, max_l, nl, l)
        
        h = self.model.slice_value(min_h, max_h, nh, ih)
        k = self.model.slice_value(min_k, max_k, nk, ik)
        l = self.model.slice_value(min_l, max_l, nl, il)
       
        self.view.set_slice_h(h)
        self.view.set_slice_k(k)
        self.view.set_slice_l(l)
        
        norm = self.view.get_norm_exp()
        
        vmin = self.view.get_min_exp()
        vmax = self.view.get_max_exp()
        
        self.view.validate_min_exp()
        self.view.validate_max_exp()
        
        plots.plot_exp(canvas, data, h, k, l, ih, ik, il, 
                       min_h, min_k, min_l, max_h, max_k, max_l, nh, nk, nl, 
                       matrix_h, matrix_k, matrix_l, scale_h, scale_k, scale_l,
                       norm, vmin, vmax)
        
    def redraw_plot_exp(self):
        
        if (self.view.get_plot_exp() == 'Intensity'):
            self.exp_arr_m = self.signal_m 
        else:
            self.exp_arr_m = self.error_sq_m 
            
        self.view.set_min_exp(self.exp_arr_m.min())
        self.view.set_max_exp(self.exp_arr_m.max())
        
        self.draw_plot_exp()
        
    def update_experiment_table(self, item):
        
        row, col, text = self.view.get_table_item_info(item)
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        h_range = [min_h, max_h]
        k_range = [min_k, max_k]
        l_range = [min_l, max_l]
        
        binning = [nh, nk, nl]
        
        self.view.block_experiment_table_signals()
        
        if (col == 1):
            size = int(text)
            if   (row == 0): minimum, maximum, step = min_h, max_h, dh
            elif (row == 1): minimum, maximum, step = min_k, max_k, dk
            elif (row == 2): minimum, maximum, step = min_l, max_l, dl
            n = self.model.size_value(minimum, maximum, step)
            if   (row == 0): binning[0] = n
            elif (row == 1): binning[1] = n         
            elif (row == 2): binning[2] = n         
            if (size < n and size > 1): 
                self.rebin()
            else:
                self.view.set_experiment_binning_h(binning[0], min_h, max_h)
                self.view.set_experiment_binning_k(binning[1], min_k, max_k)
                self.view.set_experiment_binning_l(binning[2], min_l, max_l)
        elif (col == 2):
            minimum = float(text)
            if   (row == 0): size, step, maximum = nh, dh, max_h
            elif (row == 1): size, step, maximum = nk, dk, max_k
            elif (row == 2): size, step, maximum = nl, dl, max_l
            low = self.model.minimum_value(size, step, maximum)
            if   (row == 0): h_range = [low, max_h]
            elif (row == 1): k_range = [low, max_k]
            elif (row == 2): l_range = [low, max_l]
            if (minimum > low and minimum < maximum): 
                self.crop(h_range, k_range, l_range)
            else:
                self.view.set_experiment_binning_h(nh, h_range[0], max_h)
                self.view.set_experiment_binning_k(nk, k_range[0], max_k)
                self.view.set_experiment_binning_l(nl, l_range[0], max_l)
        elif (col == 3):
            maximum = float(text)
            if   (row == 0): size, step, minimum = nh, dh, min_h
            elif (row == 1): size, step, minimum = nk, dk, min_k
            elif (row == 2): size, step, minimum = nl, dl, min_l
            high = self.model.maximum_value(size, step, minimum)
            if   (row == 0): h_range = [min_h, high]
            elif (row == 1): k_range = [min_k, high]
            elif (row == 2): l_range = [min_l, high]
            if (maximum < high and maximum > minimum): 
                self.crop(h_range, k_range, l_range)    
            else:
                self.view.set_experiment_binning_h(nh, min_h, h_range[1])
                self.view.set_experiment_binning_k(nk, min_k, k_range[1])
                self.view.set_experiment_binning_l(nl, min_l, l_range[1])
        
        self.view.format_experiment_table()
        self.view.unblock_experiment_table_signals()
        
        self.redraw_plot_exp()
        
    def update_crop_min_h(self):

        self.view.set_experiment_table_item(0, 2, self.view.get_min_h())
        
    def update_crop_min_k(self):
        
        self.view.set_experiment_table_item(1, 2, self.view.get_min_k())
        
    def update_crop_min_l(self):
        
        self.view.set_experiment_table_item(2, 2, self.view.get_min_l())
        
    def update_crop_max_h(self):

        self.view.set_experiment_table_item(0, 3, self.view.get_max_h())
        
    def update_crop_max_k(self):
        
        self.view.set_experiment_table_item(1, 3, self.view.get_max_k())
        
    def update_crop_max_l(self):
        
        self.view.set_experiment_table_item(2, 3, self.view.get_max_l())
        
    def update_binning_h(self):
        
        nh = self.view.get_rebin_combo_h()
        if (nh is not None):
            self.view.set_experiment_table_item(0, 1, nh)
  
    def update_binning_k(self):
        
        nk = self.view.get_rebin_combo_k()
        if (nk is not None):
            self.view.set_experiment_table_item(1, 1, nk)
        
    def update_binning_l(self):
        
        nl = self.view.get_rebin_combo_l()
        if (nl is not None):
            self.view.set_experiment_table_item(2, 1, nl)
            
    def rebin_thread(self, callback):
            
        signal = self.signal_m 
        error_sq = self.error_sq_m 
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        binsize = [nh, nk, nl]
        
        self.signal_m = self.model.rebin(signal, binsize)
        self.error_sq_m = self.model.rebin(error_sq, binsize)
        
        signal = self.signal_m 
        error_sq = self.error_sq_m      
        
        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)
        
        return nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l
            
    def rebin_process_output(self, data):
        
        nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l = data
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
    def rebin_thread_complete(self):
                        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
            
    def rebin(self):
        
        rebin_data = self.view.worker(self.rebin_thread)
        self.view.result(rebin_data, self.rebin_process_output)
        self.view.finished(rebin_data, self.rebin_thread_complete)
        self.threadpool.start(rebin_data) 
        
    def crop_thread(self, h_range, k_range, l_range, callback):
                           
        signal = self.signal_m 
        error_sq = self.error_sq_m 
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        ih_min = self.model.slice_index(h_range[0], h_range[1], nh, min_h)
        ik_min = self.model.slice_index(k_range[0], k_range[1], nk, min_k)
        il_min = self.model.slice_index(l_range[0], l_range[1], nl, min_l)
        
        ih_max = self.model.slice_index(h_range[0], h_range[1], nh, max_h)
        ik_max = self.model.slice_index(k_range[0], k_range[1], nk, max_k)
        il_max = self.model.slice_index(l_range[0], l_range[1], nl, max_l)
        
        h_slice = [ih_min, ih_max+1]
        k_slice = [ik_min, ik_max+1]
        l_slice = [il_min, il_max+1]
                
        self.signal_m = self.model.crop(signal, h_slice, k_slice, l_slice)
        self.error_sq_m = self.model.crop(error_sq, h_slice, k_slice, l_slice)
        
        signal = self.signal_m 
        error_sq = self.error_sq_m      
        
        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)
        
        nh = h_slice[1]-h_slice[0]
        nk = k_slice[1]-k_slice[0]
        nl = l_slice[1]-l_slice[0]
        
        print(nh,nk,nl)
        
        return nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l
                             
    def crop_process_output(self, data):
        
        nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l = data
        
        print(nh,nk,nl)
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
    def crop_thread_complete(self):
                        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
            
    def crop(self, h_range, k_range, l_range):
        
        crop_data = self.view.worker(self.crop_thread, h_range, k_range, l_range)
        self.view.result(crop_data, self.crop_process_output)
        self.view.finished(crop_data, self.crop_thread_complete)
        self.threadpool.start(crop_data) 
        
    def populate_binning(self):
        
        self.populate_binning_h()
        self.populate_binning_k()
        self.populate_binning_l()
        
    def populate_binning_h(self):

        self.view.block_changed_combo_h(True)
                
        self.view.clear_rebin_combo_h()
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
 
        cntr_h = self.view.centered_h_checked()              

        hsteps, hsizes = self.model.rebin_parameters(nh, min_h, max_h, cntr_h)
                
        self.view.set_rebin_combo_h(hsteps, hsizes)
        
        self.view.block_changed_combo_h(False)
               
    def populate_binning_k(self):

        self.view.block_changed_combo_k(True)
        
        self.view.clear_rebin_combo_k()
        
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()                
 
        cntr_k = self.view.centered_k_checked()

        ksteps, ksizes = self.model.rebin_parameters(nk, min_k, max_k, cntr_k)
        
        self.view.set_rebin_combo_k(ksteps, ksizes)
        
        self.view.block_changed_combo_k(False)
                
    def populate_binning_l(self):
 
        self.view.block_changed_combo_l(True)
               
        self.view.clear_rebin_combo_l()
                     
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()

        cntr_l = self.view.centered_l_checked()  
        
        lsteps, lsizes = self.model.rebin_parameters(nl, min_l, max_l, cntr_l)

        self.view.set_rebin_combo_l(lsteps, lsizes)    
        
        self.view.block_changed_combo_l(False)
                        
    def populate_cropping(self):
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
    
        self.view.set_min_h(min_h)
        self.view.set_min_k(min_k)
        self.view.set_min_l(min_l)
                        
        self.view.set_max_h(max_h)
        self.view.set_max_k(max_k)
        self.view.set_max_l(max_l)
        
        self.view.validate_crop_h(min_h, max_h)
        self.view.validate_crop_k(min_k, max_k)
        self.view.validate_crop_l(min_l, max_l)
        
    def populate_slicing(self):
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        h = self.view.get_slice_h()
        k = self.view.get_slice_k()
        l = self.view.get_slice_l()
        
        if (h is None): 
            h = self.model.slice_value(min_h, max_h, nh, (nh-1) // 2)
        if (k is None): 
            k = self.model.slice_value(min_k, max_k, nk, (nk-1) // 2)
        if (l is None): 
            l = self.model.slice_value(min_l, max_l, nl, (nl-1) // 2)
            
        if (h < min_h or h > max_h): 
            h = self.model.slice_value(min_h, max_h, nh, (nh-1) // 2)
        if (k < min_k or k > max_k): 
            k = self.model.slice_value(min_k, max_k, nk, (nk-1) // 2)
        if (l < min_l or l > max_l): 
            l = self.model.slice_value(min_l, max_l, nl, (nl-1) // 2)
        
        self.view.set_slice_h(h)
        self.view.set_slice_k(k)
        self.view.set_slice_l(l)
        
        self.view.validate_slice_h(min_h, max_h)
        self.view.validate_slice_k(min_k, max_k)
        self.view.validate_slice_l(min_l, max_l)
        
    def connect_experiment_table_signals(self):
        
        self.view.index_changed_combo_h(self.update_binning_h)
        self.view.index_changed_combo_k(self.update_binning_k)
        self.view.index_changed_combo_l(self.update_binning_l)
        
        self.view.clicked_centered_h(self.populate_binning_h)
        self.view.clicked_centered_k(self.populate_binning_k)
        self.view.clicked_centered_l(self.populate_binning_l)
        
        self.view.finished_editing_min_h(self.update_crop_min_h)
        self.view.finished_editing_min_k(self.update_crop_min_k)
        self.view.finished_editing_min_l(self.update_crop_min_l)
        
        self.view.finished_editing_max_h(self.update_crop_max_h)
        self.view.finished_editing_max_k(self.update_crop_max_k)
        self.view.finished_editing_max_l(self.update_crop_max_l)
                        
        self.view.item_changed_experiment_table(
            self.update_experiment_table
        )
        
        self.redraw_plot_exp()
        
        self.view.index_changed_plot_exp(self.redraw_plot_exp)
        self.view.index_changed_norm_exp(self.redraw_plot_exp)
        
        self.view.finished_editing_slice_h(self.redraw_plot_exp)  
        self.view.finished_editing_slice_k(self.redraw_plot_exp)                
        self.view.finished_editing_slice_l(self.redraw_plot_exp)
        
        self.view.finished_editing_min_exp(self.draw_plot_exp)
        self.view.finished_editing_max_exp(self.draw_plot_exp)     
        
    def reset_data(self):
        
        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()
                
        nh, nk, nl = self.nh_raw_m, self.nk_raw_m, self.nl_raw_m
    
        min_h, max_h = self.h_range_raw_m
        min_k, max_k = self.k_range_raw_m
        min_l, max_l = self.l_range_raw_m

        self.view.create_experiment_table()
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
        self.view.format_experiment_table()
        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
        
        self.connect_experiment_table_signals()
        self.view.format_experiment_table_size()

    def cropbin(self, h_range, k_range, l_range, binsize):
        
        signal = self.signal_m 
        error_sq = self.error_sq_m 

        nh_raw, nk_raw, nl_raw = self.nh_raw_m, self.nk_raw_m, self.nl_raw_m

        h_range_raw = self.h_range_raw_m.copy()
        k_range_raw = self.k_range_raw_m.copy()
        l_range_raw = self.l_range_raw_m.copy()
        
        min_h, max_h = h_range
        min_k, max_k = k_range
        min_l, max_l = l_range
        
        min_h_raw, max_h_raw = h_range_raw
        min_k_raw, max_k_raw = k_range_raw
        min_l_raw, max_l_raw = l_range_raw     
                
        ih_min, \
        ih_max = self.model.crop_parameters(min_h, max_h,
                                            min_h_raw, max_h_raw, nh_raw)
        ik_min, \
        ik_max = self.model.crop_parameters(min_k, max_k,
                                            min_k_raw, max_k_raw, nk_raw)
        il_min, \
        il_max = self.model.crop_parameters(min_l, max_l,
                                            min_l_raw, max_l_raw, nl_raw)
           
        h_slice = [ih_min, ih_max+1]
        k_slice = [ik_min, ik_max+1]
        l_slice = [il_min, il_max+1]
                
        self.signal_m  = self.model.crop(signal, h_slice, k_slice, l_slice)
        self.error_sq_m  = self.model.crop(error_sq, h_slice, k_slice, l_slice)
        
        signal = self.signal_m 
        error_sq = self.error_sq_m 
        
        self.signal_m = self.model.rebin(signal, binsize)
        self.error_sq_m = self.model.rebin(error_sq, binsize)
        
    def reset_data_h(self):
        
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()
                
        nh = self.nh_raw_m
    
        min_h, max_h = self.h_range_raw_m
        
        self.cropbin([min_h, max_h], [min_k, max_k], 
                     [min_l, max_l], [nh, nk, nl])
        
        self.view.create_experiment_table()
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
        self.view.format_experiment_table()
        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
        
        self.connect_experiment_table_signals()
        self.view.format_experiment_table_size()
        
    def reset_data_k(self):
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()
                
        nk =  self.nk_raw_m
    
        min_k, max_k = self.k_range_raw_m
        
        self.cropbin([min_h, max_h], [min_k, max_k], 
                     [min_l, max_l], [nh, nk, nl])
    
        self.view.create_experiment_table()
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
        self.view.format_experiment_table()
        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
        
        self.connect_experiment_table_signals()
        self.view.format_experiment_table_size()
        
    def reset_data_l(self):
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        
        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()
                
        nl = self.nl_raw_m
    
        min_l, max_l = self.l_range_raw_m
        
        self.cropbin([min_h, max_h], [min_k, max_k], 
                     [min_l, max_l], [nh, nk, nl])

        self.view.create_experiment_table()
        
        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        
        self.view.format_experiment_table()
        
        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()
        
        self.connect_experiment_table_signals()
        self.view.format_experiment_table_size()
        
    def punch(self):
        
        signal = self.signal_m 
        error_sq = self.error_sq_m 
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        h_range = [min_h, max_h]
        k_range = [min_k, max_k]
        l_range = [min_l, max_l]
                
        radius_h = self.view.get_radius_h()
        radius_k = self.view.get_radius_k()
        radius_l = self.view.get_radius_l()
        
        centering = self.view.get_centering()
        outlier = self.view.get_outlier()
        punch = self.view.get_punch()
        
        self.model.punch(signal, radius_h, radius_k, radius_l, 
                         dh, dk, dl, h_range, k_range, l_range,
                         centering, outlier, punch)
        
        self.model.punch(error_sq, radius_h, radius_k, radius_l, 
                         dh, dk, dl, h_range, k_range, l_range,
                         centering, outlier, punch)
        
        self.signal_m = signal
        self.error_sq_m = error_sq
        
        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)
        
        self.redraw_plot_exp()
        
    def reset_punch(self):
        
        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        self.cropbin([min_h, max_h], [min_k, max_k], 
                     [min_l, max_l], [nh, nk, nl])

        self.redraw_plot_exp()
        
    def populate_recalculation_table(self):
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        self.view.create_recalculation_table(dh, nh, min_h, max_h,
                                             dk, nk, min_k, max_k,
                                             dl, nl, min_l, max_l)
        
        self.view.format_recalculation_table()
        
        self.view.item_changed_recalculation_table(
            self.update_recalculation_table
        )

    def update_recalculation_table(self):
                
        dh, nh, min_h, max_h = self.view.get_recalculation_binning_h()
        dk, nk, min_k, max_k = self.view.get_recalculation_binning_k()
        dl, nl, min_l, max_l = self.view.get_recalculation_binning_l()
                
        self.view.block_recalculation_table_signals()
        
        self.view.set_recalculation_binning_h(nh, min_h, max_h)
        self.view.set_recalculation_binning_k(nk, min_k, max_k)
        self.view.set_recalculation_binning_l(nl, min_l, max_l)
        
        self.view.format_recalculation_table()
        self.view.unblock_recalculation_table_signals()
        
    def load_data_thread(self, name, callback):
                           
        signal, error_sq, \
        h_range, k_range, l_range, nh, nk, nl = self.model.data(name)
        
        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)
                
        self.signal_raw_m = self.signal_m.copy()
        self.error_sq_raw_m = self.error_sq_m.copy()
        
        self.h_range_raw_m =  h_range.copy()
        self.k_range_raw_m =  k_range.copy()
        self.l_range_raw_m =  l_range.copy()
        
        self.nh_raw_m, self.nk_raw_m, self.nl_raw_m = nh, nk, nl
                             
    def load_data_progress_update(self):
        
        pass

    def load_data_thread_complete(self):
        
        self.reset_data()
        
        self.view.button_clicked_reset(self.reset_data)
        self.view.button_clicked_reset_h(self.reset_data_h)
        self.view.button_clicked_reset_k(self.reset_data_k)
        self.view.button_clicked_reset_l(self.reset_data_l)
        
        self.view.button_clicked_punch(self.punch)
        self.view.button_clicked_reset_punch(self.reset_punch)
        
        self.populate_recalculation_table()
                
    def load_NXS(self):

        if (self.view.get_atom_site_table_col_count() > 0):

            name = self.view.open_dialog_nxs()
            
            if name:
                         
                load_data = self.view.worker(self.load_data_thread, name)
                self.view.progress(load_data, self.load_data_progress_update)
                self.view.finished(load_data, self.load_data_thread_complete)
                self.threadpool.start(load_data) 
                                
    def check_batch(self):
                
        visibility = True if self.view.batch_checked() else False
        self.view.enable_runs(visibility)
        if (not visibility): self.view.set_runs(1)
    
    def check_batch_1d(self):
                
        visibility = True if self.view.batch_checked_1d() else False
        self.view.enable_runs_1d(visibility)
        if (not visibility): self.view.set_runs_1d(1)
        
    def check_batch_3d(self):
                
        visibility = True if self.view.batch_checked_3d() else False
        self.view.enable_runs_3d(visibility)
        if (not visibility): self.view.set_runs_3d(1)
        
    def check_batch_calc(self):
                
        visibility = True if self.view.batch_checked_calc() else False
        self.view.enable_runs_calc(visibility)
        if (not visibility): self.view.set_runs_calc(1)
       
    def disorder_check_mag(self):
        
        if self.view.get_disorder_mag():
            self.view.set_disorder_mag(True)
            self.view.set_disorder_occ(False)
            self.view.set_disorder_dis(False)
        else:
            self.view.set_disorder_mag(False)
            self.view.set_disorder_occ(True)
            self.view.set_disorder_dis(False)
            
    def disorder_check_occ(self):
        
        if self.view.get_disorder_occ():
            self.view.set_disorder_mag(False)
            self.view.set_disorder_occ(True)
            self.view.set_disorder_dis(False)
        else:
            self.view.set_disorder_mag(False)
            self.view.set_disorder_occ(False)
            self.view.set_disorder_dis(True)
            
    def disorder_check_dis(self):
        
        if self.view.get_disorder_dis():
            self.view.set_disorder_mag(False)
            self.view.set_disorder_occ(False)
            self.view.set_disorder_dis(True)
        else:
            self.view.set_disorder_mag(False)
            self.view.set_disorder_occ(True)
            self.view.set_disorder_dis(False)
            
    def change_type_1d(self):
        
        self.view.set_correlations_1d_type()
            
    def change_type_3d(self):
        
        self.view.set_correlations_3d_type()
        
    def preprocess_supercell(self):
        
        mask = self.model.get_mask(self.signal_m, self.error_sq_m)
        
        self.mask = mask
        
        signal, error_sq = self.signal_m, self.error_sq_m
        
        data = self.model.get_refinement_data(signal, error_sq, mask)
        
        I_expt, inv_sigma_sq = data
        self.I_expt, self.inv_sigma_sq = I_expt, inv_sigma_sq
        
        nu, nv, nw = self.view.get_nu(), self.view.get_nv(), self.view.get_nw()
        
        self.nu, self.nv, self.nw = nu, nv, nw
        
        n_atm = self.view.get_n_atm()
        n_uvw = self.model.unitcell_size(nu, nv, nw)
        n = self.view.get_n()
        
        self.n_atm, self.n_uvw, self.n = n_atm, n_uvw, n
        
        constants = self.view.get_lattice_parameters()  
        
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = constants
        
        A, B, R, C, D = self.model.crystal_matrices(*constants)
        
        self.A, self.B, self.R, self.C, self.D = A, B, R, C, D
        
        site = self.view.get_site()
        
        self.site = site 
        
        element = self.view.get_atom()    
        nucleus = self.view.get_isotope()    
        charge = self.view.get_ion()
        
        self.atm = element
        
        nuc = self.model.get_isotope(element, nucleus)
        ion = self.model.get_ion(element, charge)
                
        self.nuc, self.ion = nuc, ion
                
        occupancy = self.view.get_occupancy()
        Uiso = self.view.get_Uiso()    
        U11 = self.view.get_U11()    
        U22 = self.view.get_U22()    
        U33 = self.view.get_U33()    
        U23 = self.view.get_U23()    
        U13 = self.view.get_U13()    
        U12 = self.view.get_U12()  
        
        self.occupancy, self.Uiso = occupancy, Uiso
        
        self.U11, self.U22, self.U33 = U11, U22, U33
        self.U23, self.U13, self.U12 = U23, U13, U12

        mu = self.view.get_mu()    
        mu1 = self.view.get_mu1()    
        mu2 = self.view.get_mu2()    
        mu3 = self.view.get_mu3()   
        g = self.view.get_g()    
        
        self.mu, self.mu1, self.mu2, self.mu3, self.g = mu, mu1, mu2, mu3, g
        
        u = self.view.get_u()    
        v = self.view.get_v()    
        w = self.view.get_w()
        
        self.u, self.v, self.w = u, v, w
        
        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()
        
        self.nh, self.min_h, self.max_h = nh, min_h, max_h
        self.nk, self.min_k, self.max_k = nk, min_k, max_k
        self.nl, self.min_l, self.max_l = nl, min_l, max_l
        
        h_range = [min_h, max_h]
        k_range = [min_k, max_k]
        l_range = [min_l, max_l]
        
        h, k, l, \
        H, K, L, \
        indices, \
        inverses, \
        i_mask, \
        i_unmask = self.model.reciprocal_mapping(h_range, k_range, l_range,
                                                 nu, nv, nw, mask)
        
        self.h, self.k, self.l = h, k, l
        self.H, self.K, self.L = H, K, L
        
        self.indices, self.inverses = indices, inverses
        self.i_mask, self.i_unmask = i_mask, i_unmask
        
        Q_arrays = self.model.reciprocal_space_coordinate_transform(h, k, l, 
                                                                    B, R)
        
        Qx, Qy, Qz, Qx_norm, Qy_norm, Qz_norm, Q = Q_arrays
        
        self.Qx, self.Qy, self.Qz, self.Q = Qx, Qy, Qz, Q
        self.Qx_norm, self.Qy_norm, self.Qz_norm = Qx_norm, Qy_norm, Qz_norm
        
        r_arrays = self.model.real_space_coordinate_transform(u, v, w, element, 
                                                              A, nu, nv, nw)
       
        ux, uy, uz, rx, ry, rz, atms = r_arrays
        
        self.ux, self.uy, self.uz = ux, uy, uz
        self.rx, self.ry, self.rz = rx, ry, rz
        self.atms = atms
                                        
        exp_factors = self.model.exponential_factors(Qx, Qy, Qz, 
                                                     ux, uy, uz, 
                                                     nu, nv, nw)
         
        phase_factor, space_factor = exp_factors
        self.phase_factor, self.space_factor = exp_factors
                                        
        if (self.view.get_type() == 'Neutron'):
             factors = self.model.neutron_factors(Q, element, ion, occupancy,
                                                  g, phase_factor)
             self.factors, self.magnetic_factors = factors
        else:
             factors = self.model.xray_factors(Q, ion, occupancy, phase_factor)
             self.factors = factors
             
    def initialize_intensity(self):
        
        mask, Q = self.mask, self.Q
                     
        I_arrays = self.model.initialize_intensity(mask, Q)
        
        self.I_obs, self.I_ref, self.I_calc, self.I_raw, self.I_flat = I_arrays
        
        filt_arrays = self.model.initialize_filter(mask)
            
        self.a_filt, self.b_filt, self.c_filt, \
        self.d_filt, self.e_filt, self.f_filt, \
        self.g_filt, self.h_filt, self.i_filt = filt_arrays
             
    def refinement_statistics(self):
        
        self.acc_moves = []
        self.acc_temps = []
        self.rej_moves = []
        self.rej_temps = []
        
        self.chi_sq = [np.inf]
        self.energy = []
        self.temperature = [self.view.get_prefactor()]
        self.scale = []
        
    def initialize_disorder(self):
        
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        moment = self.moment
        
        Sx, Sy, Sz = self.model.random_moments(nu, nv, nw, n_atm, moment)
        
        self.Sx, self.Sy, self.Sz = Sx, Sy, Sz 
        
        occupancy = self.occupancy
        
        A_r = self.model.random_occupancies(nu, nv, nw, n_atm, occupancy)
        
        self.A_r = A_r 
        
        Uiso = self.Uiso
        
        Ux, Uy, Uz = self.model.random_displacements(nu, nv, nw, n_atm, Uiso)
        
        self.Ux, self.Uy, self.Uz = Ux, Uy, Uz
        
    def initialize_magnetic(self):
        
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        
        H, K, L = self.H, self.K, self.L
        Qx_norm, Qy_norm, Qz_norm = self.Qx_norm, self.Qy_norm, self.Qz_norm
        
        indices = self.indices
        magnetic_factors = self.magnetic_factors
        
        arrays = self.model.initialize_magnetic(Sx, Sy, Sz, H, K, L, 
                                                Qx_norm, Qy_norm, Qz_norm, 
                                                indices, magnetic_factors, 
                                                nu, nv, nw, n_atm)
                
        self.Sx_k, self.Sy_k, self.Sz_k, \
        self.Sx_k_orig, self.Sy_k_orig, self.Sz_k_orig, \
        self.Sx_k_cand, self.Sy_k_cand, self.Sz_k_cand, \
        self.Fx, self.Fy, self.Fz, \
        self.Fx_orig, self.Fy_orig, self.Fz_orig, \
        self.Fx_cand, self.Fy_cand, self.Fz_cand, \
        self.prod_x, self.prod_y, self.prod_z, \
        self.prod_x_orig, self.prod_y_orig, self.prod_z_orig, \
        self.prod_x_cand, self.prod_y_cand, self.prod_z_cand, \
        self.i_dft = arrays
        
    def initialize_occupational(self):
        
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        A_r = self.A_r 
        
        H, K, L = self.H, self.K, self.L
        
        indices = self.indices
        factors = self.factors
        
        arrays = self.model.initialize_occupational(A_r, H, K, L, indices, 
                                                    factors, nu, nv, nw, n_atm)
               
        self.A_k, self.A_k_orig, self.A_k_cand, \
        self.F, self.F_orig, self.F_cand, \
        self.prod, self.prod_orig, self.prod_cand, self.i_dft = arrays
        
    def initialize_displacive(self):
            
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        Ux, Uy, Uz = self.Ux, self.Uy, self.Uz
        
        H, K, L = self.H, self.K, self.L
        h, k, l = self.h, self.k, self.l
        
        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz
        
        indices = self.indices
        factors = self.factors
        
        p = self.view.get_order()
        centering = self.view.get_centering_ref()
        
        arrays = self.model.initialize_displacive(Ux, Uy, Uz, h, k, l, H, K, L, 
                                                  Qx, Qy, Qz, indices, factors, 
                                                  nu, nv, nw, n_atm, p, 
                                                  centering)
       
        self.U_r, self.U_r_orig, self.U_r_cand, self.Q_k, \
        self.U_k, self.U_k_orig, self.U_k_cand, \
        self.V_k, self.V_k_orig, self.V_k_cand, \
        self.V_k_nuc, self.V_k_nuc_orig, self.V_k_nuc_cand, \
        self.F, self.F_orig, self.F_cand, \
        self.F_nuc, self.F_nuc_orig, self.F_nuc_cand, \
        self.prod, self.prod_orig, self.prod_cand, \
        self.prod_nuc, self.prod_nuc_orig, self.prod_nuc_cand, \
        self.i_dft, self.coeffs, self.H_nuc, self.K_nuc, self.L_nuc, \
        self.cond, self.even, self.bragg = arrays
        
    def calculate_intensity(self):

        if (self.tableWidget_calc.rowCount() or self.progress > 0):

            if (self.changed_params):
                            
                if (self.allocated == False):
                    self.preprocess_supercell()
                                                    
                batch = self.view.batch_checked_calc()
                
                runs = self.view.get_runs_calc()
                
                # ---
                
                dh, nh, min_h, max_h = self.view.get_recalculation_binning_h()
                dk, nk, min_k, max_k = self.view.get_recalculation_binning_k()
                dl, nl, min_l, max_l = self.view.get_recalculation_binning_l()
                
                h_range = [min_h, max_h]
                k_range = [min_k, max_k]
                l_range = [min_l, max_l]
                
                ux, uy, uz, atm = self.ux, self.uy, self.uz, self.atm
                
                nu, nv, nw = self.nu, self.nv, self.nw
                
                folder, filename = self.folder, self.filename
                
                fname = self.fname
                
                B, R, = self.B, self.R
                
                occupancy, g = self.occupancy, self.g
                
                # ---
                
                twins = np.zeros((1,3,3))
                variants = np.array([1.])
                
                twins[0,:,:] = np.eye(3)
                                        
                axes = self.view.get_axes()       
                
                if (axes == '(h00), (0k0), (00l)'):
                    T = np.eye(3)
                else:                
                    T = np.array([[1, -1,  0],
                                  [1,  1,  0],
                                  [0,  0,  1]])*1.                
                
                laue = self.view.get_laue()
                                
                self.recalc_params = [nh,nk,nl,
                                      min_h,min_k,min_l,
                                      max_h,max_k,max_l,
                                      laue,axes,batch,runs]
                
                intensity = np.zeros((nh,nk,nl))
                            
                indices, inverses, operators, \
                Nu, Nv, Nw, \
                symop = self.model.reduced_crystal_symmetry(
                            h_range, k_range, l_range, nh, nk, nl, 
                            nu, nv, nw, T, folder, filename, laue)
                    
                if (self.displacive):
                    
                    p = self.view.get_order_calc()
    
                    coeffs, even, cntr = self.model.displacive_parameters(p)
                        
                # ---
                    
                for run in range(runs):               
    
                    if not batch: run = None
                        
                    if (self.magnetic):
                        
                        I_calc = self.model.magnetic_intensity(
                                     fname, run, ux, uy, uz, atm,
                                     h_range, k_range, l_range, indices, symop,
                                     T, B, R, twins, variants, nh, nk, nl, 
                                     nu, nv, nw, Nu, Nv, Nw, g)
                                                
                    elif (self.occupational):
                        
                        I_calc = self.model.occupational_intensity(
                                     fname, run, occupancy, ux, uy, uz, atm,
                                     h_range, k_range, l_range, indices, symop,
                                     T, B, R, twins, variants, nh, nk, nl,
                                     nu, nv, nw, Nu, Nv, Nw)
        
                    else:
                                
                        I_calc = self.model.displacive_intensity(
                                     fname, run, coeffs, ux, uy, uz, atm,
                                     h_range, k_range, l_range, indices, symop,
                                     T, B, R, twins, variants, nh, nk, nl,
                                     nu, nv, nw, Nu, Nv, Nw, p, even, cntr)
    
                intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)
                            
                intensity /= runs*operators.shape[0]
                
    def recalculation_blur(self):
                
        sigma_h, sigma_k, sigma_l = self.view.get_recalculation_filter()

        sigma = [sigma_h, sigma_k, sigma_l]
        
        intensity = self.intensity
        
        I_recalc = self.model.blurring(intensity, sigma)
        
        self.I_recalc = I_recalc
        
    def filter_sigma(self):
                
        sigma_h = self.view.get_filter_h()
        sigma_k = self.view.get_filter_k()
        sigma_l = self.view.get_filter_l()

        sigma = [sigma_h, sigma_k, sigma_l]
        
        mask = self.mask

        v_inv, boxes = self.model.gaussian(mask, sigma)
        
        self.v_inv, self.boxes = v_inv, boxes 
                    
    def run_refinement_thread(self, callback):
                           
        batch = self.view.get_run()
        runs = self.view.get_runs()
        cycles = self.view.get_cycles()
                                                        
        for b in range(batch, runs):

            if self.magnetic: self.initialize_magnetic()
            if self.occupational: self.initialize_occupational()
            if self.displacive: self.initialize_displacive()
        
            n = cycles // 1
              
            iteration = self.iteration
            for i in range(iteration, n):
                
                self.refinement_cycle()
                
                p = int(round(100*(i+1)/n))
                if (p <= 0): p = 1
                elif (p > 100): p = 100
                callback.emit([p, b])
                                
                self.iteration = i+1
                if self.stop:
                    break  
                
            Sx, Sy, Sx = self.Sx, self.Sy, self.Sz
            self.model.save_magnetic(self.fname, b, Sx, Sy, Sx)
          
            A_r = self.A_r
            self.model.save_occupational(self.fname, b, A_r)

            Ux, Uy, Ux = self.Ux, self.Uy, self.Uz
            self.model.save_displacive(self.fname, b, Ux, Uy, Ux)
                             
    def run_refinement_progress_update(self, data):
        
        progress, batch = data
        
        self.view.set_progress(progress)
        self.view.set_run(batch)
        
        self.redraw_plot_ref()
        self.draw_plot_chi_sq()
            
    def run_refinement_thread_complete(self):
        
        if (self.view.get_progress() == 100):
            batch = self.view.get_run()+1
            self.view.set_run(batch)
            self.iteration = 0
        
        self.view.enable_refinement(True)
        
    def run_refinement(self):
        
        if self.view.get_recalculation_table_row_count():            
            self.stop = False
            
            if not self.allocated:
                self.magnetic = self.view.get_disorder_mag()
                self.occupational = self.view.get_disorder_occ()
                self.displacive = self.view.get_disorder_dis()   
                
                self.preprocess_supercell()
                self.initialize_disorder()
                self.initialize_intensity()
                self.refinement_statistics()
                self.filter_sigma()
                
                self.allocated = True
            
            self.view.enable_refinement(False)
                         
            refinement = self.view.worker(self.run_refinement_thread)
            self.view.progress(refinement, self.run_refinement_progress_update)
            self.view.finished(refinement, self.run_refinement_thread_complete)
            self.threadpool.start(refinement) 
            
    def stop_refinement(self):
        
        if self.view.get_recalculation_table_row_count():
            if (self.iteration > 0 or self.view.get_run() > 0):
                self.stop = True

    def reset_refinement(self):
        
        if self.view.get_recalculation_table_row_count():
            self.allocated = False
                        
            self.stop_refinement()
                        
            self.view.set_progress(0)
            self.view.set_run(0)
            self.view.set_runs(1)
            self.iteration = 0
        
            self.view.clear_plot_ref_canvas()
            self.view.clear_plot_chi_sq_canvas()
            
    def refinement_cycle(self):
        
        N = self.n_uvw*self.n_atm*1
            
        delta = 1
        
        T = np.array([1.,1.,1.,0.,0.,0.])
        
        space_factor = self.space_factor
        I_calc, I_expt = self.I_calc, self.I_expt
        inv_sigma_sq = self.inv_sigma_sq
        
        I_raw, I_flat, I_ref = self.I_raw, self.I_flat, self.I_ref
        
        v_inv, boxes = self.v_inv, self.boxes
        
        a_filt, b_filt, c_filt = self.a_filt, self.b_filt, self.c_filt
        d_filt, e_filt, f_filt = self.d_filt, self.e_filt, self.f_filt
        g_filt, h_filt, i_filt = self.g_filt, self.h_filt, self.i_filt
        
        i_dft, inverses = self.i_dft, self.inverses
        i_mask, i_unmask = self.i_mask, self.i_unmask
        
        acc_moves, acc_temps = self.acc_moves, self.acc_temps
        rej_moves, rej_temps = self.rej_moves, self.rej_temps
        
        chi_sq, energy = self.chi_sq, self.energy
        temperature = self.temperature
        scale = self.scale
        
        nh, nk, nl = self.nh, self.nk, self.nl
        nu, nv, nw, n_atm, n = self.nu, self.nv, self.nw, self.n_atm, self.n
        
        constant = self.view.get_constant()

        if self.magnetic:
            
            Sx, Sy, Sz, mu = self.Sx, self.Sy, self.Sz, self.mu
            
            Sx_k, Sy_k, Sz_k = self.Sx_k, self.Sy_k, self.Sz_k
            
            Sx_k_orig, Sx_k_cand = self.Sx_k_orig, self.Sx_k_cand
            Sy_k_orig, Sy_k_cand = self.Sy_k_orig, self.Sy_k_cand
            Sz_k_orig, Sz_k_cand = self.Sz_k_orig, self.Sz_k_cand
            
            Fx, Fy, Fz = self.Fx, self.Fy, self.Fz
            
            Fx_orig, Fx_cand = self.Fx_orig, self.Fx_cand
            Fy_orig, Fy_cand = self.Fy_orig, self.Fy_cand
            Fz_orig, Fz_cand = self.Fz_orig, self.Fz_cand
            
            prod_x, prod_y, prod_z = self.prod_x, self.prod_y, self.prod_z
        
            prod_x_orig, prod_x_cand = self.prod_x_orig, self.prod_x_cand
            prod_y_orig, prod_y_cand = self.prod_y_orig, self.prod_y_cand 
            prod_z_orig, prod_z_cand = self.prod_z_orig, self.prod_z_cand
        
            magnetic_factors = self.magnetic_factors
        
            Qx_norm = self.Qx_norm
            Qy_norm = self.Qy_norm
            Qz_norm = self.Qz_norm
            
            fixed = self.view.fixed_moment_check()
            
            self.model.magnetic_refinement(
                Sx, Sy, Sz, Qx_norm, Qy_norm, Qz_norm, 
                Sx_k, Sy_k, Sz_k, 
                Sx_k_orig, Sy_k_orig, Sz_k_orig,
                Sx_k_cand, Sy_k_cand, Sz_k_cand,
                Fx, Fy, Fz,
                Fx_orig, Fy_orig, Fz_orig,
                Fx_cand, Fy_cand, Fz_cand,
                prod_x, prod_y, prod_z,
                prod_x_orig, prod_y_orig, prod_z_orig,  
                prod_x_cand, prod_y_cand, prod_z_cand,   
                space_factor, magnetic_factors, mu,
                I_calc, I_expt, inv_sigma_sq, I_raw, I_flat, I_ref, v_inv, 
                a_filt, b_filt, c_filt, 
                d_filt, e_filt, f_filt, 
                g_filt, h_filt, i_filt,
                boxes, i_dft, inverses, i_mask, i_unmask,
                acc_moves, acc_temps, rej_moves, rej_temps,
                chi_sq, energy, temperature, scale, constant,
                delta, fixed, T, nh, nk, nl, nu, nv, nw, n_atm, n, N)
            
        elif self.occupational:
            
            A_r, occupancy = self.A_r, self.occupancy
            
            A_k, A_k_orig, A_k_cand = self.A_k, self.A_k_orig, self.A_k_cand
            
            F, F_orig, F_cand = self.F, self.F_orig, self.F_cand
            
            prod = self.prod 
            prod_orig, prod_cand = self.prod_orig, self.prod_cand
            
            factors  = self.factors
            
            fixed = self.view.fixed_composition_check()
            
            self.model.occupational_refinement(
                A_r, A_k, A_k_orig, A_k_cand,
                F, F_orig, F_cand,
                prod, prod_orig, prod_cand,     
                space_factor, factors, occupancy,
                I_calc, I_expt, inv_sigma_sq, I_raw, I_flat, I_ref, v_inv, 
                a_filt, b_filt, c_filt,
                d_filt, e_filt, f_filt,
                g_filt, h_filt, i_filt,
                boxes, i_dft, inverses, i_mask, i_unmask,
                acc_moves, acc_temps, rej_moves, rej_temps,
                chi_sq, energy, temperature, scale, constant, fixed,
                nh, nk, nl, nu, nv, nw, n, n_atm, N)
    
        elif self.displacive:
            
            Ux, Uy, Uz = self.Ux, self.Uy, self.Uz 
            Uiso = self.Uiso
            
            U_r, U_r_orig, U_r_cand = self.U_r, self.U_r_orig, self.U_r_cand
            U_k, U_k_orig, U_k_cand = self.U_k, self.U_k_orig, self.U_k_cand
            
            V_k, V_k_nuc = self.V_k, self.V_k_nuc
            V_k_orig, V_k_nuc_orig = self.V_k_orig, self.V_k_nuc_orig
            V_k_cand, V_k_nuc_cand = self.V_k_cand, self.V_k_nuc_cand
            
            F, F_nuc = self.F, self.F_nuc
            F_orig, F_nuc_orig = self.F_orig, self.F_nuc_orig 
            F_cand, F_nuc_cand = self.F_cand, self.F_nuc_cand
            
            prod, prod_nuc = self.prod, self.prod_nuc 
            prod_orig, prod_nuc_orig = self.prod_orig, self.prod_nuc_orig
            prod_cand, prod_nuc_cand = self.prod_cand, self.prod_nuc_cand 
            
            factors = self.factors
            
            coeffs, Q_k = self.coeffs, self.Q_k 
            
            bragg, even = self.bragg, self.even
            
            fixed = self.view.fixed_displacement_check()
            
            p = self.view.get_order()
            
            self.model.displacive_refinement(
                Ux, Uy, Uz,
                U_r, U_r_orig, U_r_cand,
                U_k, U_k_orig, U_k_cand,
                V_k, V_k_nuc, V_k_orig,
                V_k_nuc_orig, V_k_cand, V_k_nuc_cand,
                F, F_nuc, F_orig, F_nuc_orig, F_cand, F_nuc_cand,
                prod, prod_nuc, prod_orig, prod_nuc_orig,    
                prod_cand, prod_nuc_cand, space_factor, factors,
                coeffs, Q_k, Uiso,
                I_calc, I_expt, inv_sigma_sq, I_raw, I_flat, I_ref, v_inv,
                a_filt, b_filt, c_filt,
                d_filt, e_filt, f_filt,
                g_filt, h_filt, i_filt,
                bragg, even, boxes, i_dft, inverses, i_mask, i_unmask, 
                acc_moves, acc_temps, rej_moves, rej_temps, 
                chi_sq, energy, temperature, scale, constant, 
                delta, fixed, T, p, nh, nk, nl,
                nu, nv, nw, n_atm, n, N)
            
        self.I_obs = I_flat.reshape(nh,nk,nl)
        self.I_obs[self.mask] = np.nan
        
        self.refinement_m = self.model.mask_array(self.I_obs)
            
    def draw_plot_ref(self):
        
        if self.allocated:
            
            canvas = self.view.get_plot_ref_canvas()
            data = self.ref_arr_m 
            
            B = self.B
            
            hkl = self.view.get_slice_hkl()
                    
            matrix_h, scale_h = self.model.matrix_transform(B, 'h')
            matrix_k, scale_k = self.model.matrix_transform(B, 'k')
            matrix_l, scale_l = self.model.matrix_transform(B, 'l')
            
            nh, min_h, max_h = self.nh, self.min_h, self.max_h
            nk, min_k, max_k = self.nk, self.min_k, self.max_k
            nl, min_l, max_l = self.nl, self.min_l, self.max_l
            
            slice_hkl = self.view.get_slice()
                   
            i_khl = (nh-1) // 2
            
            if (hkl == 'h ='):
                i_khl = (nh-1) // 2
                if (slice_hkl is not None):
                    i_khl = self.model.slice_index(min_h, max_h, nh, slice_hkl) 
                slice_hkl = self.model.slice_value(min_h, max_h, nh, i_khl)
                self.view.set_slice(slice_hkl)
            elif (hkl == 'k ='):
                i_khl = (nk-1) // 2
                if (slice_hkl is not None):
                    i_khl = self.model.slice_index(min_k, max_k, nk, slice_hkl)
                slice_hkl = self.model.slice_value(min_k, max_k, nk, i_khl)           
                self.view.set_slice(slice_hkl)
            elif (hkl == 'l ='):
                i_khl = (nl-1) // 2
                if (slice_hkl is not None):
                    i_khl = self.model.slice_index(min_l, max_l, nl, slice_hkl)
                slice_hkl = self.model.slice_value(min_l, max_l, nl, i_khl)
                self.view.set_slice(slice_hkl)
            
            norm = self.view.get_norm_ref()
            
            vmin = self.view.get_min_ref()
            vmax = self.view.get_max_ref()
            
            self.view.validate_min_ref()
            self.view.validate_max_ref()
            
            plots.plot_ref(canvas, data, hkl, slice_hkl, i_khl, 
                           min_h, min_k, min_l, max_h, max_k, max_l, 
                           nh, nk, nl, matrix_h, matrix_k, matrix_l,
                           scale_h, scale_k, scale_l, norm, vmin, vmax)
        
    def redraw_plot_ref(self):
        
        if self.allocated:
            
            plot_type = self.view.get_plot_ref()
            if (plot_type == 'Calculated'):
                self.ref_arr_m = self.refinement_m
            elif (plot_type == 'Experimental'):
                self.ref_arr_m = self.signal_m 
            else:
                self.ref_arr_m = self.error_sq_m 
                    
            self.view.set_min_ref(self.ref_arr_m.min())
            self.view.set_max_ref(self.ref_arr_m.max())
            
            self.draw_plot_ref()
            
    def draw_plot_chi_sq(self):
        
        if self.allocated:
                        
            canvas = self.view.get_plot_chi_sq_canvas()

            plot0 = self.view.get_plot_top_chi_sq()
            plot1 = self.view.get_plot_bottom_chi_sq()
                        
            acc_moves, rej_moves = self.acc_moves, self.rej_moves
            temperature, energy = self.temperature, self.energy
            chi_sq, scale = self.chi_sq, self.scale
            
            plots.chi_sq(canvas, plot0, plot1, acc_moves, rej_moves, 
                         temperature, energy, chi_sq, scale)
            
    # ---
    
    def recreate_table_1d(self):
                
        unique_pairs = np.unique(self.atm_pair1d)
        
        rows = self.get_pairs_1d_table_row_count()
        
        pairs = []
        for i in range(rows):
            left, right, active = self.view.get_pairs_1d_table_row(i)
            if (left == 'self-correlation'):
                pairs.append('0')
            else:
                pairs.append('{}_{}'.format(left,right))
                        
        if (rows == 0 or unique_pairs.tolist() != np.unique(pairs).tolist()):
            
            n_pairs = unique_pairs.size
            self.clear_pairs_1d_table()
            self.view.create_pairs_1d_table(n_pairs)
                       
            for i in range(n_pairs):
                uni = unique_pairs[i]
                if (uni == '0'):
                    uni = 'self-correlation'
                    self.view.set_pairs_1d_table_row([uni, ' ' , True], i)
                else:
                    left, right = uni.split('_')
                    self.view.set_pairs_1d_table_row([left, right, True], i)

            visible = False if self.view.get_average_1d() else True   
            self.view.enable_pairs_1d(visible)
            self.view.check_clicked_pairs_1d(self.plot_1d)
            self.view.format_pairs_1d_table()
        
    def calculate_1d_thread_complete(self):
                
        self.view.enable_calculate_1d(True)
        self.recreate_table_1d() 
        self.plot_1d()
        
    def calculate_correlations_1d(self):
        
        if (self.get_pairs_1d_table_row_count() > 0):
                        
            disorder = self.view.get_correlations_1d()
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):
                
                self.view.enable_calculate_1d(False)
                
                calculate_1d = self.view.worker(self.calculate_1d_thread)
                self.view.finished(calculate_1d, 
                                   self.calculate_1d_thread_complete)
                self.threadpool.start(calculate_1d) 

    def calculate_1d_thread(self, callback):
                
        disorder = self.view.get_correlations_1d()

        fract = self.view.get_fract_1d()
        tol = self.view.get_tol_1d()
        
        runs = self.view.get_runs_1d()
        
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        A = self.A
        
        rx, ry, rz, atms = self.rx, self.ry, self.rz, self.atms
                
        period = (A, nu, nv, nw, n_atm)
        
        corr1d_arrs, coll1d_arrs = [], []
        
        for run in range(runs):
            
            if (disorder == 'Moment'):
        
                Sx, Sy, Sz = self.model.load_magnetic(self.fname, run)
                
                corr1d, coll1d, \
                d, \
                atm_pair1d = self.model.vector_correlations_1d(Sx, Sy, Sz, 
                                                               rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr1d_arrs.append(corr1d)
                coll1d_arrs.append(coll1d)
                
            elif (disorder == 'Occupancy'):
                
                A_r = self.model.load_occupational(self.fname, run)
                
                corr1d, \
                dx, dy, dz, \
                atm_pair1d = self.model.scalar_correlations_1d(A_r, rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr1d_arrs.append(corr1d)
                
            elif (disorder == 'Displacement'):
        
                Ux, Uy, Uz = self.model.load_displacive(self.fname, run)
                
                corr1d, coll1d, \
                corr1d_, coll1d_, \
                d, \
                atm_pair1d = self.model.vector_correlations_1d(Ux, Uy, Uz, 
                                                               rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr1d_arrs.append(corr1d)
                coll1d_arrs.append(coll1d)               
                
        stats_corr1d = self.model.correlation_statistics(corr1d_arrs)
        corr1d, sigma_sq_corr1d = stats_corr1d
    
        if (disorder != 'Occupancy'):
            
            stats_coll1d = self.model.correlation_statistics(coll1d_arrs)
            coll1d, sigma_sq_coll1d = stats_coll1d
                
        if self.view.average_1d_checked():
            
            if (disorder != 'Occupancy'):
                
                data1d = [corr1d, coll1d, sigma_sq_corr1d, 
                          sigma_sq_coll1d, d, tol]
            
                ave1d = self.model.vector_average_1d(*data1d)
                
                corr1d, coll1d, sigma_sq_corr1d, \
                sigma_sq_coll1d, d = ave1d
            
            else:
                
                data1d = [corr1d, sigma_sq_corr1d, d, tol]
                
                ave1d = self.model.scalar_average_1d(*data1d)
                
                corr1d, sigma_sq_corr1d, d = ave1d
         
        self.d, self.atm_pair1d = d, atm_pair1d
        
        self.corr1d, self.sigma_sq_corr1d = corr1d, sigma_sq_corr1d
        
        if (disorder != 'Occupancy'):
                
            self.coll1d, self.sigma_sq_coll1d = coll1d, sigma_sq_coll1d

    def plot_1d(self):
        
        disorder = self.view.get_correlations_1d()
        correlation = self.view.get_plot_1d()
        norm = self.get_norm_1d()
                
        average = self.view.average_1d_checked()
        
        d, atm_pair1d = self.d, self.atm_pair1d 

        if (correlation == 'Correlation'):
            data = self.corr1d
            error = self.sigma_sq_corr1d
        else:
            data = self.coll1d
            error = self.sigma_sq_coll1d

        canvas = self.view.get_plot_1d_canvas()
        
        atoms, pairs = [], []
        for i in range(self.get_pairs_1d_table_row_count()):
            left, right, active = self.view.get_pairs_1d_table_row(i)
            if active:
                atoms.append(left)
                pairs.append(right)
        
        plots.correlations_1d(canvas, d, data, error, atm_pair1d, disorder, 
                              correlation, average, norm, atoms, pairs)

    def recreate_table_3d(self):
                
        unique_pairs = np.unique(self.atm_pair3d)
        
        rows = self.get_pairs_3d_table_row_count()
        
        pairs = []
        for i in range(rows):
            left, right, active = self.view.get_pairs_3d_table_row(i)
            if (left == 'self-correlation'):
                pairs.append('0')
            else:
                pairs.append('{}_{}'.format(left,right))
                        
        if (rows == 0 or unique_pairs.tolist() != np.unique(pairs).tolist()):
            
            n_pairs = unique_pairs.size
            self.clear_pairs_3d_table()
            self.view.create_pairs_3d_table(n_pairs)
                       
            for i in range(n_pairs):
                uni = unique_pairs[i]
                if (uni == '0'):
                    uni = 'self-correlation'
                    self.view.set_pairs_3d_table_row([uni, ' ' , True], i)
                else:
                    left, right = uni.split('_')
                    self.view.set_pairs_3d_table_row([left, right, True], i)

            visible = False if self.view.get_average_3d() else True   
            self.view.enable_pairs_3d(visible)
            self.view.check_clicked_pairs_3d(self.plot_3d)
            self.view.format_pairs_3d_table()
        
    def calculate_3d_thread_complete(self):
                
        self.view.enable_calculate_3d(True)
        self.recreate_table_3d() 
        self.plot_3d()
        
    def calculate_correlations_3d(self):
        
        if (self.get_pairs_3d_table_row_count() > 0):
                        
            disorder = self.view.get_correlations_3d()
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):
                
                self.view.enable_calculate_3d(False)
                
                calculate_3d = self.view.worker(self.calculate_3d_thread)
                self.view.finished(calculate_3d, 
                                   self.calculate_3d_thread_complete)
                self.threadpool.start(calculate_3d) 

    def calculate_3d_thread(self, callback):
                
        disorder = self.view.get_correlations_3d()

        fract = self.view.get_fract_3d()
        tol = self.view.get_tol_3d()
        
        runs = self.view.get_runs_3d()
        
        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm
        
        A = self.A
        
        rx, ry, rz, atms = self.rx, self.ry, self.rz, self.atms
        
        folder, filename = self.folder, self.filename
        
        period = (A, nu, nv, nw, n_atm)
        
        corr3d_arrs, coll3d_arrs = [], []
        
        for run in range(runs):
            
            if (disorder == 'Moment'):
        
                Sx, Sy, Sz = self.model.load_magnetic(self.fname, run)
                
                corr3d, coll3d, \
                dx, dy, dz, \
                atm_pair3d = self.model.vector_correlations_3d(Sx, Sy, Sz, 
                                                               rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr3d_arrs.append(corr3d)
                coll3d_arrs.append(coll3d)
                
            elif (disorder == 'Occupancy'):
                
                A_r = self.model.load_occupational(self.fname, run)
                
                corr3d, \
                dx, dy, dz, \
                atm_pair3d = self.model.scalar_correlations_3d(A_r, rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr3d_arrs.append(corr3d)
                
            elif (disorder == 'Displacement'):
        
                Ux, Uy, Uz = self.model.load_displacive(self.fname, run)
                
                corr3d, coll3d, \
                corr3d_, coll3d_, \
                dx, dy, dz, \
                atm_pair3d = self.model.vector_correlations_3d(Ux, Uy, Uz, 
                                                               rx, ry, rz,
                                                               atms, fract, 
                                                               tol, period)
                
                corr3d_arrs.append(corr3d)
                coll3d_arrs.append(coll3d)               
                
        stats_corr3d = self.model.correlation_statistics(corr3d_arrs)
        corr3d, sigma_sq_corr3d = stats_corr3d
    
        if (disorder != 'Occupancy'):
            
            stats_coll3d = self.model.correlation_statistics(coll3d_arrs)
            coll3d, sigma_sq_coll3d = stats_coll3d
            
        if self.view.symmetrize_checked():
            
            if (disorder != 'Occupancy'):
                
                data3d = [corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d, 
                          dx, dy, dz, atm_pair3d, A, folder, filename, tol]
            
                symm3d = self.model.vector_symmetrize_3d(*data3d)
                 
                corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d, \
                dx, dy, dz, atm_pair3d = symm3d
        
            else:
                
                data3d = [corr3d, sigma_sq_corr3d, dx, dy, dz, 
                          atm_pair3d, A, folder, filename, tol]
                       
                symm3d = self.model.scalar_symmetrizes_3d(*data3d)

                corr3d, sigma_sq_corr3d, dx, dy, dz, atm_pair3d = symm3d
                
        if self.view.average_3d_checked():
            
            if (disorder != 'Occupancy'):
                
                data3d = [corr3d, coll3d, sigma_sq_corr3d, 
                          sigma_sq_coll3d, dx, dy, dz, tol]
            
                ave3d = self.model.vector_average_3d(*data3d)
                
                corr3d, coll3d, sigma_sq_corr3d, \
                sigma_sq_coll3d, dx, dy, dz = ave3d
            
            else:
                
                data3d = [corr3d, sigma_sq_corr3d, dx, dy, dz, tol]
                
                ave3d = self.model.scalar_average_3d(*data3d)
                
                corr3d, sigma_sq_corr3d, dx, dy, dz = ave3d
         
        self.dx, self.dy, self.dz, self.atm_pair3d = dx, dy, dz, atm_pair3d
        
        self.corr3d, self.sigma_sq_corr3d = corr3d, sigma_sq_corr3d
        
        if (disorder != 'Occupancy'):
                
            self.coll3d, self.sigma_sq_coll3d = coll3d, sigma_sq_coll3d

    def plot_3d(self):
        
        disorder = self.view.get_correlations_3d()
        correlation = self.view.get_plot_3d()
        norm = self.get_norm_3d()
        
        tol = self.view.get_tol_3d()
        
        average = self.view.average_3d_checked()
        
        h, k, l = self.view.get_h(), self.view.get_k(), self.view.get_l()
        d = self.view.get_d()
        
        A, B = self.A, self.B
        
        dx, dy, dz, atm_pair3d = self.dx, self.dy, self.dz, self.atm_pair3d 

        if (correlation == 'Correlation'):
            data = self.corr3d
            error = self.sigma_sq_corr3d
        else:
            data = self.coll3d
            error = self.sigma_sq_coll3d

        canvas = self.view.get_plot_3d_canvas()
        
        atoms, pairs = [], []
        for i in range(self.get_pairs_3d_table_row_count()):
            left, right, active = self.view.get_pairs_3d_table_row(i)
            if active:
                atoms.append(left)
                pairs.append(right)
        
        plots.correlations_3d(canvas, dx, dy, dz, h, k, l, d, data, error, 
                              atm_pair3d, disorder, correlation, average, 
                              norm, atoms, pairs, A, B, tol)