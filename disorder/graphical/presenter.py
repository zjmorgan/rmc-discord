#!/usr/bin/env/python3

import numpy as np

class Presenter:

    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.finished_editing_nu(self.supercell_n)
        self.view.finished_editing_nv(self.supercell_n)
        self.view.finished_editing_nw(self.supercell_n)
        
        self.view.index_changed_type(self.change_type)
        self.view.index_changed_parameters(self.change_parameters)
        
        self.view.button_clicked_CIF(self.load_CIF)
        self.view.select_site(self.select_highlight)

    def supercell_n(self):
        
        n_atm = self.view.get_n_atm()
        if (n_atm is not None):
            nu = self.view.get_nu()
            nv = self.view.get_nv()
            nw = self.view.get_nw()
            n = self.model.supercell_size(n_atm, nu, nv, nw)
            self.view.set_n(n)
            
    def change_type(self):
        
        disorder_type = self.view.get_type()
        if (disorder_type == 'Neutron'):
            self.view.add_item_magnetic()
            bc_symbols = self.model.get_neutron_scattering_lengths()
            self.view.add_atom_combo(bc_symbols)
            mag_symbols = self.model.get_magnetic_form_factors()
            self.view.add_ion_combo(mag_symbols)
        else:
            self.view.remove_item_magnetic()
            X_symbols = self.model.get_xray_form_factors()
            self.view.add_ion_combo(X_symbols)
    
        self.change_parameters()
    
    def change_parameters(self):
        
        n_cols = self.view.get_atom_site_table_col_count()
        if (n_cols > 0):
            self.view.show_atom_site_table_cols()
            self.view.show_unit_cell_table_cols()
            self.view.clear_atom_site_table_selection()
            self.view.clear_unit_cell_table_selection()
    
    def select_highlight(self, item):
        
        i, j = item.row(), item.column()
        sites = self.view.get_every_site().astype(int)-1
        rows = np.argwhere(sites == i)
        if (len(rows) > 0):
            row_range = [rows.min(), rows.max()]
            col = self.view.unit_site_col(j)
            self.view.highlight_atoms(row_range, col)
            
    def add_remove_atoms(self):
        
        n_atm = self.view.change_site_check()
        nu = self.view.get_nu()
        nv = self.view.get_nv()
        nw = self.view.get_nw()
        self.view.set_n_atm(n_atm)
        self.view.set_n(self.model.supercell_size(n_atm, nu, nv, nw))
        
    def load_CIF(self):
        
        name = self.view.open_dialog_cif()
        
        if name:
                        
            folder, filename = name.rsplit('/', 1)
            folder += '/'
            
            a, \
            b, \
            c, \
            alpha, \
            beta, \
            gamma = self.model.load_lattice_parameters(folder, filename)
            
            lat = self.model.find_lattice(a, b, c, alpha, beta, gamma)
            
            self.view.set_lattice(lat)
            
            alpha = np.round(np.rad2deg(alpha), 4)
            beta = np.round(np.rad2deg(beta), 4)
            gamma = np.round(np.rad2deg(gamma), 4)
            
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
            atms, \
            n_atm = self.model.load_unit_cell(folder, filename)
            
            U11, U22, U33, U23, U13, U12 = np.round(displacement.T, 4)
            mu1, mu2, mu3 = np.round(moment.T, 4)
            
            A, B, R, D = self.model.crystal_matrices(a, b, c, 
                                                     alpha, beta, gamma)
            
            Uiso = self.model.atomic_displacement_parameters(U11, U22, U33, 
                                                             U23, U13, U12, D)
                        
            mu = self.model.magnetic_moments(mu1, mu2, mu3, D)
            
            Uiso = np.round(Uiso, 4)
            mu = np.round(mu, 4)
            
            self.view.set_n_atm(n_atm)
                        
            uni, ind, inv = np.unique(site, 
                                      return_index=True, 
                                      return_inverse=True)
            
            n_sites = ind.size
            
            g = np.full(n_atm, 2.0)
            
            self.view.create_atom_site_table(n_sites)
            self.view.show_atom_site_table_cols()
            
#            bc_symbols = self.model.get_neutron_scattering_lengths()
#            self.view.add_atom_combo(bc_symbols)
#            
#            X_symbols = self.model.get_xray_form_factors()
#            self.view.add_ion_combo(X_symbols)
#                
#            mag_symbols = self.model.get_magnetic_form_factors()
#            self.view.add_ion_combo(mag_symbols)
            
            self.change_type()
                        
            self.view.set_atom_combo(atms[ind])
            self.view.set_ion_combo(atms[ind])
            
            self.view.set_atom_site_occupancy(occupancy[ind])            
            self.view.set_atom_site_Uiso(Uiso[ind]) 
            self.view.set_atom_site_U11(U11[ind]) 
            self.view.set_atom_site_U22(U22[ind]) 
            self.view.set_atom_site_U33(U33[ind]) 
            self.view.set_atom_site_U23(U23[ind]) 
            self.view.set_atom_site_U13(U13[ind]) 
            self.view.set_atom_site_U12(U12[ind]) 
            self.view.set_atom_site_U11(U11[ind]) 
            self.view.set_atom_site_mu(mu[ind]) 
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
            self.view.set_unit_cell_occupancy(occupancy)
            self.view.set_unit_cell_Uiso(Uiso) 
            self.view.set_unit_cell_U11(U11) 
            self.view.set_unit_cell_U22(U22) 
            self.view.set_unit_cell_U33(U33) 
            self.view.set_unit_cell_U23(U23) 
            self.view.set_unit_cell_U13(U13) 
            self.view.set_unit_cell_U12(U12) 
            self.view.set_unit_cell_U11(U11) 
            self.view.set_unit_cell_mu(mu) 
            self.view.set_unit_cell_mu1(mu1) 
            self.view.set_unit_cell_mu2(mu2) 
            self.view.set_unit_cell_mu3(mu3) 
            self.view.set_unit_cell_u(u) 
            self.view.set_unit_cell_v(v)
            self.view.set_unit_cell_w(w)
            self.view.format_unit_cell_table()

            nu = self.view.get_nu()
            nv = self.view.get_nv()
            nw = self.view.get_nw()
            
            self.view.set_n(self.model.supercell_size(n_atm, nu, nv, nw))
            self.view.check_clicked_site(self.add_remove_atoms)