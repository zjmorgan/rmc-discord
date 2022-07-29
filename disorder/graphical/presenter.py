#!/usr/bin/env/python3

import os
import sys

import re

import numpy as np

from disorder.graphical import plots
from disorder.graphical.plots import Line

class Presenter:

    def __init__(self, model, view):

        self.model = model
        self.view = view

        self.fname = ''
        self.fname_cif = ''
        self.fname_exp = ''

        self.view.new_triggered(self.new_application)
        self.view.save_as_triggered(self.save_as_application)
        self.view.save_triggered(self.save_application)
        self.view.open_triggered(self.load_application)
        self.view.exit_triggered(self.exit_application)

        self.view.finished_editing_nu(self.supercell_n)
        self.view.finished_editing_nv(self.supercell_n)
        self.view.finished_editing_nw(self.supercell_n)

        self.view.finished_editing_a(self.edit_lattice)
        self.view.finished_editing_b(self.edit_lattice)
        self.view.finished_editing_c(self.edit_lattice)
        self.view.finished_editing_alpha(self.edit_lattice)
        self.view.finished_editing_beta(self.edit_lattice)
        self.view.finished_editing_gamma(self.edit_lattice)

        self.view.index_changed_type(self.change_type)
        self.view.index_changed_parameters(self.change_parameters)

        self.view.button_clicked_CIF(self.load_CIF)
        self.view.button_clicked_NXS(self.load_NXS)
        self.view.select_site(self.select_highlight)

        self.view.clicked_batch(self.check_batch)
        self.view.clicked_batch_1d(self.check_batch_1d)
        self.view.clicked_batch_3d(self.check_batch_3d)
        self.view.clicked_batch_calc_1d(self.check_batch_calc_1d)
        self.view.clicked_batch_calc_3d(self.check_batch_calc_3d)

        self.view.clicked_disorder_mag(self.disorder_check_mag)
        self.view.clicked_disorder_occ(self.disorder_check_occ)
        self.view.clicked_disorder_dis(self.disorder_check_dis)

        self.view.clicked_run(self.run_refinement)
        self.view.clicked_stop(self.stop_refinement)
        self.view.clicked_reset(self.reset_refinement)
        self.view.clicked_continue(self.continue_refinement)

        self.view.index_changed_correlations_1d(self.change_type_1d)
        self.view.index_changed_correlations_3d(self.change_type_3d)

        self.view.button_clicked_calculate_1d(self.calculate_correlations_1d)
        self.view.button_clicked_calculate_3d(self.calculate_correlations_3d)

        self.view.finished_editing_min_ref(self.draw_plot_ref)
        self.view.finished_editing_max_ref(self.draw_plot_ref)
        self.view.finished_editing_slice(self.draw_plot_ref)

        self.view.index_changed_slice_hkl(self.redraw_plot_ref)
        self.view.index_changed_plot_ref(self.redraw_plot_ref)
        self.view.index_changed_norm_ref(self.redraw_plot_ref)

        self.view.index_changed_plot_top_chi_sq(self.draw_plot_chi_sq)
        self.view.index_changed_plot_bottom_chi_sq(self.draw_plot_chi_sq)

        self.view.index_changed_norm_1d(self.plot_1d)
        self.view.index_changed_norm_3d(self.plot_3d)

        self.view.index_changed_plot_1d(self.plot_1d)
        self.view.index_changed_plot_3d(self.plot_3d)

        self.view.finished_editing_h(self.plot_3d)
        self.view.finished_editing_k(self.plot_3d)
        self.view.finished_editing_l(self.plot_3d)
        self.view.finished_editing_d(self.plot_3d)

        self.view.clicked_disorder_mag_recalc_1d(
            self.disorder_check_mag_recalc_1d
        )
        self.view.clicked_disorder_occ_recalc_1d(
            self.disorder_check_occ_recalc_1d
        )
        self.view.clicked_disorder_dis_recalc_1d(
            self.disorder_check_dis_recalc_1d
        )

        self.view.clicked_disorder_mag_recalc_3d(
            self.disorder_check_mag_recalc_3d
        )
        self.view.clicked_disorder_occ_recalc_3d(
            self.disorder_check_occ_recalc_3d
        )
        self.view.clicked_disorder_dis_recalc_3d(
            self.disorder_check_dis_recalc_3d
        )
        self.view.clicked_disorder_struct_recalc_3d(
            self.disorder_check_struct_recalc_3d
        )

        self.view.index_changed_norm_calc_1d(self.draw_plot_calc_1d)
        self.view.index_changed_marker_calc_1d(self.draw_plot_calc_1d)

        self.view.finished_editing_min_calc_3d(self.draw_plot_calc_3d)
        self.view.finished_editing_max_calc_3d(self.draw_plot_calc_3d)
        self.view.finished_editing_slice_calc_3d(self.draw_plot_calc_3d)

        self.view.index_changed_profiles_calc_1d(self.redraw_plot_calc_1d)
        self.view.index_changed_slice_hkl_calc_3d(self.redraw_plot_calc_3d)

        self.view.index_changed_norm_calc_1d(self.redraw_plot_calc_1d)
        self.view.index_changed_norm_calc_3d(self.redraw_plot_calc_3d)

        self.view.button_clicked_calc_1d(self.recalculate_intensity_1d)
        self.view.button_clicked_calc_3d(self.recalculate_intensity_3d)

        # ---

        self.view.button_clicked_save_intensity_exp(self.save_intensity_exp)
        self.view.button_clicked_save_intensity_ref(self.save_intensity_ref)
        self.view.button_clicked_save_chi_sq(self.save_chi_sq)
        self.view.button_clicked_save_1d(self.save_correlations_1d)
        self.view.button_clicked_save_3d(self.save_correlations_3d)
        self.view.button_clicked_save_calc_3d(self.save_intensity_calc_3d)

        self.view.button_clicked_save_CIF(self.save_CIF)
        self.view.button_clicked_save_dis_CIF(self.save_dis_CIF)
        self.view.button_clicked_save_CSV(self.save_correlations_CSV)
        self.view.button_clicked_save_VTK(self.save_correlations_VTK)

        self.view.button_clicked_save_recalc_CSV(self.save_intens_CSV)
        self.view.button_clicked_save_recalc_VTK(self.save_intens_VTK)

        # ---

        self.folder = '.'

        self.magnetic = False
        self.occupational = False
        self.displacive = False

        self.allocated = False
        self.iteration = 0

        self.ref = None
        self.threadpool = self.view.create_thread_pool()

        self.ds_total_intensity = None
        self.ds_bragg_intensity = None
        self.ds_diffuse_intensity = None

        self.intensity = None

    def new_application(self):

        self.view.clear_application()

    def save_as_application(self):

        self.fname = self.view.open_dialog_save(self.folder)
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

        self.fname = fname.split('.ini')[0]

        if (self.view.get_atom_site_table_row_count() > 0):
            self.model.save_crystal(self.fname_cif, self.fname+'.cif')

        if (self.view.get_experiment_table_row_count() > 0):
            signal_raw = self.signal_raw_m.data
            error_sq_raw = self.error_sq_raw_m.data

            h_range = self.h_range_raw_m
            k_range = self.k_range_raw_m
            l_range = self.l_range_raw_m

            nh, nk, nl = self.nh_raw_m, self.nk_raw_m, self.nl_raw_m

            self.model.save_data(self.fname, signal_raw, error_sq_raw,
                                 h_range, k_range, l_range, nh, nk, nl)

            signal, error_sq = self.signal_m.data, self.error_sq_m.data

            self.model.save_region_of_interest(self.fname, signal, error_sq)

        if (self.view.get_pairs_1d_table_row_count() > 0):
            disorder = self.view.get_correlations_1d()
            if (disorder != 'Occupancy'):
                self.model.save_vector_1d(self.fname, self.corr1d, self.coll1d,
                                          self.sigma_sq_corr1d,
                                          self.sigma_sq_coll1d, self.d,
                                          self.atm_pair1d)
            else:
                self.model.save_scalar_1d(self.fname, self.corr1d,
                                          self.sigma_sq_corr1d, self.d,
                                          self.atm_pair1d)

        if (self.view.get_pairs_3d_table_row_count() > 0):
            disorder = self.view.get_correlations_3d()
            if (disorder != 'Occupancy'):
                self.model.save_vector_3d(self.fname, self.corr3d, self.coll3d,
                                          self.sigma_sq_corr3d,
                                          self.sigma_sq_coll3d,
                                          self.dx, self.dy, self.dz,
                                          self.atm_pair3d)
            else:
                self.model.save_scalar_3d(self.fname, self.corr3d,
                                          self.sigma_sq_corr3d,
                                          self.dx, self.dy, self.dz,
                                          self.atm_pair3d)

        if (self.view.get_atom_site_recalculation_3d_row_count() > 0):
            if (self.intensity is not None):
                self.model.save_recalculation_3d(self.fname, self.intensity)

        if (self.view.get_atom_site_recalculation_1d_row_count() > 0):
            if (self.ds_total_intensity is not None):
                self.model.save_recalculation_1d(self.fname,
                                                 (self.ds_total_intensity,
                                                  self.ds_bragg_intensity,
                                                  self.ds_diffuse_intensity))

    def load_application(self):

        self.fname = self.view.open_dialog_load(self.folder)
        if self.fname:
            self.view.clear_application()
            self.view.load_widgets(self.fname)
            fname = self.fname
            if not fname.endswith('.ini'): fname += '.ini'
            self.fname = fname.split('.ini')[0]
            fname = self.fname
            if (self.view.get_atom_site_table_row_count() > 0):
                atom = self.view.get_atom_combo()
                ion = self.view.get_ion_combo()
                self.connect_table_signals()
                self.view.set_atom_combo(atom)
                self.view.set_ion_combo(ion)
                self.view.change_site_check()
                lat = self.view.get_lattice()
                self.lattice_variables(lat)
                self.fname_cif = fname+'.cif'
            if (self.view.get_experiment_table_row_count() > 0):
                self.load_data_thread('{}-intensity.npz'.format(fname), None)
                signal, error_sq =self.model.load_region_of_interest(fname)
                self.signal_m = self.model.mask_array(signal)
                self.error_sq_m = self.model.mask_array(error_sq)
                self.view.format_experiment_table()
                self.view.format_recalculation_1d_table()
                self.view.format_recalculation_3d_table()
                self.connect_experiment_buttons()
                self.connect_experiment_table_signals()
            if (self.view.get_progress() > 0 or self.view.get_run() > 0):
                run = self.view.get_run()
                if (self.view.get_progress() == 100): run -= 1
                Sx, Sy, Sz = self.model.load_magnetic(fname, run)
                A_r = self.model.load_occupational(fname, run)
                Ux, Uy, Uz = self.model.load_displacive(fname, run)
                self.Sx, self.Sy, self.Sz = Sx, Sy, Sz
                self.A_r = A_r
                self.Ux, self.Uy, self.Uz = Ux, Uy, Uz
                self.preprocess_supercell()
                self.initialize_intensity()
                self.filter_sigma()
                stats = self.model.load_refinement(fname, run)
                self.I_obs, self.chi_sq, self.energy, self.temperature, \
                self.scale, self.acc_moves, self.rej_moves, \
                self.acc_temps, self.rej_temps = stats
                self.iteration = len(self.scale) // (self.n_uvw*self.n_atm)
                scale = self.scale[-1]
                self.refinement_m = self.model.mask_array(self.I_obs*scale)
                self.allocated = True
                plot_type = self.view.get_plot_ref()
                if (plot_type == 'Calculated'):
                    self.ref_arr_m = self.refinement_m
                elif (plot_type == 'Experimental'):
                    self.ref_arr_m = self.signal_m
                else:
                    self.ref_arr_m = self.error_sq_m
                self.draw_plot_ref()
                self.draw_plot_chi_sq()
                technique = self.view.get_type()
                self.view.set_type_recalc(technique)
                magnetic = True if technique == 'Neutron' else False
                self.view.enable_disorder_mag_recalc_1d(magnetic)
                self.view.enable_disorder_occ_recalc_1d(True)
                self.view.enable_disorder_dis_recalc_1d(True)
                self.view.enable_disorder_mag_recalc_3d(magnetic)
                self.view.enable_disorder_occ_recalc_3d(True)
                self.view.enable_disorder_dis_recalc_3d(True)
            if (self.view.get_pairs_1d_table_row_count() > 0):
                disorder = self.view.get_correlations_1d()
                if (disorder != 'Occupancy'):
                    self.corr1d, self.coll1d,
                    self.sigma_sq_corr1d, self.sigma_sq_coll1d, self.d, \
                    self.atm_pair1d = self.model.load_vector_1d(self.fname)
                else:
                    self.corr1d, self.sigma_sq_corr1d, self.d, \
                    self.atm_pair1d = self.model.load_scalar_1d(self.fname)
                visible = False if self.view.get_average_1d() else True
                self.view.enable_pairs_1d(visible)
                self.view.check_clicked_pairs_1d(self.plot_1d)
                self.view.format_pairs_1d_table()
                self.plot_1d()
            if (self.view.get_pairs_3d_table_row_count() > 0):
                disorder = self.view.get_correlations_3d()
                if (disorder != 'Occupancy'):
                    self.corr3d, self.coll3d, \
                    self.sigma_sq_corr3d, self.sigma_sq_coll3d, \
                    self.dx, self.dy, self.dz, \
                    self.atm_pair3d = self.model.load_vector_3d(self.fname)
                else:
                     self.corr3d, self.sigma_sq_corr3d, \
                     self.dx, self.dy, self.dz, \
                     self.atm_pair3d = self.model.load_scalar_3d(self.fname)
                visible = False if self.view.get_average_3d() else True
                self.view.enable_pairs_3d(visible)
                self.view.check_clicked_pairs_3d(self.plot_3d)
                self.view.format_pairs_3d_table()
                self.plot_3d()
            if (self.view.get_atom_site_recalculation_3d_row_count() > 0):
                self.view.format_atom_site_recalculation_3d_table()
                intensity = self.model.load_recalculation_3d(fname)
                if (intensity is not None):
                    self.intensity = intensity
                    self.redraw_plot_calc_3d()
                self.view.item_changed_recalculation_3d_table(
                    self.update_recalculation_3d_table
                )
            if (self.view.get_atom_site_recalculation_1d_row_count() > 0):
                self.view.format_atom_site_recalculation_1d_table()
                ds_total_intensity, ds_bragg_intensity, \
                ds_diffuse_intensity = self.model.load_recalculation_1d(fname)
                if (ds_total_intensity is not None):
                    self.ds_total_intensity = ds_total_intensity
                    self.ds_bragg_intensity = ds_bragg_intensity
                    self.ds_diffuse_intensity = ds_diffuse_intensity
                    self.redraw_plot_calc_1d()
                self.view.item_changed_recalculation_1d_table(
                    self.update_recalculation_1d_table
                )

    def save_intensity_exp(self):

        filename = self.view.save_intensity_exp(self.folder)

        if filename:

            if (filename.endswith('.png') or filename.endswith('.pdf')):
                parse = filename.split('.')
                filename, ext = '.'.join(parse[0:-1]), parse[-1]
            else:
                ext =  '.pdf'

            fig_h = self.view.canvas_exp_h.figure
            fig_k = self.view.canvas_exp_k.figure
            fig_l = self.view.canvas_exp_l.figure

            fig_h.savefig(filename+'-0kl'+ext)
            fig_k.savefig(filename+'-h0l'+ext)
            fig_l.savefig(filename+'-hk0'+ext)

    def save_intensity_ref(self):

        filename = self.view.save_intensity_ref(self.folder)

        if filename:

            if (filename.endswith('.png') or filename.endswith('.pdf')):
                parse = filename.split('.')
                filename, ext = '.'.join(parse[0:-1]), parse[-1]
            else:
                ext =  '.pdf'

            fig = self.view.canvas_ref.figure
            fig.savefig(filename+ext)

    def save_chi_sq(self):

        filename = self.view.save_chi_sq()

        if filename:

            if (filename.endswith('.png') or filename.endswith('.pdf')):
                parse = filename.split('.')
                filename, ext = '.'.join(parse[0:-1]), parse[-1]
            else:
                ext =  '.pdf'

            fig = self.view.canvas_chi_sq.figure
            fig.savefig(filename+ext)

    def save_correlations_1d(self):

        filename = self.view.save_correlations_1d(self.folder)

        if filename:

            if (filename.endswith('.png') or filename.endswith('.pdf')):
                parse = filename.split('.')
                filename, ext = '.'.join(parse[0:-1]), parse[-1]
            else:
                ext =  '.pdf'

            fig = self.view.canvas_1d.figure
            fig.savefig(filename+ext)

    def save_correlations_3d(self):

        filename = self.view.save_correlations_3d(self.folder)

        if filename:

            if (filename.endswith('.png') or filename.endswith('.pdf')):
                parse = filename.split('.')
                filename, ext = '.'.join(parse[0:-1]), parse[-1]
            else:
                ext =  '.pdf'

            fig = self.view.canvas_3d.figure
            fig.savefig(filename+ext)

    def save_intensity_calc_3d(self):

        filename = self.view.save_intensity_calc_3d(self.folder)

        if filename:

            fig = self.view.canvas_calc_3d.figure
            fig.savefig(filename)

    def save_CIF(self):

        if (self.view.get_unit_cell_table_row_count() > 0):

            fname = self.view.save_CIF(self.folder)

            if fname:

                if (not fname.endswith('.cif')): fname += '.cif'

                folder, filename = self.fname_cif.rsplit('/', 1)

                atm = self.view.get_atom()

                u = self.view.get_u()
                v = self.view.get_v()
                w = self.view.get_w()

                nu = self.view.get_nu()
                nv = self.view.get_nv()
                nw = self.view.get_nw()

                occ = self.view.get_occupancy()

                U11 = self.view.get_U11()
                U22 = self.view.get_U22()
                U33 = self.view.get_U33()
                U23 = self.view.get_U23()
                U13 = self.view.get_U13()
                U12 = self.view.get_U12()

                disp = np.column_stack((U11,U22,U33,U23,U13,U12))

                mu1 = self.view.get_mu1()
                mu2 = self.view.get_mu2()
                mu3 = self.view.get_mu3()

                mom = np.column_stack((mu1,mu2,mu3))

                self.model.save_supercell(fname, atm, occ, disp, mom, u, v, w,
                                          nu, nv, nw, folder, filename)

    def save_dis_CIF(self):

        if self.allocated:

            fname = self.view.save_CIF(self.folder)

            if fname:

                if (not fname.endswith('.cif')): fname += '.cif'

                folder, filename = self.fname_cif.rsplit('/', 1)

                Sx, Sy, Sz = self.Sx, self.Sy, self.Sz

                occupancy = self.occupancy

                A_r = self.A_r

                Ux, Uy, Uz = self.Ux, self.Uy, self.Uz

                n_atm = self.n_atm

                delta = ((A_r.reshape(A_r.size // n_atm, n_atm)+1)*occupancy)
                delta = delta.flatten()

                rx, ry, rz = self.rx, self.ry, self.rz

                atm = self.atm

                A = self.A

                nu, nv, nw = self.nu, self.nv, self.nw

                self.model.save_disorder(fname, Sx, Sy, Sz, delta, Ux, Uy, Uz,
                                         rx, ry, rz, nu, nv, nw, atm, A,
                                         folder, filename)

    def save_correlations_CSV(self):

        if (self.view.get_pairs_1d_table_row_count() > 0):

            filename = self.view.save_correlations_CSV(self.folder)

            if filename:

                if (not filename.endswith('.csv')): filename += '.csv'

                disorder = self.view.get_correlations_1d()
                average = self.view.average_1d_checked()

                if (disorder != 'Occupancy'):
                    if average:
                        data = self.d, self.corr1d, self.coll1d
                        header = 'd,corr,coll'
                    else:
                        data = self.d, self.corr1d, self.coll1d, \
                               self.atm_pair1d
                        header = 'd,corr,coll,pair'
                else:
                    if average:
                        data = self.d, self.corr1d
                        header = 'd,corr'
                    else:
                        data = self.d, self.corr1d, self.atm_pair1d
                        header = 'd,corr,pair'

                self.model.save_correlations_1d(filename, data, header)

    def save_correlations_VTK(self):

        if (self.view.get_pairs_3d_table_row_count() > 0):

            filename = self.view.save_correlations_VTK(self.folder)

            if filename:

                if (not filename.endswith('.vtm')): filename += '.vtm'

                disorder = self.view.get_correlations_3d()
                average = self.view.average_3d_checked()

                if (disorder != 'Occupancy'):
                    if not average:
                        data = self.dx, self.dy, self.dz, \
                               self.corr3d, self.coll3d
                        label = 'vector-pair'
                    else:
                        data = self.dx, self.dy, self.dz, \
                               self.corr3d, self.coll3d, self.atm_pair3d
                        label = 'vector'
                else:
                    if not average:
                        data = self.dx, self.dy, self.dz, self.corr3d
                        label = 'scalar-pair'
                    else:
                        data = self.dx, self.dy, self.dz, \
                               self.corr3d, self.atm_pair3d
                        label = 'scalar'

                self.model.save_correlations_3d(filename, data, label)
                
    def save_intens_CSV(self):
    
        if (self.view.get_atom_site_recalculation_1d_row_count() > 0):
            
            filename = self.view.save_intens_CSV(self.folder)

            if filename:  
                
                if (not filename.endswith('.csv')): filename += '.csv'

                dQ, nQ, min_Q, max_Q = self.view.get_recalculation_1d_binning()

                Q = np.linspace(min_Q, max_Q, nQ)
                
                total = self.ds_total_intensity
                bragg = self.ds_bragg_intensity
                diffuse = self.ds_diffuse_intensity
                
                intensity = (total, bragg, diffuse)
            
                self.model.save_intensity_1d(filename, Q, intensity)      
                
    def save_intens_VTK(self):
    
        if (self.view.get_atom_site_recalculation_3d_row_count() > 0):
            
            filename = self.view.save_intens_VTK(self.folder)

            if filename:  

                if (not filename.endswith('.vts')): filename += '.vts'
                
                binning_h = self.view.get_recalculation_3d_binning_h()
                binning_k = self.view.get_recalculation_3d_binning_k()
                binning_l = self.view.get_recalculation_3d_binning_l()
                
                nh, min_h, max_h = binning_h[1:]
                nk, min_k, max_k = binning_k[1:]
                nl, min_l, max_l = binning_l[1:]
                
                h = np.linspace(min_h, max_h, nh)
                k = np.linspace(min_k, max_k, nk)
                l = np.linspace(min_l, max_l, nl)
        
                h, k, l = np.meshgrid(h,k,l)
        
                intensity = self.intensity
                B, T = self.B, self.T
                
                Bp = np.dot(B,T)
                
                self.model.save_intensity_3d(filename, h, k, l, intensity, Bp)
            
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
            mu = self.model.magnetic_symmetry([mag_op[row]], mu)[0]
            mu[comp] = float(moment)
            mu = self.model.magnetic_symmetry([mag_op[row]], mu)[0]
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
            coord = self.model.reverse_symmetry([op[row]], coord)[0]
            coord[comp] = float(coordinate)
            coord = self.model.symmetry([op[row]], coord)[0]
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

    def edit_lattice(self):

        a, b, c, alpha, beta, gamma = self.view.get_lattice_parameters()

        lat = self.view.get_lattice()

        if (lat == 'Tetragonal' or lat == 'Hexagonal'):
            b = a
        elif (lat == 'Cubic'):
            c = b = a
        elif (lat == 'Rhobmohedral'):
            c = b = a
            gamma = beta = alpha

        self.view.set_lattice_parameters(a, b, c, alpha, beta, gamma)

    def lattice_variables(self, lat):

        self.view.set_lattice(lat)

        a, b, c, alpha, beta, gamma = self.view.get_lattice_parameters()

        self.view.set_a_visible(False)
        self.view.set_b_visible(False)
        self.view.set_c_visible(False)
        self.view.set_alpha_visible(False)
        self.view.set_beta_visible(False)
        self.view.set_gamma_visible(False)

        if (lat == 'Cubic'):
            self.view.set_a_visible(True)
        elif (lat == 'Hexagonal' or lat == 'Tetragonal'):
            self.view.set_a_visible(True)
            self.view.set_c_visible(True)
        elif (lat == 'Rhobmohedral'):
            self.view.set_a_visible(True)
            self.view.set_alpha_visible(True)
        elif (lat == 'Orthorhombic'):
            self.view.set_a_visible(True)
            self.view.set_b_visible(True)
            self.view.set_c_visible(True)
        elif (lat == 'Monoclinic'):
            self.view.set_a_visible(True)
            self.view.set_b_visible(True)
            self.view.set_c_visible(True)
            if (not np.isclose(beta, np.pi/2)):
                self.view.set_beta_visible(True)
            else:
                self.view.set_gamma_visible(True)
        else:
            self.view.set_a_visible(False)
            self.view.set_b_visible(False)
            self.view.set_c_visible(False)
            self.view.set_alpha_visible(False)
            self.view.set_beta_visible(False)
            self.view.set_gamma_visible(False)

    def load_CIF(self):

        name = self.view.open_dialog_cif(self.folder)

        if name:

            self.view.enable_load_CIF(False)

            self.fname_cif = name

            self.view.clear_unit_cell_table()
            self.view.clear_atom_site_table()

            folder, filename = name.rsplit('/', 1)
            folder += '/'

            parameters = self.model.load_lattice_parameters(folder, filename)
            a, b, c, alpha, beta, gamma = parameters

            lat = self.model.find_lattice(a, b, c, alpha, beta, gamma)

            self.view.set_lattice_parameters(a, b, c, alpha, beta, gamma)

            self.lattice_variables(lat)

            group, hm = self.model.load_space_group(folder, filename)

            self.view.set_space_group(group, hm)

            if (len(hm) > 1):
                centering = hm[0]
                self.view.set_centering(centering)
                self.view.set_centering_ref(centering)
                self.view.set_centering_calc_3d(centering)

            uc_dict = self.model.load_unit_cell(folder, filename)

            u = uc_dict['u']
            v = uc_dict['v']
            w = uc_dict['w']
            occupancy = uc_dict['occupancy']
            displacement = uc_dict['displacement']
            moment = uc_dict['moment']
            site = uc_dict['site']
            op = uc_dict['operator']
            mag_op = uc_dict['magnetic_operator']
            atm = uc_dict['atom']
            n_atm = uc_dict['n_atom']

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

            self.view.enable_load_CIF(True)

    def draw_plot_exp(self):

        canvas_h, canvas_k, canvas_l = self.view.get_plot_exp_canvas()
        data = self.exp_arr_m

        constants = self.view.get_lattice_parameters()
        A, B, R, C, D = self.model.crystal_matrices(*constants)

        Bp = self.model.matrix_transform(B)

        matrix_h = Bp[1:3,1:3].copy()
        matrix_k = Bp[0:3:2,0:3:2].copy()
        matrix_l = Bp[0:2,0:2].copy()

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

        self.view.block_slices()

        self.view.set_slice_h(h)
        self.view.set_slice_k(k)
        self.view.set_slice_l(l)

        self.view.unblock_slices()

        norm = self.view.get_norm_exp()

        vmin = self.view.get_min_exp()
        vmax = self.view.get_max_exp()

        self.view.validate_min_exp()
        self.view.validate_max_exp()

        plot_exp_h = plots.HeatMap(canvas_h)
        plot_exp_k = plots.HeatMap(canvas_k)
        plot_exp_l = plots.HeatMap(canvas_l)

        plot_exp_h.clear_canvas()
        plot_exp_k.clear_canvas()
        plot_exp_l.clear_canvas()

        plot_exp_h.plot_data(data[ih,:,:], min_k, min_l, max_k, max_l)
        plot_exp_k.plot_data(data[:,ik,:], min_h, min_l, max_h, max_l)
        plot_exp_l.plot_data(data[:,:,il], min_h, min_k, max_h, max_k)

        plot_exp_h.set_normalization(vmin, vmax, norm=norm)
        plot_exp_k.set_normalization(vmin, vmax, norm=norm)
        plot_exp_l.set_normalization(vmin, vmax, norm=norm)

        plot_exp_h.create_colorbar(orientation='horizontal')
        plot_exp_k.create_colorbar(orientation='horizontal')
        plot_exp_l.create_colorbar(orientation='horizontal')

        plot_exp_h.transform_axes(matrix_h)
        plot_exp_k.transform_axes(matrix_k)
        plot_exp_l.transform_axes(matrix_l)

        plot_exp_h.set_labels(r'$h={}$'.format(h), r'$(0k0)$', r'$(00l)$')
        plot_exp_k.set_labels(r'$k={}$'.format(k), r'$(h00)$', r'$(00l)$')
        plot_exp_l.set_labels(r'$l={}$'.format(l), r'$(0k0)$', r'$(0k0)$')

        plot_exp_h.tight_layout()
        plot_exp_k.tight_layout()
        plot_exp_l.tight_layout()

        plot_exp_h.draw_canvas()
        plot_exp_k.draw_canvas()
        plot_exp_l.draw_canvas()

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

        data = nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l

        h_range = [min_h, max_h]
        k_range = [min_k, max_k]
        l_range = [min_l, max_l]

        binning = [nh, nk, nl]

        self.view.block_experiment_table_signals()
        self.view.format_experiment_table()

        if   (row == 0): step, size, minimum, maximum = dh, nh, min_h, max_h
        elif (row == 1): step, size, minimum, maximum = dk, nk, min_k, max_k
        elif (row == 2): step, size, minimum, maximum = dl, nl, min_l, max_l

        if (col == 1):
            size = int(text)
            n = self.model.size_value(minimum, maximum, step)
            if   (row == 0): binning[0] = n
            elif (row == 1): binning[1] = n
            elif (row == 2): binning[2] = n
            if (size < n and size > 1):
                self.rebin(data)
            else:
                self.view.set_experiment_binning_h(binning[0], min_h, max_h)
                self.view.set_experiment_binning_k(binning[1], min_k, max_k)
                self.view.set_experiment_binning_l(binning[2], min_l, max_l)
                self.view.format_experiment_table()
                self.view.unblock_experiment_table_signals()
        elif (col == 2):
            minimum = float(text)
            low = self.model.minimum_value(size, step, maximum)
            if   (row == 0): h_range = [low, max_h]
            elif (row == 1): k_range = [low, max_k]
            elif (row == 2): l_range = [low, max_l]
            if (minimum > low and minimum < maximum):
                self.crop(data, h_range, k_range, l_range)
            else:
                self.view.set_experiment_binning_h(nh, h_range[0], max_h)
                self.view.set_experiment_binning_k(nk, k_range[0], max_k)
                self.view.set_experiment_binning_l(nl, l_range[0], max_l)
                self.view.format_experiment_table()
                self.view.unblock_experiment_table_signals()
        elif (col == 3):
            maximum = float(text)
            high = self.model.maximum_value(size, step, minimum)
            if   (row == 0): h_range = [min_h, high]
            elif (row == 1): k_range = [min_k, high]
            elif (row == 2): l_range = [min_l, high]
            if (maximum < high and maximum > minimum):
                self.crop(data, h_range, k_range, l_range)
            else:
                self.view.set_experiment_binning_h(nh, min_h, h_range[1])
                self.view.set_experiment_binning_k(nk, min_k, k_range[1])
                self.view.set_experiment_binning_l(nl, min_l, l_range[1])
                self.view.format_experiment_table()
                self.view.unblock_experiment_table_signals()

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

    def rebin_thread(self, data, callback):

        signal = self.signal_m
        error_sq = self.error_sq_m

        nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l = data

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
        self.view.format_experiment_table()

    def rebin_complete(self):

        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()

        self.view.format_experiment_table()
        self.redraw_plot_exp()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def rebin(self, data):

        self.view.enable_cropbin_signals(False)

        self.rebin_data = self.view.worker(self.rebin_thread, data)
        self.view.result(self.rebin_data, self.rebin_process_output)
        self.view.finished(self.rebin_data, self.rebin_complete)
        self.threadpool.start(self.rebin_data)

    def crop_thread(self, data, h_range, k_range, l_range, callback):

        signal = self.signal_m
        error_sq = self.error_sq_m

        nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l = data

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

        min_h = self.model.slice_value(h_range[0], h_range[1], nh, ih_min)
        min_k = self.model.slice_value(k_range[0], k_range[1], nk, ik_min)
        min_l = self.model.slice_value(l_range[0], l_range[1], nl, il_min)

        max_h = self.model.slice_value(h_range[0], h_range[1], nh, ih_max)
        max_k = self.model.slice_value(k_range[0], k_range[1], nk, ik_max)
        max_l = self.model.slice_value(l_range[0], l_range[1], nl, il_max)

        nh = h_slice[1]-h_slice[0]
        nk = k_slice[1]-k_slice[0]
        nl = l_slice[1]-l_slice[0]

        return nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l

    def crop_process_output(self, data):

        nh, nk, nl, min_h, min_k, min_l, max_h, max_k, max_l = data

        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)
        self.view.format_experiment_table()

    def crop_complete(self):

        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()

        self.view.format_experiment_table()
        self.redraw_plot_exp()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def crop(self, data, h_range, k_range, l_range):

        self.view.enable_cropbin_signals(False)

        self.crop_data = self.view.worker(self.crop_thread, data,
                                          h_range, k_range, l_range)
        self.view.result(self.crop_data, self.crop_process_output)
        self.view.finished(self.crop_data, self.crop_complete)
        self.threadpool.start(self.crop_data)

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

        self.view.enable_cropbin_signals(False)
        self.view.block_experiment_table_signals()

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
        self.view.format_experiment_table()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

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

        self.signal_m = self.model.crop(signal, h_slice, k_slice, l_slice)
        self.error_sq_m = self.model.crop(error_sq, h_slice, k_slice, l_slice)

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.rebin(signal, binsize)
        self.error_sq_m = self.model.rebin(error_sq, binsize)

    def reset_data_h(self):

        self.view.enable_cropbin_signals(False)
        self.view.block_experiment_table_signals()

        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()

        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()

        nh = self.nh_raw_m

        min_h, max_h = self.h_range_raw_m

        self.cropbin([min_h, max_h], [min_k, max_k],
                     [min_l, max_l], [nh, nk, nl])

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

        self.view.create_experiment_table()

        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)

        self.view.format_experiment_table()

        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()

        self.connect_experiment_table_signals()
        self.view.format_experiment_table()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def reset_data_k(self):

        self.view.enable_cropbin_signals(False)
        self.view.block_experiment_table_signals()

        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()

        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()

        nk = self.nk_raw_m

        min_k, max_k = self.k_range_raw_m

        self.cropbin([min_h, max_h], [min_k, max_k],
                     [min_l, max_l], [nh, nk, nl])

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

        self.view.create_experiment_table()

        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)

        self.view.format_experiment_table()

        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()

        self.connect_experiment_table_signals()
        self.view.format_experiment_table()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def reset_data_l(self):

        self.view.enable_cropbin_signals(False)
        self.view.block_experiment_table_signals()

        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()

        self.view.clear_experiment_table()

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()

        nl = self.nl_raw_m

        min_l, max_l = self.l_range_raw_m

        self.cropbin([min_h, max_h], [min_k, max_k],
                     [min_l, max_l], [nh, nk, nl])

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

        self.view.create_experiment_table()

        self.view.set_experiment_binning_h(nh, min_h, max_h)
        self.view.set_experiment_binning_k(nk, min_k, max_k)
        self.view.set_experiment_binning_l(nl, min_l, max_l)

        self.view.format_experiment_table()

        self.populate_binning()
        self.populate_cropping()
        self.populate_slicing()

        self.connect_experiment_table_signals()
        self.view.format_experiment_table()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def punch_thread(self, callback):

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
                         h_range, k_range, l_range, centering, outlier, punch)

        self.signal_m = signal
        self.error_sq_m = error_sq

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

    def punch_complete(self):

        self.redraw_plot_exp()

        self.view.enable_cropbin_signals(True)
        self.view.unblock_experiment_table_signals()

    def punch(self):

        self.view.enable_cropbin_signals(False)

        self.punch_data = self.view.worker(self.punch_thread)
        self.view.finished(self.punch_data, self.punch_complete)
        self.threadpool.start(self.punch_data)

    def reset_punch(self):

        self.view.enable_cropbin_signals(False)

        self.signal_m = self.signal_raw_m.copy()
        self.error_sq_m = self.error_sq_raw_m.copy()

        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()

        self.cropbin([min_h, max_h], [min_k, max_k],
                     [min_l, max_l], [nh, nk, nl])

        signal = self.signal_m
        error_sq = self.error_sq_m

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

        self.redraw_plot_exp()

        self.view.enable_cropbin_signals(True)

    def populate_recalculation_1d_table(self):

        dQ, nQ, min_Q, max_Q = 0.1, 41, 0.1, 1.0

        self.view.create_recalculation_1d_table(dQ, nQ, min_Q, max_Q)

        self.view.format_recalculation_1d_table()

        self.view.item_changed_recalculation_1d_table(
            self.update_recalculation_1d_table
        )

    def update_recalculation_1d_table(self):

        dQ, nQ, min_Q, max_Q = self.view.get_recalculation_1d_binning()

        self.view.block_recalculation_1d_table_signals()

        self.view.set_recalculation_1d_binning(nQ, min_Q, max_Q)

        self.view.format_recalculation_1d_table()
        self.view.unblock_recalculation_1d_table_signals()

    def populate_recalculation_3d_table(self):

        dh, nh, min_h, max_h = self.view.get_experiment_binning_h()
        dk, nk, min_k, max_k = self.view.get_experiment_binning_k()
        dl, nl, min_l, max_l = self.view.get_experiment_binning_l()

        self.view.create_recalculation_3d_table(dh, nh, min_h, max_h,
                                                dk, nk, min_k, max_k,
                                                dl, nl, min_l, max_l)

        self.view.format_recalculation_3d_table()

        self.view.item_changed_recalculation_3d_table(
            self.update_recalculation_3d_table
        )

    def update_recalculation_3d_table(self):

        dh, nh, min_h, max_h = self.view.get_recalculation_3d_binning_h()
        dk, nk, min_k, max_k = self.view.get_recalculation_3d_binning_k()
        dl, nl, min_l, max_l = self.view.get_recalculation_3d_binning_l()

        self.view.block_recalculation_3d_table_signals()

        self.view.set_recalculation_3d_binning_h(nh, min_h, max_h)
        self.view.set_recalculation_3d_binning_k(nk, min_k, max_k)
        self.view.set_recalculation_3d_binning_l(nl, min_l, max_l)

        self.view.format_recalculation_3d_table()
        self.view.unblock_recalculation_3d_table_signals()

    def load_data_thread(self, name, callback):

        signal, error_sq, \
        h_range, k_range, l_range, nh, nk, nl = self.model.load_data(name)

        self.signal_m = self.model.mask_array(signal)
        self.error_sq_m = self.model.mask_array(error_sq)

        self.signal_raw_m = self.signal_m.copy()
        self.error_sq_raw_m = self.error_sq_m.copy()

        self.h_range_raw_m = h_range.copy()
        self.k_range_raw_m = k_range.copy()
        self.l_range_raw_m = l_range.copy()

        self.nh_raw_m, self.nk_raw_m, self.nl_raw_m = nh, nk, nl

    def load_data_progress(self):

        pass

    def load_data_complete(self):

        self.reset_data()
        self.connect_experiment_buttons()
        self.populate_recalculation_1d_table()
        self.populate_recalculation_3d_table()
        self.view.enable_load_NXS(True)

    def connect_experiment_buttons(self):

        self.view.button_clicked_reset(self.reset_data)
        self.view.button_clicked_reset_h(self.reset_data_h)
        self.view.button_clicked_reset_k(self.reset_data_k)
        self.view.button_clicked_reset_l(self.reset_data_l)

        self.view.button_clicked_punch(self.punch)
        self.view.button_clicked_reset_punch(self.reset_punch)

    def load_NXS(self):

        if (self.view.get_atom_site_table_col_count() > 0):

            name = self.view.open_dialog_nxs()

            if name:

                self.view.enable_load_NXS(False)

                self.fname_exp = name

                self.load = self.view.worker(self.load_data_thread, name)
                self.view.progress(self.load, self.load_data_progress)
                self.view.finished(self.load, self.load_data_complete)
                self.threadpool.start(self.load)

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

    def check_batch_calc_1d(self):

        visibility = True if self.view.batch_checked_calc_1d() else False
        self.view.enable_runs_calc_1d(visibility)
        if (not visibility): self.view.set_runs_calc_1d(1)

    def check_batch_calc_3d(self):

        visibility = True if self.view.batch_checked_calc_3d() else False
        self.view.enable_runs_calc_3d(visibility)
        if (not visibility): self.view.set_runs_calc_3d(1)

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

        self.view.clear_plot_1d_canvas()
        self.view.clear_pairs_1d_table_table()
        self.view.set_correlations_1d_type()

    def change_type_3d(self):

        self.view.clear_plot_3d_canvas()
        self.view.clear_pairs_3d_table_table()
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
        i_unmask = self.model.reciprocal_space_mapping(h_range,
                                                       k_range,
                                                       l_range,
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

        heisenberg = self.view.get_moment_constraint() == 'Heisenberg'
        isotropic = self.view.get_displacement_constraint() == 'Isotropic'

        self.heisenberg, self.isotropic = heisenberg, isotropic

        mux, muy, muz = self.model.transform_moments(mu1, mu2, mu3, C)

        self.moment = np.row_stack((mux, muy, muz)) if not heisenberg else mu

        u11, u22, u33, u23, u13, u12 = U11, U22, U33, U23, U13, U12

        if isotropic:
            ADPs = self.model.anisotropic_parameters(Uiso, D)
            u11, u22, u33, u23, u13, u12 = ADPs

        L = self.model.decompose_adps(u11, u22, u33, u23, u13, u12, D)

        self.Lxx, self.Lyy, self.Lzz, self.Lyz, self.Lxz, self.Lxy = L

        uxx, uyy, uzz, \
        uyz, uxz, uxx = self.model.transform_adps(u11, u22, u33,
                                                  u23, u13, u12, D)

        self.displacement = np.row_stack((uxx, uyy, uzz, uyz, uxz, uxx))

        # ---

        adp_type = self.view.get_adp_type()

        if (adp_type != 'Anisotropic' or self.view.get_disorder_dis()):
            if (adp_type != 'Isotropic' or self.view.get_disorder_dis()):
                Uiso *= 0

            ADPs = self.model.anisotropic_parameters(Uiso, D)

            U11, U22, U33, U23, U13, U12 = ADPs

        constants_ = self.model.reciprocal_lattice_parameters(*constants)

        a_, b_, c_, alpha_, beta_, gamma_ = constants_

        T = self.model.debye_waller_factors(h_range, k_range, l_range,
                                            nh, nk, nl,
                                            U11, U22, U33, U23, U13, U12,
                                            a_, b_, c_)

        if (self.view.get_type() == 'Neutron'):
             factors = self.model.neutron_factors(Q, element, ion, occupancy,
                                                  T, g, phase_factor)

             self.factors, self.magnetic_factors = factors
        else:
             factors = self.model.xray_factors(Q, ion, occupancy,
                                               T, phase_factor)

             self.factors = factors

        self.p = self.view.get_order()

        self.fixed_mag = self.view.fixed_moment_check()
        self.fixed_occ = self.view.fixed_composition_check()
        self.fixed_dis = self.view.fixed_displacement_check()

    def initialize_intensity(self):

        mask, Q = self.mask, self.Q

        I_arrays = self.model.initialize_intensity(mask, Q)

        self.I_obs, self.I_ref, self.I_calc, self.I_raw, self.I_flat = I_arrays

        filt_arrays = self.model.initialize_filter(mask)

        self.a_filt, self.b_filt, self.c_filt, \
        self.d_filt, self.e_filt, self.f_filt, \
        self.g_filt, self.h_filt, self.i_filt = filt_arrays

        self.refinement_m = self.model.mask_array(self.I_obs)

    def refinement_statistics(self):

        self.acc_moves = []
        self.acc_temps = []
        self.rej_moves = []
        self.rej_temps = []

        self.chi_sq = [np.inf]
        self.energy = []
        self.temperature = [self.view.get_prefactor()]
        self.scale = []

    def populate_atom_site_recalculation_1d_table(self):

        occupancy, Uiso, moment = self.occupancy, self.Uiso, self.mu

        self.view.clear_atom_site_recalculation_1d_table()

        site, atm = self.site, self.atm
        _, ind = np.unique(site, return_index=True)

        data = atm[ind], occupancy[ind], Uiso[ind], moment[ind]

        self.view.create_atom_site_recalculation_1d_table(*data)

    def populate_atom_site_recalculation_3d_table(self):

        occupancy, Uiso, moment = self.occupancy, self.Uiso, self.mu

        self.view.clear_atom_site_recalculation_3d_table()

        site, atm = self.site, self.atm
        _, ind = np.unique(site, return_index=True)

        data = atm[ind], occupancy[ind], Uiso[ind], moment[ind]

        self.view.create_atom_site_recalculation_3d_table(*data)

    def initialize_disorder(self):

        nu, nv, nw, n_atm = self.nu, self.nv, self.nw, self.n_atm

        c = 1 if self.magnetic else 0

        fixed_mag = self.fixed_mag

        moment = self.moment*c

        Sx, Sy, Sz = self.model.random_moments(nu, nv, nw, n_atm,
                                               moment, fixed_mag)

        self.Sx, self.Sy, self.Sz = Sx, Sy, Sz

        b, c = (0, 1) if self.occupational else (1, 0)

        occupancy = self.occupancy*c+b

        A_r = self.model.random_occupancies(nu, nv, nw, n_atm, occupancy)

        self.A_r = A_r

        c = 1 if self.displacive else 0

        fixed_dis = self.fixed_dis

        displacement = self.displacement*c

        Ux, Uy, Uz = self.model.random_displacements(nu, nv, nw, n_atm,
                                                     displacement, fixed_dis)

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
        #h, k, l = self.h, self.k, self.l

        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        indices = self.indices
        factors = self.factors

        p = self.view.get_order()
        centering = self.view.get_centering_ref()

        lat = self.view.get_lattice()

        if (lat == 'Rhombohedral'):
            if (centering == 'R'):
                centering = 'P'

        arrays = self.model.initialize_displacive(Ux, Uy, Uz, H, K, L,
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

    def run_refinement_thread(self, callback):

        batch = self.batch
        runs = self.runs
        cycles = self.cycles

        for b in range(batch, runs):

            if self.magnetic: self.initialize_magnetic()
            if self.occupational: self.initialize_occupational()
            if self.displacive: self.initialize_displacive()

            n = cycles // 1

            iteration = self.iteration
            for i in range(iteration, n):

                self.refinement_cycle()

                p = int(round(100*(i+1)/n))

                if (p != self.progress):

                    if (p <= 0): p = 1
                    elif (p > 100): p = 100

                    x = self.chi_sq[-1]

                    callback.emit([p, b, x])

                self.iteration = i+1

                if not self.ref.proceed():
                    break

            Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
            self.model.save_magnetic(self.fname, b, Sx, Sy, Sz)

            A_r = self.A_r
            self.model.save_occupational(self.fname, b, A_r)

            Ux, Uy, Uz = self.Ux, self.Uy, self.Uz
            self.model.save_displacive(self.fname, b, Ux, Uy, Uz)

            self.model.save_refinement(self.fname, b, self.I_obs, self.chi_sq,
                                       self.energy, self.temperature,
                                       self.scale, self.acc_moves,
                                       self.rej_moves, self.acc_temps,
                                       self.rej_temps)

            if not self.ref.proceed():
                break

    def run_refinement_progress(self, data):

        progress, batch, chi_sq = data

        self.progress = progress

        self.view.set_progress(progress)
        self.view.set_run(batch)
        self.view.set_chi_sq(chi_sq)

        self.fast_redraw_plot_ref()
        self.fast_redraw_plot_chi_sq()

    def run_refinement_complete(self):

        if (self.view.get_progress() == 100):
            batch = self.view.get_run()+1
            self.view.set_run(batch)
            self.iteration = 0

        self.view.enable_refinement(True)
        self.view.enable_reset_refinement(True)
        self.view.enable_continue_refinement(True)

        self.fast_redraw_plot_ref()
        self.fast_redraw_plot_chi_sq()

        self.save_application()

    def continue_refinement(self):

        if (self.view.get_progress() == 100):
            batch = self.view.get_run()-1
            self.view.set_run(batch)
            cycles = self.cycles + self.view.get_cycles()
            self.view.set_cycles(cycles)
            self.iteration = self.cycles
            progress = int(round(100*self.iteration/cycles))
            self.view.set_progress(progress)
            self.run_refinement()

    def run_refinement(self):

        if self.view.get_recalculation_3d_table_row_count():

            self.save_application()

            if self.fname:
                self.magnetic = self.view.get_disorder_mag()
                self.occupational = self.view.get_disorder_occ()
                self.displacive = self.view.get_disorder_dis()

                technique = self.view.get_type()
                self.view.set_type_recalc(technique)

                magnetic = True if technique == 'Neutron' else False

                self.view.enable_disorder_mag_recalc_1d(magnetic)
                self.view.enable_disorder_occ_recalc_1d(True)
                self.view.enable_disorder_dis_recalc_1d(True)

                self.view.enable_disorder_mag_recalc_3d(magnetic)
                self.view.enable_disorder_occ_recalc_3d(True)
                self.view.enable_disorder_dis_recalc_3d(True)

                if (not self.allocated):

                    self.preprocess_supercell()
                    self.initialize_disorder()
                    self.initialize_intensity()
                    self.refinement_statistics()
                    self.filter_sigma()

                    self.allocated = True

                if (not self.view.get_atom_site_recalculation_1d_row_count()):

                    self.populate_atom_site_recalculation_1d_table()

                if (not self.view.get_atom_site_recalculation_3d_row_count()):

                    self.populate_atom_site_recalculation_3d_table()

                self.batch = self.view.get_run()
                self.runs = self.view.get_runs()
                self.cycles = self.view.get_cycles()

                self.progress = self.view.get_progress()

                self.constant = self.view.get_constant()

                self.view.enable_refinement(False)
                self.view.enable_reset_refinement(False)
                self.view.enable_continue_refinement(False)

                self.redraw_plot_ref()
                self.draw_plot_chi_sq()

                self.ref = self.view.worker(self.run_refinement_thread)
                self.ref.stop = False

                self.view.progress(self.ref, self.run_refinement_progress)
                self.view.finished(self.ref, self.run_refinement_complete)
                self.threadpool.start(self.ref)

    def filter_sigma(self):

        sigma_h = self.view.get_filter_h()
        sigma_k = self.view.get_filter_k()
        sigma_l = self.view.get_filter_l()

        sigma = [sigma_h, sigma_k, sigma_l]

        mask = self.mask

        v_inv, boxes = self.model.gaussian(mask, sigma)

        self.v_inv, self.boxes = v_inv, boxes

    def stop_refinement(self):

        if self.view.get_recalculation_3d_table_row_count():
            if (self.iteration > 0 or self.view.get_run() > 0):
                if self.allocated:
                    if (self.ref is not None):
                        self.ref.abort()

    def reset_refinement(self):

        if self.view.get_recalculation_3d_table_row_count():
            self.allocated = False

            self.stop_refinement()

            self.view.set_progress(0)
            self.view.set_run(0)
            self.view.set_runs(1)
            self.iteration = 0

            self.view.clear_plot_ref_canvas()
            self.view.clear_plot_chi_sq_canvas()
            self.view.clear_plot_1d_canvas()
            self.view.clear_plot_3d_canvas()
            self.view.clear_calc_1d_canvas()
            self.view.clear_calc_3d_canvas()

            self.view.clear_pairs_1d_table()
            self.view.clear_pairs_3d_table()
            self.view.clear_atom_site_recalculation_1d_table()
            self.view.clear_atom_site_recalculation_3d_table()

    def refinement_cycle(self):

        N = self.n_uvw*self.n_atm*1

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

        constant = self.constant

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

            fixed_mag = self.fixed_mag

            heisenberg = self.heisenberg

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
                fixed_mag, heisenberg, nh, nk, nl, nu, nv, nw, n_atm, n, N)

        elif self.occupational:

            A_r, occupancy = self.A_r, self.occupancy

            A_k, A_k_orig, A_k_cand = self.A_k, self.A_k_orig, self.A_k_cand

            F, F_orig, F_cand = self.F, self.F_orig, self.F_cand

            prod = self.prod
            prod_orig, prod_cand = self.prod_orig, self.prod_cand

            factors  = self.factors

            fixed_occ = self.fixed_occ

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
                chi_sq, energy, temperature, scale, constant, fixed_occ,
                nh, nk, nl, nu, nv, nw, n_atm, n, N)

        elif self.displacive:

            Ux, Uy, Uz = self.Ux, self.Uy, self.Uz

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

            fixed_dis = self.fixed_dis

            isotropic = self.isotropic

            Lxx, Lyy, Lzz = self.Lxx, self.Lyy, self.Lzz
            Lyz, Lxz, Lxy = self.Lyz, self.Lxz, self.Lxy

            p = self.p

            self.model.displacive_refinement(
                Ux, Uy, Uz,
                U_r, U_r_orig, U_r_cand,
                U_k, U_k_orig, U_k_cand,
                V_k, V_k_nuc, V_k_orig,
                V_k_nuc_orig, V_k_cand, V_k_nuc_cand,
                F, F_nuc, F_orig, F_nuc_orig, F_cand, F_nuc_cand,
                prod, prod_nuc, prod_orig, prod_nuc_orig,
                prod_cand, prod_nuc_cand, space_factor, factors,
                coeffs, Q_k, Lxx, Lyy, Lzz, Lyz, Lxz, Lxy,
                I_calc, I_expt, inv_sigma_sq, I_raw, I_flat, I_ref, v_inv,
                a_filt, b_filt, c_filt,
                d_filt, e_filt, f_filt,
                g_filt, h_filt, i_filt,
                bragg, even, boxes, i_dft, inverses, i_mask, i_unmask,
                acc_moves, acc_temps, rej_moves, rej_temps,
                chi_sq, energy, temperature, scale, constant,
                fixed_dis, isotropic, p, nh, nk, nl, nu, nv, nw, n_atm, n, N)

        self.I_obs = I_flat.reshape(nh,nk,nl)
        self.I_obs[self.mask] = np.nan

        self.refinement_m = self.model.mask_array(self.I_obs*scale[-1])

    def draw_plot_ref(self):

        if self.allocated:

            canvas = self.view.get_plot_ref_canvas()
            data = self.ref_arr_m

            B = self.B

            hkl = self.view.get_slice_hkl()

            Bp = self.model.matrix_transform(B)

            nh, min_h, max_h = self.nh, self.min_h, self.max_h
            nk, min_k, max_k = self.nk, self.min_k, self.max_k
            nl, min_l, max_l = self.nl, self.min_l, self.max_l

            slice_hkl = self.view.get_slice()

            i_hkl = (nh-1) // 2

            if (hkl == 'h ='):
                i_hkl = (nh-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_h, max_h, nh, slice_hkl)
                slice_hkl = self.model.slice_value(min_h, max_h, nh, i_hkl)
                self.view.set_slice(slice_hkl)
            elif (hkl == 'k ='):
                i_hkl = (nk-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_k, max_k, nk, slice_hkl)
                slice_hkl = self.model.slice_value(min_k, max_k, nk, i_hkl)
                self.view.set_slice(slice_hkl)
            elif (hkl == 'l ='):
                i_hkl = (nl-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_l, max_l, nl, slice_hkl)
                slice_hkl = self.model.slice_value(min_l, max_l, nl, i_hkl)
                self.view.set_slice(slice_hkl)

            self.hkl = hkl
            self.i_hkl = i_hkl

            norm = self.view.get_norm_ref()

            vmin = self.view.get_min_ref()
            vmax = self.view.get_max_ref()

            self.view.validate_min_ref()
            self.view.validate_max_ref()

            plot_ref = plots.HeatMap(canvas)
            plot_ref.clear_canvas()

            if (hkl == 'h ='):
                slice_data = data[i_hkl,:,:]
                limits = min_k, min_l, max_k, max_l
                matrix = Bp[1:3,1:3]
                labels = r'$h={}$'.format(slice_hkl), r'$(0k0)$', r'$(00l)$'
            elif (hkl == 'k ='):
                slice_data = data[:,i_hkl,:]
                limits = min_h, min_l, max_h, max_l
                matrix = Bp[0::2,0::2]
                labels = r'$k={}$'.format(slice_hkl), r'$(h00)$', r'$(00l)$'
            else:
                slice_data = data[:,:,i_hkl]
                limits = min_h, min_k, max_h, max_k
                matrix = Bp[0:2,0:2]
                labels = r'$l={}$'.format(slice_hkl), r'$(0k0)$', r'$(0k0)$'

            plot_ref.plot_data(slice_data, *limits)
            plot_ref.transform_axes(matrix)
            plot_ref.set_labels(*labels)

            plot_ref.set_normalization(vmin, vmax, norm=norm)
            plot_ref.create_colorbar()

            plot_ref.tight_layout()
            plot_ref.draw_canvas()

            self.plot_ref = plot_ref

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

    def fast_redraw_plot_ref(self):

        plot_ref = self.plot_ref
        plot_type = self.view.get_plot_ref()

        if (plot_type == 'Calculated'):
            self.ref_arr_m = self.refinement_m

            self.view.set_min_ref(self.ref_arr_m.min())
            self.view.set_max_ref(self.ref_arr_m.max())

            vmin = self.view.get_min_ref()
            vmax = self.view.get_max_ref()

            self.view.validate_min_ref()
            self.view.validate_max_ref()

            data, hkl, i_hkl = self.ref_arr_m, self.hkl, self.i_hkl

            if (hkl == 'h ='):
                slice_data = data[i_hkl,:,:]
            elif (hkl == 'k ='):
                slice_data = data[:,i_hkl,:]
            else:
                slice_data = data[:,:,i_hkl]

            plot_ref.update_data(slice_data, vmin, vmax)
            plot_ref.draw_canvas()

    def draw_plot_chi_sq(self):

        if self.allocated:

            canvas = self.view.get_plot_chi_sq_canvas()

            plot_l = self.view.get_plot_top_chi_sq()
            plot_r = self.view.get_plot_bottom_chi_sq()

            self.plot_l, self.plot_r = plot_l, plot_r

            acc_moves, rej_moves = self.acc_moves, self.rej_moves
            temperature, energy = self.temperature, self.energy
            chi_sq, scale = self.chi_sq, self.scale

            chi_sq_plot = Line(canvas)
            chi_sq_plot.clear_canvas()

            xlabel = 'Moves'
            norm = twin_norm = 'logarithmic'
            twin_ylabel = ''
            twin = True

            if (plot_l == 'Accepted' and plot_r == 'Rejected'):
                ylabel = r'$\chi^2$'
                twin = False
                yl, yr = acc_moves.copy(), rej_moves.copy()
            elif (plot_l == 'Rejected' and plot_r == 'Accepted'):
                ylabel = r'$\chi^2$'
                twin = False
                yl, yr = rej_moves.copy(), acc_moves.copy()

            if (plot_l == 'Accepted' and not plot_r == 'Rejected'):
                ylabel = r'$\chi^2$'
                yl = acc_moves.copy()
            elif (plot_l == 'Rejected' and not plot_r == 'Accepted'):
                ylabel = r'$\chi^2$'
                yl = rej_moves.copy()
            elif (plot_l == 'Temperature'):
                ylabel = r'$T$'
                yl = temperature.copy()
            elif (plot_l == 'Energy'):
                ylabel = r'$\Delta\chi^2$'
                yl = energy.copy()
                norm = 'linear'
            elif (plot_l == 'Chi-squared'):
                ylabel = r'$\chi^2$'
                yl = chi_sq.copy()
            elif (plot_l == 'Scale factor'):
                ylabel = r'$s$'
                yl = scale.copy()
                norm = 'linear'

            if (plot_r == 'Accepted' and not plot_l == 'Rejected'):
                twin_ylabel = r'$\chi^2$'
                yr = acc_moves.copy()
            elif (plot_r == 'Rejected' and not plot_l == 'Accepted'):
                twin_ylabel = r'$\chi^2$'
                yr = rej_moves.copy()
            elif (plot_r == 'Temperature'):
                twin_ylabel = r'$T$'
                yr = temperature.copy()
            elif (plot_r == 'Energy'):
                twin_ylabel = r'$\Delta\chi^2$'
                yr = energy.copy()
                twin_norm = 'linear'
            elif (plot_r == 'Chi-squared'):
                twin_ylabel = r'$\chi^2$'
                yr = chi_sq.copy()
            elif (plot_r == 'Scale factor'):
                twin_ylabel = r'$s$'
                yr = scale.copy()
                twin_norm = 'linear'

            xl, xr = np.arange(len(yl)), np.arange(len(yr))

            chi_sq_plot.plot_data(xl, yl, marker='-', label=plot_l)
            chi_sq_plot.plot_data(xr, yr, marker='-', label=plot_r, twin=twin)

            chi_sq_plot.set_labels(xlabel=xlabel,
                                   ylabel=ylabel,
                                   twin_ylabel=twin_ylabel)

            chi_sq_plot.set_normalization(norm)
            chi_sq_plot.set_normalization(twin_norm, twin=twin)

            chi_sq_plot.tight_layout()

            chi_sq_plot.use_scientific()
            chi_sq_plot.use_scientific(twin=twin)

            chi_sq_plot.show_legend()
            chi_sq_plot.draw_canvas()

            self.chi_sq_plot = chi_sq_plot

    def fast_redraw_plot_chi_sq(self):

        chi_sq_plot = self.chi_sq_plot

        plot_l, plot_r = self.plot_l, self.plot_r

        acc_moves, rej_moves = self.acc_moves, self.rej_moves
        temperature, energy = self.temperature, self.energy
        chi_sq, scale = self.chi_sq, self.scale

        twin = True

        if (plot_l == 'Accepted' and plot_r == 'Rejected'):
            twin = False
        elif (plot_l == 'Rejected' and plot_r == 'Accepted'):
            twin = False

        if (plot_l == 'Accepted'):
            yl = acc_moves.copy()
        elif (plot_l == 'Rejected'):
            yl = rej_moves
        elif (plot_l == 'Temperature'):
            yl = temperature.copy()
        elif (plot_l == 'Energy'):
            yl = energy.copy()
        elif (plot_l == 'Chi-squared'):
            yl = chi_sq.copy()
        elif (plot_l == 'Scale factor'):
            yl = scale.copy()

        if (plot_r == 'Accepted'):
            yr = acc_moves.copy()
        elif (plot_r == 'Rejected'):
            yr = rej_moves.copy()
        elif (plot_r == 'Temperature'):
            yr = temperature.copy()
        elif (plot_r == 'Energy'):
            yr = energy.copy()
        elif (plot_r == 'Chi-squared'):
            yr = chi_sq.copy()
        elif (plot_r == 'Scale factor'):
            yr = scale.copy()

        xl, xr = np.arange(len(yl)), np.arange(len(yr))

        chi_sq_plot.update_data(xl, yl, i=0)
        chi_sq_plot.update_data(xr, yr, i=1)

        chi_sq_plot.reset_view()

        if twin:
            chi_sq_plot.reset_view(twin)

        chi_sq_plot.draw_canvas()

    # ---

    def recreate_table_1d(self):

        unique_pairs = np.unique(self.atm_pair1d)

        rows = self.view.get_pairs_1d_table_row_count()

        pairs = []
        for i in range(rows):
            left, right, active = self.view.get_pairs_1d_table_row(i)
            if (left == 'self-correlation'):
                pairs.append('0')
            else:
                pairs.append('{}_{}'.format(left,right))

        n_pairs = unique_pairs.size
        self.view.clear_pairs_1d_table()
        self.view.create_pairs_1d_table(n_pairs)

        for i in range(n_pairs):
            uni = unique_pairs[i]
            if (uni == '0'):
                uni = 'self-correlation'
                self.view.set_pairs_1d_table_row([uni, ' ', True], i)
            else:
                left, right = uni.split('_')
                self.view.set_pairs_1d_table_row([left, right, True], i)

        visible = False if self.view.get_average_1d() else True
        self.view.enable_pairs_1d(visible)
        self.view.check_clicked_pairs_1d(self.plot_1d)
        self.view.format_pairs_1d_table()

    def calculate_1d_complete(self):

        self.view.enable_calculate_1d(True)
        self.recreate_table_1d()
        self.plot_1d()

    def calculate_correlations_1d(self):

        if (self.view.get_progress() > 0 and self.allocated):

            self.view.enable_calculate_1d(False)

            self.calc_1d = self.view.worker(self.calculate_1d_thread)
            self.view.finished(self.calc_1d, self.calculate_1d_complete)
            self.threadpool.start(self.calc_1d)

    def calculate_1d_thread(self, callback):

        disorder = self.view.get_correlations_1d()

        fract = self.view.get_fract_1d()
        tol = self.view.get_tol_1d()

        runs = self.view.get_runs_1d()

        nu, nv, nw = self.nu, self.nv, self.nw

        rx, ry, rz, atms = self.rx, self.ry, self.rz, self.atms

        A = self.A

        corr1d_arrs, coll1d_arrs = [], []

        for run in range(runs):

            if (disorder == 'Moment'):

                Sx, Sy, Sz = self.model.load_magnetic(self.fname, run)

                args = [Sx, Sy, Sz, rx, ry, rz,
                        atms, nu, nv, nw, A, fract, tol]

                data = self.model.vector_correlations_1d(*args)

                corr1d, coll1d, d, atm_pair1d = data

                corr1d_arrs.append(corr1d)
                coll1d_arrs.append(coll1d)

            elif (disorder == 'Occupancy'):

                A_r = self.model.load_occupational(self.fname, run)

                args = [A_r, rx, ry, rz, atms, nu, nv, nw, A, fract, tol]

                data = self.model.scalar_correlations_1d(*args)
               
                corr1d, d, atm_pair1d = data

                corr1d_arrs.append(corr1d)

            elif (disorder == 'Displacement'):

                Ux, Uy, Uz = self.model.load_displacive(self.fname, run)

                args = [Ux, Uy, Uz, rx, ry, rz, 
                        atms, nu, nv, nw, A, fract, tol]

                data = self.model.vector_correlations_1d(*args)

                corr1d, coll1d, d, atm_pair1d = data

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
                          sigma_sq_coll1d, d, atm_pair1d, tol]

                ave1d = self.model.vector_average_1d(*data1d)

                corr1d, coll1d, sigma_sq_corr1d, \
                sigma_sq_coll1d, d, atm_pair1d = ave1d

            else:

                data1d = [corr1d, sigma_sq_corr1d, d, atm_pair1d, tol]

                ave1d = self.model.scalar_average_1d(*data1d)

                corr1d, sigma_sq_corr1d, d, atm_pair1d = ave1d

        self.d, self.atm_pair1d = d, atm_pair1d

        self.corr1d, self.sigma_sq_corr1d = corr1d, sigma_sq_corr1d

        if (disorder != 'Occupancy'):

            self.coll1d, self.sigma_sq_coll1d = coll1d, sigma_sq_coll1d

    def plot_1d(self):

        if (self.view.get_pairs_1d_table_row_count() > 0):

            disorder = self.view.get_correlations_1d()
            correlation = self.view.get_plot_1d()

            norm = self.view.get_norm_1d()
            if (norm == 'Logarithmic'):
                norm = 'SymLog'

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
            for i in range(self.view.get_pairs_1d_table_row_count()):
                left, right, active = self.view.get_pairs_1d_table_row(i)
                if active:
                    atoms.append(left)
                    pairs.append(right)

            correlations_1d = Line(canvas)
            correlations_1d.clear_canvas()

            if average:
                correlations_1d.plot_data(d, data, marker='o')
            else:
                for atom, pair in zip(atoms, pairs):
                    if (atom == r'self-correlation'):
                        mask = atm_pair1d == '0'
                        label = atom
                    else:
                        mask = atm_pair1d == '{}-{}'.format(atom,pair)

                        atm_l = re.sub(r'[^a-zA-Z]', '', atom)
                        atm_r = re.sub(r'[^a-zA-Z]', '', pair)

                        pre_l, post_l = atom.split(atm_l)
                        pre_r, post_r = pair.split(atm_r)

                        left =  [pre_l, atm_l, post_l]
                        right = [pre_r, atm_r, post_r]

                        label = r'$^{{{}}}${}$^{{{}}}$'.format(*left)+'-'+\
                                r'$^{{{}}}${}$^{{{}}}$'.format(*right)

                    correlations_1d.plot_data(d[mask], data[mask], 
                                              yerr=error[mask],
                                              marker='o', label=label)
                correlations_1d.show_legend()

            correlations_1d.set_normalization(norm)

            xlabel = r'$r$ [$\AA$]'

            if (correlation == 'Correlation'):
                correlations_1d.draw_horizontal()
                correlations_1d.set_limits(-1,1)
                if (disorder == 'Moment'):
                    ylabel = r'$\langle\mathbf{S}(0)\cdot\mathbf{S}(r)\rangle$'
                elif (disorder == 'Occupancy'):
                    ylabel = r'$\langle\sigma(0)\cdot\sigma(r)\rangle$'
                else:
                    ylabel = r'$\langle\hat{\mathbf{u}}(0)\cdot'\
                                     r'\hat{\mathbf{u}}(r)\rangle$'
            else:
                correlations_1d.set_limits(0,1)
                if (disorder == 'Moment'):
                    ylabel = r'$\langle|\mathbf{S}(0)\cdot'\
                                      r'\mathbf{S}(r)|^2\rangle$'
                else:
                    ylabel = r'$\langle|\hat{\mathbf{u}}(0)\cdot'\
                                      r'\hat{\mathbf{u}}(r)|^2\rangle$'

            correlations_1d.set_labels(xlabel=xlabel, ylabel=ylabel)
            correlations_1d.draw_canvas()

            self.correlations_1d = correlations_1d

    def recreate_table_3d(self):

        unique_pairs = np.unique(self.atm_pair3d)

        rows = self.view.get_pairs_3d_table_row_count()

        pairs = []
        for i in range(rows):
            left, right, active = self.view.get_pairs_3d_table_row(i)
            if (left == 'self-correlation'):
                pairs.append('0')
            else:
                pairs.append('{}_{}'.format(left,right))

        n_pairs = unique_pairs.size
        self.view.clear_pairs_3d_table()
        self.view.create_pairs_3d_table(n_pairs)

        for i in range(n_pairs):
            uni = unique_pairs[i]
            if (uni == '0'):
                uni = 'self-correlation'
                self.view.set_pairs_3d_table_row([uni, ' ', True], i)
            else:
                left, right = uni.split('_')
                self.view.set_pairs_3d_table_row([left, right, True], i)

        visible = False if self.view.get_average_3d() else True
        self.view.enable_pairs_3d(visible)
        self.view.check_clicked_pairs_3d(self.plot_3d)
        self.view.format_pairs_3d_table()

    def calculate_3d_process_output(self, data):

        self.view.set_symmetrize(data)

    def calculate_3d_complete(self):

        self.view.enable_calculate_3d(True)
        self.recreate_table_3d()
        self.plot_3d()

    def calculate_correlations_3d(self):

        if (self.view.get_progress() > 0 and self.allocated):

            self.view.enable_calculate_3d(False)

            self.calc_3d = self.view.worker(self.calculate_3d_thread)
            self.view.finished(self.calc_3d, self.calculate_3d_complete)
            self.view.result(self.calc_3d, self.calculate_3d_process_output)
            self.threadpool.start(self.calc_3d)

    def calculate_3d_thread(self, callback):

        disorder = self.view.get_correlations_3d()

        fract = self.view.get_fract_3d()
        tol = self.view.get_tol_3d()

        runs = self.view.get_runs_3d()

        nu, nv, nw = self.nu, self.nv, self.nw

        rx, ry, rz, atms = self.rx, self.ry, self.rz, self.atms

        A = self.A

        corr3d_arrs, coll3d_arrs = [], []

        for run in range(runs):

            if (disorder == 'Moment'):

                Sx, Sy, Sz = self.model.load_magnetic(self.fname, run)

                args = [Sx, Sy, Sz, rx, ry, rz,
                        atms, nu, nv, nw, A, fract, tol]

                data = self.model.vector_correlations_3d(*args)

                corr3d, coll3d, dx, dy, dz, atm_pair3d = data

                corr3d_arrs.append(corr3d)
                coll3d_arrs.append(coll3d)

            elif (disorder == 'Occupancy'):

                A_r = self.model.load_occupational(self.fname, run)

                args = [A_r, rx, ry, rz, atms, nu, nv, nw, A, fract, tol]

                data = self.model.scalar_correlations_3d(*args)

                corr3d, dx, dy, dz, atm_pair3d = data

                corr3d_arrs.append(corr3d)

            elif (disorder == 'Displacement'):

                Ux, Uy, Uz = self.model.load_displacive(self.fname, run)

                args = [Ux, Uy, Uz, rx, ry, rz,
                        atms, nu, nv, nw, A, fract, tol]

                data = self.model.vector_correlations_3d(*args)

                corr3d, coll3d, dx, dy, dz, atm_pair3d = data

                corr3d_arrs.append(corr3d)
                coll3d_arrs.append(coll3d)

        stats_corr3d = self.model.correlation_statistics(corr3d_arrs)
        corr3d, sigma_sq_corr3d = stats_corr3d

        if (disorder != 'Occupancy'):

            stats_coll3d = self.model.correlation_statistics(coll3d_arrs)
            coll3d, sigma_sq_coll3d = stats_coll3d

        symm = self.view.get_symmetrize()

        if (symm == 'cif'):

            fname_cif = self.fname_cif

            folder = os.path.dirname(fname_cif)
            filename = os.path.basename(fname_cif)

            symm = self.model.find_laue(folder, filename)

        if (symm != 'None'):

            if (disorder != 'Occupancy'):

                data3d = [corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d,
                          dx, dy, dz, atm_pair3d, A, symm, tol]

                symm3d = self.model.vector_symmetrize_3d(*data3d)

                corr3d, coll3d, sigma_sq_corr3d, sigma_sq_coll3d, \
                dx, dy, dz, atm_pair3d = symm3d

            else:

                data3d = [corr3d, sigma_sq_corr3d, dx, dy, dz,
                          atm_pair3d, A, symm, tol]

                symm3d = self.model.scalar_symmetrizes_3d(*data3d)

                corr3d, sigma_sq_corr3d, dx, dy, dz, atm_pair3d = symm3d

        if self.view.average_3d_checked():

            if (disorder != 'Occupancy'):

                data3d = [corr3d, coll3d, sigma_sq_corr3d,
                          sigma_sq_coll3d, dx, dy, dz, atm_pair3d, tol]

                ave3d = self.model.vector_average_3d(*data3d)

                corr3d, coll3d, sigma_sq_corr3d, \
                sigma_sq_coll3d, dx, dy, dz, atm_pair3d = ave3d

            else:

                data3d = [corr3d, sigma_sq_corr3d, dx, dy, dz, atm_pair3d, tol]

                ave3d = self.model.scalar_average_3d(*data3d)

                corr3d, sigma_sq_corr3d, dx, dy, dz, atm_pair3d = ave3d

        self.dx, self.dy, self.dz, self.atm_pair3d = dx, dy, dz, atm_pair3d

        self.corr3d, self.sigma_sq_corr3d = corr3d, sigma_sq_corr3d

        if (disorder != 'Occupancy'):

            self.coll3d, self.sigma_sq_coll3d = coll3d, sigma_sq_coll3d

        return symm

    def plot_3d(self):

        if (self.view.get_pairs_3d_table_row_count() > 0):

            disorder = self.view.get_correlations_3d()
            correlation = self.view.get_plot_3d()

            norm = self.view.get_norm_3d()
            if (norm == 'Logarithmic'):
                norm = 'SymLog'

            tol = self.view.get_tol_3d()

            average = self.view.average_3d_checked()

            h, k, l = self.view.get_h(), self.view.get_k(), self.view.get_l()

            if (h**2+k**2+l**2 == 0): l = 1

            d = self.view.get_d()

            A, B = self.A, self.B

            dx, dy, dz, atm_pair3d = self.dx, self.dy, self.dz, self.atm_pair3d

            if (correlation == 'Correlation'):
                data = self.corr3d
                error = self.sigma_sq_corr3d
            else:
                data = self.coll3d
                error = self.sigma_sq_coll3d

            mask_params = dx, dy, dz, h, k, l, d, A, B, tol

            proj_params = self.model.mask_plane(*mask_params)

            cor_aspect, projx, projy, Dx, Dy, plane = proj_params

            canvas = self.view.get_plot_3d_canvas()

            atoms, pairs = [], []
            for i in range(self.view.get_pairs_3d_table_row_count()):
                left, right, active = self.view.get_pairs_3d_table_row(i)
                if active:
                    atoms.append(left)
                    pairs.append(right)

            data, error = data[plane], error[plane]
            atm_pair3d = atm_pair3d[plane]

            if not average:
                mask = np.full(data.size, False)
                for atom, pair in zip(atoms, pairs):
                    if (atom == 'self-correlation'):
                        mask[atm_pair3d == '0'] = True
                    else:
                        mask[atm_pair3d == atom+'_'+pair] = True

                Dx, Dy, atm_pair3d = Dx[mask], Dy[mask], atm_pair3d[mask]
                data, error = data[mask], error[mask]

            scale = np.gcd.reduce([h, k, l])

            if (scale != 0):
                h, k, l = np.array([h, k, l]) // scale

            H = '{}'.format(h) if (h >= 0) else r'\bar{{{}}}'.format(h)
            K = '{}'.format(k) if (k >= 0) else r'\bar{{{}}}'.format(k)
            L = '{}'.format(k) if (l >= 0) else r'\bar{{{}}}'.format(l)

            d = np.round(d, int(-np.log10(tol)))

            plane = r'$({}{}{})\cdot[uvw]={}$'.format(H,K,L,d)

            uvw = np.array(['u','v','w'])

            varx = np.repeat(uvw[np.argwhere(np.isclose(projx,1))[0][0]],3)
            vary = np.repeat(uvw[np.argwhere(np.isclose(projy,1))[0][0]],3)

            projx = np.round(projx,4)
            projy = np.round(projy,4)

            coeffx = projx.astype(str)
            coeffy = projy.astype(str)

            varx[np.isclose(projx,0)] = ''
            vary[np.isclose(projy,0)] = ''

            coeffx[np.isclose(projx,0)] = '0'
            coeffy[np.isclose(projy,0)] = '0'

            coeffx[np.isclose(projx,1)] = ''
            coeffy[np.isclose(projy,1)] = ''

            coeffx[np.isclose(projx,1)] = '-'
            coeffy[np.isclose(projy,1)] = '-'

            axisx = [coeff+var for coeff, var in zip(coeffx, varx)]
            axisy = [coeff+var for coeff, var in zip(coeffy, vary)]

            xlabel = r'$[{},{},{}]$'.format(*axisx)
            ylabel = r'$[{},{},{}]$'.format(*axisy)

            if (correlation == 'Correlation'):
                colormap, vmin = 'diverging', -1
                if (disorder == 'Moment'):
                    label = r'$\langle\mathbf{S}(\mathbf{0})'\
                               r'\cdot\mathbf{S}(\mathbf{r})\rangle$'
                elif (disorder == 'Occupancy'):
                    label = r'$\langle\sigma(\mathbf{0})'\
                               r'\cdot\sigma(\mathbf{r})\rangle$'
                else:
                    label = r'$\langle\hat{\mathbf{u}}(\mathbf{0})'\
                               r'\cdot\hat{\mathbf{u}}(\mathbf{r})\rangle$'
            else:
                colormap, vmin = 'binary', 0
                if (disorder == 'Moment'):
                    label = r'$\langle|\mathbf{S}(\mathbf{0})'\
                                r'\cdot\mathbf{S}(\mathbf{r})|^2\rangle$'
                else:
                    label = r'$\langle|\hat{\mathbf{u}}(\mathbf{0})'\
                                r'\cdot\hat{\mathbf{u}}(\mathbf{r})|^2\rangle$'

            correlations_3d = plots.Scatter(canvas)
            correlations_3d.clear_canvas()

            correlations_3d.plot_data(Dx, Dy, data)

            correlations_3d.update_colormap(colormap)
            correlations_3d.create_colorbar(norm=norm)

            correlations_3d.set_normalization(vmin, 1, norm=norm)
            correlations_3d.set_aspect(cor_aspect)

            if (norm == 'SymLog'): correlations_3d.reformat_colorbar()

            correlations_3d.set_colorbar_label(label)
            correlations_3d.set_labels(xlabel=xlabel, ylabel=ylabel)
            correlations_3d.draw_canvas()

            self.view.set_h(h)
            self.view.set_k(k)
            self.view.set_l(l)
            self.view.set_d(d)

            self.correlations_3d = correlations_3d

    # ---

    def disorder_check_mag_recalc_1d(self):

        if self.view.get_disorder_mag_recalc_1d():
            self.view.set_disorder_mag_recalc_1d(True)
            self.view.set_disorder_occ_recalc_1d(False)
            self.view.set_disorder_dis_recalc_1d(False)
        else:
            self.view.set_disorder_mag_recalc_1d(False)
            self.view.set_disorder_occ_recalc_1d(True)
            self.view.set_disorder_dis_recalc_1d(False)

    def disorder_check_occ_recalc_1d(self):

        if self.view.get_disorder_occ_recalc_1d():
            self.view.set_disorder_mag_recalc_1d(False)
            self.view.set_disorder_occ_recalc_1d(True)
            self.view.set_disorder_dis_recalc_1d(False)
        else:
            self.view.set_disorder_mag_recalc_1d(False)
            self.view.set_disorder_occ_recalc_1d(False)
            self.view.set_disorder_dis_recalc_1d(True)

    def disorder_check_dis_recalc_1d(self):

        if self.view.get_disorder_dis_recalc_1d():
            self.view.set_disorder_mag_recalc_1d(False)
            self.view.set_disorder_occ_recalc_1d(False)
            self.view.set_disorder_dis_recalc_1d(True)
        else:
            self.view.set_disorder_mag_recalc_1d(False)
            self.view.set_disorder_occ_recalc_1d(True)
            self.view.set_disorder_dis_recalc_1d(False)

    def recalculate_intensity_1d_thread(self, callback):

        if self.view.get_recalculation_1d_table_row_count():

            if (self.allocated == False): self.preprocess_supercell()

            # batch = self.view.batch_checked_calc_1d()

            runs = self.view.get_runs_calc_1d()

            # ---

            dQ, nQ, min_Q, max_Q = self.view.get_recalculation_1d_binning()

            Q_range = [2*np.pi*min_Q, 2*np.pi*max_Q]

            active = self.view.get_active_atom_site_1d()

            site = self.site

            _, inv = np.unique(site, return_inverse=True)

            mask = active[inv]

            #ux, uy, uz = self.ux[mask], self.uy[mask], self.uz[mask]

            nuc, ion = self.nuc[mask], self.ion[mask]

            atm = nuc if (self.view.get_type_recalc() == 'Neutron') else ion

            nu, nv, nw = self.nu, self.nv, self.nw

            rx, ry, rz = self.rx, self.ry, self.rz

            #fname_cif = self.fname_cif

            #folder = os.path.dirname(fname_cif)
            #filename = os.path.basename(fname_cif)

            fname = self.fname
        
            A = self.A
            D = self.D

            occupancy = self.occupancy[mask]

            U11 = self.U11[mask]
            U22 = self.U22[mask]
            U33 = self.U33[mask]
            U23 = self.U23[mask]
            U13 = self.U13[mask]
            U12 = self.U12[mask]

            Uiso = self.Uiso[mask]

            g = self.g[mask]

            adp_type = self.view.get_adp_type_recalc_1d()

            if (adp_type != 'Anisotropic'):
                if (adp_type != 'Isotropic'):
                    Uiso *= 0

                ADPs = self.model.anisotropic_parameters(Uiso, D)

                U11, U22, U33, U23, U13, U12 = ADPs

            if self.view.get_disorder_dis_recalc_1d():

                p = self.view.get_order_calc_1d()

            # ---

            self.ds_total_intensity = np.zeros(nQ)
            self.ds_bragg_intensity = np.zeros(nQ)
            self.ds_diffuse_intensity = np.zeros(nQ)

            for run in range(runs):

                if self.view.get_disorder_mag_recalc_1d():

                    I_calc = self.model.magnetic_intensity_1d(
                                 fname, run, occupancy,
                                 U11, U22, U33, U23, U13, U12, rx, ry, rz, ion,
                                 Q_range, nQ, A, D, nu, nv, nw, g, mask)

                    self.ds_diffuse_intensity[:] += I_calc

                elif self.view.get_disorder_occ_recalc_1d():

                    I_calc = self.model.occupational_intensity_1d(
                                 fname, run, occupancy,
                                 U11, U22, U33, U23, U13, U12, rx, ry, rz, atm,
                                 Q_range, nQ, A, D, nu, nv, nw, mask)

                    self.ds_total_intensity[:] += I_calc

                elif self.view.get_disorder_dis_recalc_1d():

                    I_calc = self.model.displacive_intensity_1d(
                                 fname, run, occupancy, rx, ry, rz, atm,
                                 Q_range, nQ, A, D, nu, nv, nw, p, mask)

                    self.ds_total_intensity[:] += I_calc

                I_calc = self.model.structural_intensity_1d(
                             occupancy, U11, U22, U33, U23, U13, U12,
                             rx, ry, rz, atm, Q_range, nQ, A, D,
                             nu, nv, nw, mask)

                self.ds_bragg_intensity[:] += I_calc

            if (self.view.get_disorder_occ_recalc_1d() or \
                self.view.get_disorder_dis_recalc_1d()):
                self.ds_diffuse_intensity = self.ds_total_intensity\
                                          - self.ds_bragg_intensity
            elif self.view.get_disorder_mag_recalc_1d():
                self.ds_total_intensity = self.ds_bragg_intensity.copy()
            else:
                self.ds_total_intensity = self.ds_bragg_intensity.copy()

            self.ds_total_intensity /= runs
            self.ds_bragg_intensity /= runs
            self.ds_diffuse_intensity /= runs

            self.ds_total_intensity[self.ds_total_intensity < 0] = np.nan
            self.ds_bragg_intensity[self.ds_bragg_intensity < 0] = np.nan
            self.ds_diffuse_intensity[self.ds_diffuse_intensity < 0] = np.nan

    def recalculate_intensity_1d_complete(self):

        self.redraw_plot_calc_1d()

        self.view.enable_recalculation_1d(True)

    def recalculate_intensity_1d(self):

        if (self.view.get_recalculation_1d_table_row_count() and \
            self.allocated):

            self.view.enable_recalculation_1d(False)

            self.recalc_1d = self.view.worker(
                self.recalculate_intensity_1d_thread)
            self.view.finished(self.recalc_1d,
                               self.recalculate_intensity_1d_complete)
            self.threadpool.start(self.recalc_1d)

    def redraw_plot_calc_1d(self):

        if self.ds_total_intensity is not None:

            self.draw_plot_calc_1d()

    def draw_plot_calc_1d(self):

        if self.ds_total_intensity is not None:

            canvas = self.view.get_plot_calc_1d_canvas()

            data = [self.ds_total_intensity,
                    self.ds_bragg_intensity,
                    self.ds_diffuse_intensity]

            labels = ['Total', 'Bragg', 'Diffuse']

            dQ, nQ, min_Q, max_Q = self.view.get_recalculation_1d_binning()

            Q = 2*np.pi*np.linspace(min_Q, max_Q, nQ)

            profiles = self.view.get_profiles_calc_1d()

            if (profiles == 'Total only'):
                indices = [0]
            elif (profiles == 'Diffuse only'):
                indices = [2]
            elif (profiles == 'Bragg only'):
                indices = [1]
            elif (profiles == 'Total and Bragg'):
                indices = [0,1]
            elif (profiles == 'Diffuse and Bragg'):
                indices = [1,2]
            elif (profiles == 'Total and Diffuse'):
                indices = [0,2]
            else:
                indices = [0,1,2]

            norm = self.view.get_norm_calc_1d()

            marker = self.view.get_marker_calc_1d()

            if (marker == '-'):
                markers = ['-', '-', '-']
            if (marker == 'o'):
                markers = ['o', 'o', 'o']
            else:
                markers = ['-o', '-o', '-o']

            plot_calc_1d = Line(canvas)
            plot_calc_1d.clear_canvas()

            for i in indices:
                plot_calc_1d.plot_data(Q/(2*np.pi), data[i], \
                                       marker=markers[i], label=labels[i])

            plot_calc_1d.set_normalization(norm)
            plot_calc_1d.show_legend()

            plot_calc_1d.set_labels(xlabel=r'$Q/2\pi$ [$\AA$]',
                                    ylabel=r'$I(Q)$ [arb. unit]')

            plot_calc_1d.draw_canvas()

    def disorder_check_mag_recalc_3d(self):

        if self.view.get_disorder_mag_recalc_3d():
            self.view.set_disorder_mag_recalc_3d(True)
            self.view.set_disorder_occ_recalc_3d(False)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(False)
        else:
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(True)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(False)

    def disorder_check_occ_recalc_3d(self):

        if self.view.get_disorder_occ_recalc_3d():
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(True)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(False)
        else:
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(False)
            self.view.set_disorder_dis_recalc_3d(True)
            self.view.set_disorder_struct_recalc_3d(False)

    def disorder_check_dis_recalc_3d(self):

        if self.view.get_disorder_dis_recalc_3d():
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(False)
            self.view.set_disorder_dis_recalc_3d(True)
            self.view.set_disorder_struct_recalc_3d(False)
        else:
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(True)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(False)

    def disorder_check_struct_recalc_3d(self):

        if self.view.get_disorder_struct_recalc_3d():
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(False)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(True)
        else:
            self.view.set_disorder_mag_recalc_3d(False)
            self.view.set_disorder_occ_recalc_3d(True)
            self.view.set_disorder_dis_recalc_3d(False)
            self.view.set_disorder_struct_recalc_3d(False)

    def recalculate_intensity_3d_thread(self, callback):

        if self.view.get_recalculation_3d_table_row_count():

            if (self.allocated == False): self.preprocess_supercell()

            # batch = self.view.batch_checked_calc_3d()

            runs = self.view.get_runs_calc_3d()

            # ---

            dh, nh, min_h, max_h = self.view.get_recalculation_3d_binning_h()
            dk, nk, min_k, max_k = self.view.get_recalculation_3d_binning_k()
            dl, nl, min_l, max_l = self.view.get_recalculation_3d_binning_l()

            h_range = [min_h, max_h]
            k_range = [min_k, max_k]
            l_range = [min_l, max_l]

            active = self.view.get_active_atom_site_3d()

            site = self.site

            _, inv = np.unique(site, return_inverse=True)

            mask = active[inv]

            ux, uy, uz = self.ux[mask], self.uy[mask], self.uz[mask]

            nuc, ion = self.nuc[mask], self.ion[mask]

            atm = nuc if (self.view.get_type_recalc() == 'Neutron') else ion

            nu, nv, nw = self.nu, self.nv, self.nw

            fname_cif = self.fname_cif

            folder = os.path.dirname(fname_cif)
            filename = os.path.basename(fname_cif)

            fname = self.fname

            B, R, D, = self.B, self.R, self.D

            occupancy = self.occupancy[mask]

            U11 = self.U11[mask]
            U22 = self.U22[mask]
            U33 = self.U33[mask]
            U23 = self.U23[mask]
            U13 = self.U13[mask]
            U12 = self.U12[mask]

            Uiso = self.Uiso[mask]

            g = self.g[mask]

            adp_type = self.view.get_adp_type_recalc_3d()

            if (adp_type != 'Anisotropic'):
                if (adp_type != 'Isotropic'):
                    Uiso *= 0

                ADPs = self.model.anisotropic_parameters(Uiso, D)

                U11, U22, U33, U23, U13, U12 = ADPs

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
                
            self.T = T

            laue = self.view.get_laue()

            if (laue == 'cif'):

                laue = self.model.find_laue(folder, filename)

            self.intensity = np.zeros((nh,nk,nl))

            indices, inverses, operators, \
            Nu, Nv, Nw, \
            symop = self.model.reduced_reciprocal_space_symmetry(
                        h_range, k_range, l_range, nh, nk, nl,
                        nu, nv, nw, T, laue)

            if self.view.get_disorder_dis_recalc_3d() or \
               self.view.get_disorder_struct_recalc_3d():

                p = self.view.get_order_calc_3d()

                lat = self.view.get_lattice()

                cent = self.view.get_centering_calc_3d()

                if (lat == 'Rhombohedral'):
                    if (cent == 'R'): cent = 'P'

                coeffs, even, cntr = self.model.displacive_parameters(p, cent)

            for run in range(runs):

                if self.view.get_disorder_mag_recalc_3d():

                    I_calc = self.model.magnetic_intensity_3d(
                                 fname, run, occupancy,
                                 U11, U22, U33, U23, U13, U12, ux, uy, uz, ion,
                                 h_range, k_range, l_range, indices, symop,
                                 T, B, R, D, twins, variants, nh, nk, nl,
                                 nu, nv, nw, Nu, Nv, Nw, g, mask)

                    self.intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)

                elif self.view.get_disorder_occ_recalc_3d():

                    I_calc = self.model.occupational_intensity_3d(
                                 fname, run, occupancy,
                                 U11, U22, U33, U23, U13, U12, ux, uy, uz, atm,
                                 h_range, k_range, l_range, indices, symop,
                                 T, B, R, D, twins, variants, nh, nk, nl,
                                 nu, nv, nw, Nu, Nv, Nw, mask)

                    self.intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)

                elif self.view.get_disorder_dis_recalc_3d():

                    I_calc = self.model.displacive_intensity_3d(
                                 fname, run, coeffs, occupancy, ux, uy, uz,
                                 atm, h_range, k_range, l_range, indices,
                                 symop, T, B, R, twins, variants, nh, nk, nl,
                                 nu, nv, nw, Nu, Nv, Nw, p, even, cntr, mask)

                    self.intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)

                elif self.view.get_disorder_struct_recalc_3d():

                    I_calc = self.model.structural_intensity_3d(
                                occupancy, U11, U22, U33, U23, U13, U12,
                                ux, uy, uz, atm,
                                h_range, k_range, l_range, indices, symop,
                                T, B, R, D, twins, variants, nh, nk, nl,
                                nu, nv, nw, Nu, Nv, Nw, cntr, mask)

                    self.intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)

                self.intensity /= runs*len(operators)

                self.recalculation_3d_blur()

            return laue

    def recalculation_3d_blur(self):

        sigma_h, sigma_k, sigma_l = self.view.get_recalculation_3d_filter()

        sigma = [sigma_h, sigma_k, sigma_l]

        intensity = self.intensity

        I_recalc = self.model.blurring(intensity, sigma)

        self.intensity = I_recalc

    def recalculate_intensity_3d_output(self, data):

        self.view.set_laue(data)

    def recalculate_intensity_3d_complete(self):

        self.redraw_plot_calc_3d()

        self.view.enable_recalculation_3d(True)

    def recalculate_intensity_3d(self):

        if (self.view.get_recalculation_3d_table_row_count() and \
            self.allocated):

            self.view.enable_recalculation_3d(False)

            self.recalc_3d = self.view.worker(
                self.recalculate_intensity_3d_thread)
            self.view.finished(self.recalc_3d,
                               self.recalculate_intensity_3d_complete)
            self.view.result(self.recalc_3d,
                             self.recalculate_intensity_3d_output)
            self.threadpool.start(self.recalc_3d)

    def redraw_plot_calc_3d(self):

        if self.intensity is not None:

            self.view.set_min_calc_3d(self.intensity.min())
            self.view.set_max_calc_3d(self.intensity.max())

            self.draw_plot_calc_3d()

    def draw_plot_calc_3d(self):

        if self.intensity is not None:

            canvas = self.view.get_plot_calc_3d_canvas()
            data = self.intensity

            B = self.B

            axes = self.view.get_axes()

            if (axes == '(h00), (0k0), (00l)'):
                T = np.eye(3)
            else:
                T = np.array([[1, -1,  0],
                              [1,  1,  0],
                              [0,  0,  1]])*1.

            hkl = self.view.get_slice_hkl_calc_3d()

            Bp = self.model.matrix_transform(B, T=T)

            dh, nh, min_h, max_h = self.view.get_recalculation_3d_binning_h()
            dk, nk, min_k, max_k = self.view.get_recalculation_3d_binning_k()
            dl, nl, min_l, max_l = self.view.get_recalculation_3d_binning_l()

            slice_hkl = self.view.get_slice_calc_3d()

            if (hkl == 'h ='):
                i_hkl = (nh-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_h, max_h, nh, slice_hkl)
                slice_hkl = self.model.slice_value(min_h, max_h, nh, i_hkl)
                self.view.set_slice_calc_3d(slice_hkl)
            elif (hkl == 'k ='):
                i_hkl = (nk-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_k, max_k, nk, slice_hkl)
                slice_hkl = self.model.slice_value(min_k, max_k, nk, i_hkl)
                self.view.set_slice_calc_3d(slice_hkl)
            elif (hkl == 'l ='):
                i_hkl = (nl-1) // 2
                if (slice_hkl is not None):
                    i_hkl = self.model.slice_index(min_l, max_l, nl, slice_hkl)
                slice_hkl = self.model.slice_value(min_l, max_l, nl, i_hkl)
                self.view.set_slice_calc_3d(slice_hkl)

            norm = self.view.get_norm_calc_3d()

            vmin = self.view.get_min_calc_3d()
            vmax = self.view.get_max_calc_3d()

            if (norm == 'Logarithmic'):
                data_range = data.max()-data.min()
                if (vmin <= 0):
                    data -= vmin
                    vmin -= vmin
                    data += 0.001*data_range
                    vmin += 0.001*data_range

            self.view.validate_min_calc_3d()
            self.view.validate_max_calc_3d()

            plot_calc_3d = plots.HeatMap(canvas)
            plot_calc_3d.clear_canvas()

            if (hkl == 'h ='):
                slice_data = data[i_hkl,:,:]
                limits = min_k, min_l, max_k, max_l
                matrix = Bp[1:3,1:3]
                labels = r'$h={}$'.format(slice_hkl), r'$(0k0)$', r'$(00l)$'
            elif (hkl == 'k ='):
                slice_data = data[:,i_hkl,:]
                limits = min_h, min_l, max_h, max_l
                matrix = Bp[0::2,0::2]
                labels = r'$k={}$'.format(slice_hkl), r'$(h00)$', r'$(00l)$'
            else:
                slice_data = data[:,:,i_hkl]
                limits = min_h, min_k, max_h, max_k
                matrix = Bp[0:2,0:2]
                labels = r'$l={}$'.format(slice_hkl), r'$(0k0)$', r'$(0k0)$'

            plot_calc_3d.plot_data(slice_data, *limits)
            plot_calc_3d.transform_axes(matrix)
            plot_calc_3d.set_labels(*labels)

            plot_calc_3d.set_normalization(vmin, vmax, norm=norm)
            plot_calc_3d.create_colorbar()

            plot_calc_3d.draw_canvas()