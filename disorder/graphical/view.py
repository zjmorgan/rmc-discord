#!/ur/bin/env/python3

from PyQt5 import QtWidgets, QtGui, QtCore, uic

import os
import sys

import numpy as np

from disorder.graphical.utilities import FractionalDelegate
from disorder.graphical.utilities import StandardDoubleDelegate
from disorder.graphical.utilities import PositiveDoubleDelegate
from disorder.graphical.utilities import SizeIntDelegate

from disorder.graphical.utilities import Worker
from disorder.graphical.utilities import save_gui, load_gui

_root = os.path.abspath(os.path.dirname(__file__))

sys.path.append(_root)

qtCreatorFile_MainWindow = os.path.join(_root, 'mainwindow.ui')
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile_MainWindow)

class View(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):

        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)   
        
        icon = os.path.join(_root, 'logo.png')
                
        self.setWindowIcon(QtGui.QIcon(icon))
        
        self.comboBox_type.addItem('Neutron')
        self.comboBox_type.addItem('X-ray')

        self.comboBox_parameters.addItem('Site parameters')
        self.comboBox_parameters.addItem('Structural parameters')
        self.comboBox_parameters.addItem('Magnetic parameters')
        
        self.lineEdit_a.setEnabled(False)
        self.lineEdit_b.setEnabled(False)
        self.lineEdit_c.setEnabled(False)
  
        self.lineEdit_alpha.setEnabled(False)
        self.lineEdit_beta.setEnabled(False)
        self.lineEdit_gamma.setEnabled(False)
        
        self.lineEdit_a.setValidator(QtGui.QDoubleValidator(0.0001, 1000, 8))
        self.lineEdit_b.setValidator(QtGui.QDoubleValidator(0.0001, 1000, 8))
        self.lineEdit_c.setValidator(QtGui.QDoubleValidator(0.0001, 1000, 8))
        
        self.lineEdit_alpha.setValidator(QtGui.QDoubleValidator(5, 175, 8))
        self.lineEdit_beta.setValidator(QtGui.QDoubleValidator(5, 175, 8))
        
        self.lineEdit_lat.setEnabled(False)
     
        self.lineEdit_n_atm.setEnabled(False)
        self.lineEdit_n.setEnabled(False)
        self.lineEdit_space_group.setEnabled(False)
        self.lineEdit_space_group_hm.setEnabled(False)
        
        self.lineEdit_nu.setValidator(QtGui.QIntValidator(1, 32))
        self.lineEdit_nv.setValidator(QtGui.QIntValidator(1, 32))
        self.lineEdit_nw.setValidator(QtGui.QIntValidator(1, 32))
                
        # ---
                
        self.comboBox_centering.addItem('P')
        self.comboBox_centering.addItem('I')
        self.comboBox_centering.addItem('F')
        self.comboBox_centering.addItem('A')
        self.comboBox_centering.addItem('B')
        self.comboBox_centering.addItem('C')
        self.comboBox_centering.addItem('R')
        
        self.comboBox_punch.addItem('Box')
        self.comboBox_punch.addItem('Ellipsoid')
        
        self.lineEdit_outlier.setValidator(QtGui.QDoubleValidator(0, 99999, 4))

        self.lineEdit_radius_h.setValidator(QtGui.QIntValidator(0, 100))
        self.lineEdit_radius_k.setValidator(QtGui.QIntValidator(0, 100))
        self.lineEdit_radius_l.setValidator(QtGui.QIntValidator(0, 100))
        
        self.comboBox_plot_exp.addItem('Intensity')
        self.comboBox_plot_exp.addItem('Error')

        self.comboBox_norm_exp.addItem('Linear')
        self.comboBox_norm_exp.addItem('Logarithmic')
        
        # ---

        self.lineEdit_runs.setValidator(QtGui.QIntValidator(1, 100))  
        self.lineEdit_cycles.setValidator(QtGui.QIntValidator(1, 10000))
       
        validator = QtGui.QDoubleValidator(0, 99999, 4)
        self.lineEdit_filter_ref_h.setValidator(validator)
        self.lineEdit_filter_ref_k.setValidator(validator)
        self.lineEdit_filter_ref_l.setValidator(validator)

        self.lineEdit_order.setValidator(QtGui.QIntValidator(0, 10))
        
        self.comboBox_centering_ref.addItem('P')
        self.comboBox_centering_ref.addItem('I')
        self.comboBox_centering_ref.addItem('F')
        self.comboBox_centering_ref.addItem('A')
        self.comboBox_centering_ref.addItem('B')
        self.comboBox_centering_ref.addItem('C')
        self.comboBox_centering_ref.addItem('R')
        
        self.comboBox_slice.addItem('h =')
        self.comboBox_slice.addItem('k =')
        self.comboBox_slice.addItem('l =')
        
        notation = QtGui.QDoubleValidator.ScientificNotation
        validator = QtGui.QDoubleValidator(0, 1e+10, 4, notation=notation)
        
        self.lineEdit_prefactor.setValidator(validator)
        self.lineEdit_tau.setValidator(validator)
        
        self.comboBox_plot_top_chi_sq.addItem('Accepted')
        self.comboBox_plot_top_chi_sq.addItem('Rejected')
        self.comboBox_plot_top_chi_sq.addItem('Temperature')
        self.comboBox_plot_top_chi_sq.addItem('Energy')
        self.comboBox_plot_top_chi_sq.addItem('Chi-squared')
        self.comboBox_plot_top_chi_sq.addItem('Scale factor')
        
        self.comboBox_plot_bottom_chi_sq.addItem('Accepted')
        self.comboBox_plot_bottom_chi_sq.addItem('Rejected')
        self.comboBox_plot_bottom_chi_sq.addItem('Temperature')
        self.comboBox_plot_bottom_chi_sq.addItem('Energy')
        self.comboBox_plot_bottom_chi_sq.addItem('Chi-squared')
        self.comboBox_plot_bottom_chi_sq.addItem('Scale factor')
        
        self.comboBox_plot_ref.addItem('Calculated')
        self.comboBox_plot_ref.addItem('Experimental')
        self.comboBox_plot_ref.addItem('Error')
        
        self.comboBox_norm_ref.addItem('Linear')
        self.comboBox_norm_ref.addItem('Logarithmic')

        self.lineEdit_chi_sq.setEnabled(False)
        
        disorder = self.tabWidget_disorder.tabBar()
        
        self.checkBox_mag = QtWidgets.QCheckBox()
        self.checkBox_mag.setObjectName('Magnetic')
        
        self.checkBox_occ = QtWidgets.QCheckBox()
        self.checkBox_occ.setObjectName('Occupational')
        
        self.checkBox_dis = QtWidgets.QCheckBox()
        self.checkBox_dis.setObjectName('Displacive')
        
        disorder.setTabButton(0, QtWidgets.QTabBar.LeftSide, self.checkBox_mag)
        disorder.setTabButton(1, QtWidgets.QTabBar.LeftSide, self.checkBox_occ)
        disorder.setTabButton(2, QtWidgets.QTabBar.LeftSide, self.checkBox_dis)
        
        # self.comboBox_moment_constraint.addItem('Heisenberg')
        # self.comboBox_moment_constraint.addItem('Ising')
        
        # self.comboBox_displacement_parameters.addItem('Isotropic')
        # self.comboBox_displacement_parameters.addItem('Anisotropic')

        # ---

        self.lineEdit_runs_corr_1d.setValidator(QtGui.QIntValidator(1, 100))  
        self.lineEdit_runs_corr_3d.setValidator(QtGui.QIntValidator(1, 100))
        
        self.lineEdit_fract_1d.setValidator(QtGui.QDoubleValidator(0, 1, 4))
        self.lineEdit_fract_3d.setValidator(QtGui.QDoubleValidator(0, 1, 4))
        
        notation = QtGui.QDoubleValidator.ScientificNotation
        validator = QtGui.QDoubleValidator(1e-6, 1e-1, 4, notation=notation)
        
        self.lineEdit_tol_1d.setValidator(validator)
        self.lineEdit_tol_3d.setValidator(validator)
        
        self.lineEdit_plane_h.setValidator(QtGui.QIntValidator(-99, 99))
        self.lineEdit_plane_k.setValidator(QtGui.QIntValidator(-99, 99))
        self.lineEdit_plane_l.setValidator(QtGui.QIntValidator(-99, 99))
        self.lineEdit_plane_d.setValidator(QtGui.QDoubleValidator(-99, 99, 4))

        vectors = ['Correlation', 'Collinearity']
        scalars = ['Correlation']
        
        self.comboBox_correlations_1d.addItem('Moment', vectors)
        self.comboBox_correlations_1d.addItem('Occupancy', scalars)
        self.comboBox_correlations_1d.addItem('Displacement', vectors)

        self.comboBox_correlations_3d.addItem('Moment', vectors)
        self.comboBox_correlations_3d.addItem('Occupancy', scalars)
        self.comboBox_correlations_3d.addItem('Displacement', vectors) 
        
        self.comboBox_plot_1d.addItem('Correlation')

        self.comboBox_norm_1d.addItem('Linear')
        self.comboBox_norm_1d.addItem('Logarithmic')

        self.comboBox_plot_3d.addItem('Correlation')

        self.comboBox_norm_3d.addItem('Linear')
        self.comboBox_norm_3d.addItem('Logarithmic')
        
        self.comboBox_laue_corr.addItem('None')        
        self.comboBox_laue_corr.addItem('-1')
        self.comboBox_laue_corr.addItem('2/m')
        self.comboBox_laue_corr.addItem('mmm')
        self.comboBox_laue_corr.addItem('4/m')
        self.comboBox_laue_corr.addItem('4/mmm')
        self.comboBox_laue_corr.addItem('-3')
        self.comboBox_laue_corr.addItem('-3m')
        self.comboBox_laue_corr.addItem('6/m')
        self.comboBox_laue_corr.addItem('6/mmm')
        self.comboBox_laue_corr.addItem('m-3')
        self.comboBox_laue_corr.addItem('m-3m')
        self.comboBox_laue_corr.addItem('cif')
        
        # ---
        
        self.lineEdit_type.setEnabled(False)
        
        self.lineEdit_runs_calc.setValidator(QtGui.QIntValidator(1, 100))  

        self.comboBox_norm_calc.addItem('Linear')
        self.comboBox_norm_calc.addItem('Logarithmic')
        
        self.comboBox_axes.addItem('(h00), (0k0), (00l)')
        self.comboBox_axes.addItem('(hh0), (-kk0), (00l)')
        
        self.comboBox_laue.addItem('None')        
        self.comboBox_laue.addItem('-1')
        self.comboBox_laue.addItem('2/m')
        self.comboBox_laue.addItem('mmm')
        self.comboBox_laue.addItem('4/m')
        self.comboBox_laue.addItem('4/mmm')
        self.comboBox_laue.addItem('-3')
        self.comboBox_laue.addItem('-3m')
        self.comboBox_laue.addItem('6/m')
        self.comboBox_laue.addItem('6/mmm')
        self.comboBox_laue.addItem('m-3')
        self.comboBox_laue.addItem('m-3m')
        self.comboBox_laue.addItem('cif')
        
        self.comboBox_centering_calc.addItem('P')
        self.comboBox_centering_calc.addItem('I')
        self.comboBox_centering_calc.addItem('F')
        self.comboBox_centering_calc.addItem('A')
        self.comboBox_centering_calc.addItem('B')
        self.comboBox_centering_calc.addItem('C')
        self.comboBox_centering_calc.addItem('R')
        
        self.lineEdit_order_calc.setValidator(QtGui.QIntValidator(0, 10))
        
        self.comboBox_slice_calc.addItem('h =')
        self.comboBox_slice_calc.addItem('k =')
        self.comboBox_slice_calc.addItem('l =')
                       
        self.clear_application()
        
        # ---
        
        self.unit_table = {'site': 0, 'atom': 1, 'isotope': 2, 'ion': 3,
                           'occupancy': 4, 'Uiso': 5,
                           'U11': 6, 'U22': 7, 'U33': 8,
                           'U23': 9, 'U13': 10, 'U12': 11,
                           'U1': 12, 'U2': 13, 'U3': 14,
                           'mu': 15, 'mu1': 16, 'mu2': 17, 'mu3': 18, 'g': 19,
                           'u': 20, 'v': 21, 'w': 22, 
                           'operator': 23, 'moment': 24}
        
        self.atom_table = {'atom': 0, 'ion': 1, 'occupancy': 2, 
                           'U11': 3, 'U22': 4, 'U33': 5, 
                           'U23': 6, 'U13': 7, 'U12': 8,
                           'mu1': 9, 'mu2': 10, 'mu3': 11, 'g': 12,
                           'u': 13, 'v': 14, 'w': 15, 'active': 16}
        
        self.label_map = {'site': 'site', 'atom': 'atom', 
                          'isotope': 'isotope', 'ion': 'ion',
                          'occupancy': 'occ', 'Uiso': 'Uiso', 
                          'U11': 'U\u2081\u2081', 'U13': 'U\u2081\u2083', 
                          'U22': 'U\u2082\u2082', 'U23': 'U\u2082\u2083', 
                          'U33': 'U\u2083\u2083', 'U12': 'U\u2081\u2082',
                          'U1': 'U\u2081', 'U2': 'U\u2082', 'U3': 'U\u2083',
                          'mu': 'mu', 'mu1': 'mu\u2081',  'mu2': 'mu\u2082', 
                          'mu3': 'mu\u2083', 'g': 'g', 'u': 'x', 'v': 'y', 
                          'w': 'z', 'operator': 'operator', 'moment': 'moment'}
        
    def create_thread(self):
        return QtCore.QThread()
    
    def create_thread_pool(self):
        return QtCore.QThreadPool()
    
    def worker(self, *args, **kwargs):
        return Worker(*args, **kwargs)
        
    def progress(self, worker, slot):
        worker.signals.progress.connect(slot)
        
    def result(self, worker, slot):
        worker.signals.result.connect(slot)
        
    def finished(self, worker, slot):
        worker.signals.finished.connect(slot)
        
    def offload(self, worker, thread):
        worker.moveToThread(thread)
        
    def new_triggered(self, slot): 
        self.actionNew.triggered.connect(slot)
    
    def timer(parent):
        return QtCore.QTimer(parent)
    
    def timeout(self, timer, slot):
        timer.timeout.connect(slot)
        
    def clear_application(self):
        self.lineEdit_n_atm.setText('')
        
        self.lineEdit_nu.setText('1')
        self.lineEdit_nv.setText('1')
        self.lineEdit_nw.setText('1')
        
        self.lineEdit_n.setText('')
            
        self.lineEdit_space_group.setText('')
        self.lineEdit_space_group_hm.setText('')
        
        self.lineEdit_a.setText('')
        self.lineEdit_b.setText('')
        self.lineEdit_c.setText('')
        
        self.lineEdit_alpha.setText('')
        self.lineEdit_beta.setText('')
        self.lineEdit_gamma.setText('')
        
        self.lineEdit_lat.setText('')

        self.tableWidget_CIF.clearContents()        
        self.tableWidget_CIF.setRowCount(0)
        self.tableWidget_CIF.setColumnCount(0)

        self.tableWidget_atm.clearContents()        
        self.tableWidget_atm.setRowCount(0)
        self.tableWidget_atm.setColumnCount(0)
           
        self.set_a_visible(False)
        self.set_b_visible(False)
        self.set_c_visible(False)
        self.set_alpha_visible(False)
        self.set_beta_visible(False)
        self.set_gamma_visible(False)
        
        # ---
        
        self.comboBox_rebin_h.clear()
        self.comboBox_rebin_k.clear()
        self.comboBox_rebin_l.clear()
        
        self.lineEdit_min_h.setText('')
        self.lineEdit_min_k.setText('')
        self.lineEdit_min_l.setText('')
        
        self.lineEdit_max_h.setText('')
        self.lineEdit_max_k.setText('')
        self.lineEdit_max_l.setText('')
        
        self.lineEdit_radius_h.setText('0')
        self.lineEdit_radius_k.setText('0')
        self.lineEdit_radius_l.setText('0')
        
        self.lineEdit_outlier.setText('1.5')
        
        self.lineEdit_slice_h.setText('')
        self.lineEdit_slice_k.setText('')
        self.lineEdit_slice_l.setText('')
        
        self.lineEdit_min_exp.setText('')       
        self.lineEdit_max_exp.setText('')       

        try: self.tableWidget_exp.disconnect() 
        except Exception: pass  
      
        self.tableWidget_exp.clearContents()
        self.tableWidget_exp.setRowCount(0)
        self.tableWidget_exp.setColumnCount(0)
        
        self.canvas_exp_h.figure.clear()
        self.canvas_exp_k.figure.clear()
        self.canvas_exp_l.figure.clear()
        self.canvas_exp_h.draw()
        self.canvas_exp_k.draw()
        self.canvas_exp_l.draw()
        
        # ---
                
        self.lineEdit_cycles.setText('10')
       
        self.lineEdit_filter_ref_h.setText('0.0')
        self.lineEdit_filter_ref_k.setText('0.0')
        self.lineEdit_filter_ref_l.setText('0.0')

        self.checkBox_batch.setChecked(False)
        self.lineEdit_runs.setText('1')
        self.lineEdit_runs.setEnabled(False)
        
        self.checkBox_fixed_moment.setChecked(True)
        self.checkBox_fixed_composition.setChecked(True)
        self.checkBox_fixed_displacement.setChecked(True)
        
        self.lineEdit_run.setEnabled(False)
        self.lineEdit_run.setText('0')

        self.lineEdit_order.setText('2')
       
        self.lineEdit_slice.setText('')
        
        self.comboBox_slice.setCurrentIndex(2)
        
        self.lineEdit_prefactor.setText('%1.2e' % 1e+4)
        self.lineEdit_tau.setText('%1.2e' % 1e-3)
        
        self.lineEdit_min_ref.setText('')
        self.lineEdit_max_ref.setText('')

        self.lineEdit_chi_sq.setText('') 

        self.comboBox_plot_top_chi_sq.setCurrentIndex(0)
        self.comboBox_plot_bottom_chi_sq.setCurrentIndex(1)
        
        self.comboBox_type.setCurrentIndex(0)
        
        self.checkBox_mag.setEnabled(True)
        
        self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
        self.checkBox_occ.setCheckState(QtCore.Qt.Checked)
        self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)
        
        self.canvas_ref.figure.clear()
        self.canvas_chi_sq.figure.clear()
        self.canvas_ref.draw()
        self.canvas_chi_sq.draw()
        
        # ---
    
        self.tableWidget_pairs_1d.clearContents()
        self.tableWidget_pairs_1d.setRowCount(0)
        self.tableWidget_pairs_1d.setColumnCount(0)
        
        self.tableWidget_pairs_3d.clearContents()
        self.tableWidget_pairs_3d.setRowCount(0)
        self.tableWidget_pairs_3d.setColumnCount(0)
        
        self.comboBox_correlations_1d.setCurrentIndex(1)   
        self.comboBox_correlations_3d.setCurrentIndex(1)   
        
        self.lineEdit_fract_1d.setText('0.125')
        self.lineEdit_fract_3d.setText('0.125')
        
        self.lineEdit_tol_1d.setText('1e-04')
        self.lineEdit_tol_3d.setText('1e-04')
        
        self.lineEdit_plane_h.setText('0')
        self.lineEdit_plane_k.setText('0')
        self.lineEdit_plane_l.setText('1')
        self.lineEdit_plane_d.setText('0.0')
        
        self.checkBox_batch_corr_1d.setCheckState(QtCore.Qt.Unchecked)
        self.lineEdit_runs_corr_1d.setText('1')
        self.lineEdit_runs_corr_1d.setEnabled(False)

        self.checkBox_batch_corr_3d.setCheckState(QtCore.Qt.Unchecked)
        self.lineEdit_runs_corr_3d.setText('1')
        self.lineEdit_runs_corr_3d.setEnabled(False)
        
        self.checkBox_average_1d.setCheckState(QtCore.Qt.Checked)
        self.checkBox_average_3d.setCheckState(QtCore.Qt.Checked)
        
        self.canvas_1d.figure.clear()
        self.canvas_3d.figure.clear()
        self.canvas_1d.draw()
        self.canvas_3d.draw()
        
        # ---
        
        self.checkBox_mag_recalc.setEnabled(False)
        self.checkBox_occ_recalc.setEnabled(False)
        self.checkBox_dis_recalc.setEnabled(False)
        
        self.lineEdit_type.setText('')   
                
        self.checkBox_mag_recalc.setCheckState(QtCore.Qt.Unchecked)
        self.checkBox_occ_recalc.setCheckState(QtCore.Qt.Unchecked)
        self.checkBox_dis_recalc.setCheckState(QtCore.Qt.Unchecked)
        
        try: self.tableWidget_calc.disconnect() 
        except Exception: pass
                        
        self.tableWidget_calc.clearContents()
        self.tableWidget_calc.setRowCount(0)
        self.tableWidget_calc.setColumnCount(0)

        self.tableWidget_recalc.clearContents()
        self.tableWidget_recalc.setRowCount(0)
        self.tableWidget_recalc.setColumnCount(0)
        
        self.tableWidget_recalc.clearContents()
        self.tableWidget_recalc.setRowCount(0)
        self.tableWidget_recalc.setColumnCount(0)
        
        self.lineEdit_order_calc.setText('2')
        
        self.comboBox_slice_calc.setCurrentIndex(2)

        self.lineEdit_slice_calc.setText('')
        
        self.lineEdit_min_calc.setText('')       
        self.lineEdit_max_calc.setText('')

        self.checkBox_batch_calc.setCheckState(QtCore.Qt.Unchecked)
        self.lineEdit_runs_calc.setText('1')
        self.lineEdit_runs_calc.setEnabled(False)
        
        self.canvas_calc.figure.clear()
        self.canvas_calc.draw()
        
    def save_as_triggered(self, slot):
        self.actionSave_As.triggered.connect(slot)

    def save_triggered(self, slot):
        self.actionSave.triggered.connect(slot)
        
    def open_dialog_save(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder,
                                                        'ini files *.ini',
                                                        options=options)   
        return filename
        
    def save_widgets(self, filename):
        settings = QtCore.QSettings(filename, QtCore.QSettings.IniFormat)
        save_gui(self, settings)

    def open_triggered(self, slot):
        self.actionOpen.triggered.connect(slot)
        
    def open_dialog_load(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
                                                        folder, 
                                                        'ini files *.ini',
                                                        options=options)   
        return filename
        
    def load_widgets(self, filename):
        settings = QtCore.QSettings(filename, QtCore.QSettings.IniFormat)
        load_gui(self, settings)

    def exit_triggered(self, slot):
        self.actionExit.triggered.connect(slot)

    def close_application(self):
        choice = QtWidgets.QMessageBox.question(self, 'Exit application', 
                                                'Are you sure?',
                                                QtWidgets.QMessageBox.Yes |
                                                QtWidgets.QMessageBox.No,
                                                QtWidgets.QMessageBox.Yes)
        
        return choice == QtWidgets.QMessageBox.Yes
    
    def finished_editing_a(self, slot):
        self.lineEdit_a.editingFinished.connect(slot)
        
    def finished_editing_b(self, slot):
        self.lineEdit_b.editingFinished.connect(slot)
        
    def finished_editing_c(self, slot):
        self.lineEdit_c.editingFinished.connect(slot)
        
    def finished_editing_alpha(self, slot):
        self.lineEdit_alpha.editingFinished.connect(slot)
        
    def finished_editing_beta(self, slot):
        self.lineEdit_beta.editingFinished.connect(slot)
        
    def finished_editing_gamma(self, slot):
        self.lineEdit_gamma.editingFinished.connect(slot)
    
    def set_a_visible(self, visible):
        self.lineEdit_a.setEnabled(visible)
        
    def set_b_visible(self, visible):
        self.lineEdit_b.setEnabled(visible)
        
    def set_c_visible(self, visible):
        self.lineEdit_c.setEnabled(visible)
        
    def set_alpha_visible(self, visible):
        self.lineEdit_alpha.setEnabled(visible)
        
    def set_beta_visible(self, visible):
        self.lineEdit_beta.setEnabled(visible)
        
    def set_gamma_visible(self, visible):
        self.lineEdit_gamma.setEnabled(visible)
    
    def get_table_item_info(self, item):
        return item.row(), item.column(), item.text()
        
    def get_every_site(self):
        return self.get_every_unit_cell_table_col(self.unit_table['site'], int)
    
    def get_every_atom(self):
        return self.get_every_unit_cell_table_col(self.unit_table['atom'], str)
    
    def get_every_isotope(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['isotope'], str)
    
    def get_every_ion(self):
        return self.get_every_unit_cell_table_col(self.unit_table['ion'], str)

    def get_every_occupancy(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['occupancy'], float)

    def get_every_Uiso(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['Uiso'], float)
    
    def get_every_U11(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U11'], float)
    
    def get_every_U22(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U22'], float)
    
    def get_every_U33(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U33'], float)
    
    def get_every_U23(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U23'], float)
    
    def get_every_U13(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U13'], float)
    
    def get_every_U12(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['U12'], float)
    
    def get_every_U1(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U1'], float)
    
    def get_every_U2(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U2'], float)
    
    def get_every_U3(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U3'], float)

    def get_every_mu(self):
        return self.get_every_unit_cell_table_col(self.unit_table['mu'], float)
    
    def get_every_mu1(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['mu1'], float)
    
    def get_every_mu2(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['mu2'], float)
    
    def get_every_mu3(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['mu3'], float)
   
    def get_every_g(self):
        return self.get_every_unit_cell_table_col(self.unit_table['g'], float)
    
    def get_every_u(self):
        return self.get_every_unit_cell_table_col(self.unit_table['u'], float)
    
    def get_every_v(self):
        return self.get_every_unit_cell_table_col(self.unit_table['v'], float)
    
    def get_every_w(self):
        return self.get_every_unit_cell_table_col(self.unit_table['w'], float)
    
    def get_every_operator(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['operator'], str)

    def get_every_magnetic_operator(self):
        return self.get_every_unit_cell_table_col(
                   self.unit_table['moment'], str)    
    
    # ---

    def get_site(self):
        return self.get_unit_cell_table_col(self.unit_table['site'], int)

    def get_atom(self):
        return self.get_unit_cell_table_col(self.unit_table['atom'], str)
    
    def get_isotope(self):
        return self.get_unit_cell_table_col(self.unit_table['isotope'], str)
    
    def get_ion(self):
        return self.get_unit_cell_table_col(self.unit_table['ion'], str)

    def get_occupancy(self):
        return self.get_unit_cell_table_col(
                    self.unit_table['occupancy'], float)

    def get_Uiso(self):
        return self.get_unit_cell_table_col(self.unit_table['Uiso'], float)
    
    def get_U11(self):
        return self.get_unit_cell_table_col(self.unit_table['U11'], float)
    
    def get_U22(self):
        return self.get_unit_cell_table_col(self.unit_table['U22'], float)
    
    def get_U33(self):
        return self.get_unit_cell_table_col(self.unit_table['U33'], float)
    
    def get_U23(self):
        return self.get_unit_cell_table_col(self.unit_table['U23'], float)
    
    def get_U13(self):
        return self.get_unit_cell_table_col(self.unit_table['U13'], float)
    
    def get_U12(self):
        return self.get_unit_cell_table_col(self.unit_table['U12'], float)
    
    def get_U1(self):
        return self.get_unit_cell_table_col(self.unit_table['U1'], float)
    
    def get_U2(self):
        return self.get_unit_cell_table_col(self.unit_table['U2'], float)
    
    def get_U3(self):
        return self.get_unit_cell_table_col(self.unit_table['U3'], float)

    def get_mu(self):
        return self.get_unit_cell_table_col(self.unit_table['mu'], float)
    
    def get_mu1(self):
        return self.get_unit_cell_table_col(self.unit_table['mu1'], float)
    
    def get_mu2(self):
        return self.get_unit_cell_table_col(self.unit_table['mu2'], float)
    
    def get_mu3(self):
        return self.get_unit_cell_table_col(self.unit_table['mu3'], float)
   
    def get_g(self):
        return self.get_unit_cell_table_col(self.unit_table['g'], float)
    
    def get_u(self):
        return self.get_unit_cell_table_col(self.unit_table['u'], float)
    
    def get_v(self):
        return self.get_unit_cell_table_col(self.unit_table['v'], float)
    
    def get_w(self):
        return self.get_unit_cell_table_col(self.unit_table['w'], float)
        
    def get_nu(self):
        return int(self.lineEdit_nu.text())
    
    def get_nv(self):
        return int(self.lineEdit_nv.text())
    
    def get_nw(self):
        return int(self.lineEdit_nw.text())
    
    def finished_editing_nu(self, slot):
        self.lineEdit_nu.editingFinished.connect(slot)

    def finished_editing_nv(self, slot):
        self.lineEdit_nv.editingFinished.connect(slot)

    def finished_editing_nw(self, slot):
        self.lineEdit_nw.editingFinished.connect(slot)
        
    def get_n_atm(self):
        n_atm = self.lineEdit_n_atm.text()
        if (n_atm == ''):
            return None
        else:
            return int(n_atm)
        
    def set_n_atm(self, value):
        self.lineEdit_n_atm.setText(str(value))
        
    def get_n(self):
        return int(self.lineEdit_n.text())
        
    def set_n(self, value):
        self.lineEdit_n.setText(str(value))
        
    def get_lattice_parameters(self):
        return float(self.lineEdit_a.text()), \
               float(self.lineEdit_b.text()), \
               float(self.lineEdit_c.text()), \
               np.deg2rad(float(self.lineEdit_alpha.text())), \
               np.deg2rad(float(self.lineEdit_beta.text())), \
               np.deg2rad(float(self.lineEdit_gamma.text()))
        
    def set_lattice_parameters(self, a, b, c, alpha, beta, gamma):
        alpha = np.round(np.rad2deg(alpha),8)
        beta = np.round(np.rad2deg(beta),8)
        gamma = np.round(np.rad2deg(gamma),8)
        self.lineEdit_a.setText(str(a))
        self.lineEdit_b.setText(str(b))
        self.lineEdit_c.setText(str(c))
        self.lineEdit_alpha.setText(str(alpha))
        self.lineEdit_beta.setText(str(beta))
        self.lineEdit_gamma.setText(str(gamma))
        
    def get_lattice(self):
        return self.lineEdit_lat.text()
        
    def set_lattice(self, lat):
        self.lineEdit_lat.setText(lat)
        
    def set_space_group(self, group, hm):
        self.lineEdit_space_group.setText(str(group))
        self.lineEdit_space_group_hm.setText(hm)
        
    def open_dialog_cif(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
                                                        folder, 
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options)
        return filename
    
    def button_clicked_CIF(self, slot):
        self.pushButton_load_CIF.clicked.connect(slot)
        
    def enable_load_CIF(self, visible):
        self.pushButton_load_CIF.setEnabled(visible)
        
    def get_every_unit_cell_table_col(self, j, dtype):       
        data = []
        for i in range(self.tableWidget_CIF.rowCount()):
            item = self.tableWidget_CIF.item(i, j)
            if (item is not None):
                data.append(str(item.text()))
            else:
                data.append(str(''))
        return np.array(data).astype(dtype)
                
    def get_unit_cell_table_col(self, j, dtype):
        data = []
        for i in range(self.tableWidget_CIF.rowCount()):
            if (not self.tableWidget_CIF.isRowHidden(i)):
                data.append(str(self.tableWidget_CIF.item(i, j).text()))
        return np.array(data).astype(dtype)
    
    def get_atom_site_table_col(self, j, dtype):
        data = []
        for i in range(self.tableWidget_atm.rowCount()):
            if (not self.tableWidget_atm.isRowHidden(i)):
                data.append(str(self.tableWidget_atm.item(i, j).text()))
        return np.array(data).astype(dtype)
    
    def set_unit_cell_table_col(self, data, j):
        for i in range(self.tableWidget_CIF.rowCount()):
            text = str(data[i])
            if not text: text = '-'
            item = QtWidgets.QTableWidgetItem(text)
            self.tableWidget_CIF.setItem(i, j, item)
            
    def set_atom_site_table_col(self, data, j):
        for i in range(self.tableWidget_atm.rowCount()):
            text = str(data[i])
            if not text: text = '-'
            item = QtWidgets.QTableWidgetItem(text)
            self.tableWidget_atm.setItem(i, j, item)
            
    def set_unit_cell_site(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['site'])
        
    def set_unit_cell_atom(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['atom'])

    def set_unit_cell_isotope(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['isotope'])
        
    def set_unit_cell_ion(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['ion'])
           
    def set_unit_cell_occupancy(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['occupancy'])
        
    def set_unit_cell_Uiso(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['Uiso'])
        
    def set_unit_cell_U11(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U11'])
        
    def set_unit_cell_U22(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U22'])
        
    def set_unit_cell_U33(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U33'])
         
    def set_unit_cell_U23(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U23'])
        
    def set_unit_cell_U13(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U13'])
        
    def set_unit_cell_U12(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U12'])

    def set_unit_cell_U1(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U1'])
        
    def set_unit_cell_U2(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U2'])
        
    def set_unit_cell_U3(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['U3'])

    def set_unit_cell_mu(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['mu'])
        
    def set_unit_cell_mu1(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['mu1'])
        
    def set_unit_cell_mu2(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['mu2'])
        
    def set_unit_cell_mu3(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['mu3'])
        
    def set_unit_cell_g(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['g'])
        
    def set_unit_cell_u(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['u'])
    
    def set_unit_cell_v(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['v'])
    
    def set_unit_cell_w(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['w'])
        
    def set_unit_cell_operator(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['operator'])
    
    def set_unit_cell_magnetic_operator(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['moment'])
        
    def set_atom_site_occupancy(self, data):
        self.set_atom_site_table_col(data, self.atom_table['occupancy'])
        
    def set_atom_site_U11(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U11'])
        
    def set_atom_site_U22(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U22'])
        
    def set_atom_site_U33(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U33'])
         
    def set_atom_site_U23(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U23'])
        
    def set_atom_site_U13(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U13'])
        
    def set_atom_site_U12(self, data):
        self.set_atom_site_table_col(data, self.atom_table['U12'])
        
    def set_atom_site_mu1(self, data):
        self.set_atom_site_table_col(data, self.atom_table['mu1'])
        
    def set_atom_site_mu2(self, data):
        self.set_atom_site_table_col(data, self.atom_table['mu2'])
        
    def set_atom_site_mu3(self, data):
        self.set_atom_site_table_col(data, self.atom_table['mu3'])
        
    def set_atom_site_g(self, data):
        self.set_atom_site_table_col(data, self.atom_table['g'])
    
    def set_atom_site_u(self, data):
        self.set_atom_site_table_col(data, self.atom_table['u'])
    
    def set_atom_site_v(self, data):
        self.set_atom_site_table_col(data, self.atom_table['v'])
    
    def set_atom_site_w(self, data):
        self.set_atom_site_table_col(data, self.atom_table['w'])
        
    def create_unit_cell_table(self, n_atm):
        self.tableWidget_CIF.setRowCount(n_atm)
        self.tableWidget_CIF.setColumnCount(len(self.unit_table))
        
        horiz_lbl = [key for key in self.unit_table.keys()]
        self.tableWidget_CIF.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['{}'.format(s+1) for s in range(n_atm)]
        self.tableWidget_CIF.setVerticalHeaderLabels(vert_lbl)
    
    def create_atom_site_table(self, n_site):
        self.tableWidget_atm.setRowCount(n_site)
        self.tableWidget_atm.setColumnCount(len(self.atom_table))
        
        horiz_lbl = [key for key in self.atom_table.keys()]
        self.tableWidget_atm.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['{}'.format(s+1) for s in range(n_site)]
        self.tableWidget_atm.setVerticalHeaderLabels(vert_lbl)
        
    def clear_unit_cell_table(self):
        # self.tableWidget_CIF.disconnect()
        self.tableWidget_CIF.clearContents()
        self.tableWidget_CIF.setRowCount(0)
        self.tableWidget_CIF.setColumnCount(0)
        
    def clear_atom_site_table(self):
        # self.tableWidget_atm.disconnect()
        self.tableWidget_atm.clearContents()
        self.tableWidget_atm.setRowCount(0)
        self.tableWidget_atm.setColumnCount(0)
        
    def format_atom_site_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        resize = QtWidgets.QHeaderView.ResizeToContents

        for i in range(self.tableWidget_atm.rowCount()):
            for j in range(self.tableWidget_atm.columnCount()):
                item = self.tableWidget_atm.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
        
        horiz_hdr = self.tableWidget_atm.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        horiz_hdr.setSectionResizeMode(self.atom_table['occupancy'], resize)
        
        for col_name in ['occupancy', 'u', 'v', 'w']:
            j = self.atom_table[col_name]
            delegate = FractionalDelegate(self.tableWidget_atm)
            self.tableWidget_atm.setItemDelegateForColumn(j, delegate)

        for col_name in ['mu1', 'mu2', 'mu3', 'g']:
            j = self.atom_table[col_name]
            delegate = StandardDoubleDelegate(self.tableWidget_atm)
            self.tableWidget_atm.setItemDelegateForColumn(j, delegate)            
            
        for col_name in ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']:
            j = self.atom_table[col_name]
            delegate = StandardDoubleDelegate(self.tableWidget_atm)
            self.tableWidget_atm.setItemDelegateForColumn(j, delegate)            
    
    def format_unit_cell_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        resize = QtWidgets.QHeaderView.ResizeToContents
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        
        for i in range(self.tableWidget_CIF.rowCount()):
            for j in range(self.tableWidget_CIF.columnCount()):
                item = self.tableWidget_CIF.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    item.setFlags(flags)
                                
        horiz_hdr = self.tableWidget_CIF.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        horiz_hdr.setSectionResizeMode(self.unit_table['occupancy'], resize)
        
    def format_atom_site_table_col(self, j):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        for i in range(self.tableWidget_atm.rowCount()):
            item = self.tableWidget_atm.item(i, j)
            if (item is not None and item.text() != ''):
                item.setTextAlignment(alignment)
                
    def format_unit_cell_table_col(self, j):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        for i in range(self.tableWidget_CIF.rowCount()):
            item = self.tableWidget_CIF.item(i, j)
            if (item is not None and item.text() != ''):
                item.setTextAlignment(alignment)
    
    def show_atom_site_table_cols(self):
        disorder_index = self.comboBox_type.currentIndex()    
        disorder_type = self.comboBox_type.itemText(disorder_index)
        
        index = self.comboBox_parameters.currentIndex()    
        parameters = self.comboBox_parameters.itemText(index)
        
        if (parameters == 'Site parameters'):
            if (disorder_type == 'Neutron'):
                cols = ['atom']
            else:
                cols = ['ion']
            cols += ['occupancy','u','v','w','active']
        elif (parameters == 'Structural parameters'):
            cols = ['U11','U22','U33','U23','U13','U12']
        else:
            cols = ['ion','mu1','mu2','mu3','g','active']
            
        show = [self.atom_table[key] for key in cols]         
        for i in range(len(self.atom_table)):
            if i in show:
                self.tableWidget_atm.setColumnHidden(i, False)
            else:
                self.tableWidget_atm.setColumnHidden(i, True)
                
    def show_unit_cell_table_cols(self):
        disorder_index = self.comboBox_type.currentIndex()    
        disorder_type = self.comboBox_type.itemText(disorder_index)
        
        index = self.comboBox_parameters.currentIndex()    
        parameters = self.comboBox_parameters.itemText(index)
        
        cols = ['site','atom']
        if (disorder_type == 'Neutron' and \
            parameters != 'Magnetic parameters'):
            cols += ['isotope']
        else:
            cols += ['ion']
        if (parameters == 'Site parameters'):
            cols += ['occupancy','u','v','w']
        elif (parameters == 'Structural parameters'):
            cols += ['Uiso','U1','U2','U3']
        else:
            cols += ['mu','mu1','mu2','mu3']
            
        show = [self.unit_table[key] for key in cols]         
        for i in range(len(self.unit_table)):
            if i in show:
                self.tableWidget_CIF.setColumnHidden(i, False)
            else:
                self.tableWidget_CIF.setColumnHidden(i, True)
                
    def unit_site_col(self, col):
        key = list(self.atom_table.keys())[col]
        
        if key in ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']: key = 'Uiso'
        if key in ['U1', 'U2', 'U3']: key = 'Uiso'
        if key in ['g']: key = 'mu'
                
        if (self.unit_table.get(key) is None):
            return 0
        else:
            return self.unit_table[key]
    
    def select_site(self, slot):
        self.tableWidget_atm.clicked.connect(slot)
                
    def highlight_atoms(self, row_range, col):
        selection = QtWidgets.QTableWidgetSelectionRange(row_range[0], col, 
                                                         row_range[1], col)
        
        self.tableWidget_CIF.clearSelection()
        self.tableWidget_CIF.setRangeSelected(selection, True)
        
    def clear_atom_site_table_selection(self):
        self.tableWidget_atm.clearSelection()
        
    def clear_unit_cell_table_selection(self):
        self.tableWidget_CIF.clearSelection()
                
    def get_atom_site_table_row_count(self):
        return self.tableWidget_atm.rowCount()
    
    def get_atom_site_table_col_count(self):
        return self.tableWidget_atm.columnCount()
    
    def get_unit_cell_table_row_count(self):
        return self.tableWidget_CIF.rowCount()
    
    def get_unit_cell_table_col_count(self):
        return self.tableWidget_CIF.columnCount()
    
    def index_changed_type(self, slot):
        self.comboBox_type.currentIndexChanged.connect(slot)
        
    def get_type(self):
        index = self.comboBox_type.currentIndex()    
        return self.comboBox_type.itemText(index)
        
    def add_item_magnetic(self):
        if (self.comboBox_parameters.count() == 2):
            self.comboBox_parameters.addItem('Magnetic parameters')

    def remove_item_magnetic(self):
        if (self.comboBox_parameters.count() == 3):
            self.comboBox_parameters.removeItem(2)
        
    def index_changed_parameters(self, slot):
        self.comboBox_parameters.currentIndexChanged.connect(slot)
        
    def add_atom_combo(self, data):
        j = self.atom_table['atom']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = QtWidgets.QComboBox()
            combo.setObjectName('comboBox_site'+str(i))
            for item in data:
                combo.addItem(item)
            self.tableWidget_atm.setCellWidget(i, j, combo)
            
    def add_ion_combo(self, data): 
        j = self.atom_table['ion']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = QtWidgets.QComboBox()
            combo.setObjectName('comboBox_ion'+str(i))
            for item in data:
                combo.addItem(item)
            self.tableWidget_atm.setCellWidget(i, j, combo)
            
    def add_mag_ion_combo(self, i, data): 
        j = self.atom_table['ion']
        combo = QtWidgets.QComboBox()
        combo.setObjectName('comboBox_ion'+str(i))
        for item in data:
            combo.addItem(item)
        self.tableWidget_atm.setCellWidget(i, j, combo)
            
    def set_atom_combo(self, atm):
        j = self.atom_table['atom']
        for i in range(self.tableWidget_atm.rowCount()):
            a = atm[i]
            combo = self.tableWidget_atm.cellWidget(i, j)
            if (combo is not None):
                index = combo.findText(a, QtCore.Qt.MatchFixedString)
                if (index < 0 and len(a) > 2):
                    index = combo.findText(a[:2], QtCore.Qt.MatchStartsWith)
                if (index < 0 and len(a) > 0):
                    index = combo.findText(a[0], QtCore.Qt.MatchStartsWith)
                if (index >= 0):
                    combo.setCurrentIndex(index)
            else:
                combo = QtWidgets.QComboBox()
                combo.setObjectName('comboBox_site'+str(i))
                combo.addItem(a)
                self.tableWidget_atm.setCellWidget(i, j, combo)
                        
    def set_ion_combo(self, atm):
        j = self.atom_table['ion']
        for i in range(self.tableWidget_atm.rowCount()):
            a = atm[i]
            combo = self.tableWidget_atm.cellWidget(i, j)
            if (combo is not None):
                index = combo.findText(a, QtCore.Qt.MatchFixedString)
                if (index < 0 and len(a) > 2):
                    index = combo.findText(a[:2], QtCore.Qt.MatchStartsWith)
                if (index < 0 and len(a) > 0):
                    index = combo.findText(a[0], QtCore.Qt.MatchStartsWith)
                if (index >= 0):
                    combo.setCurrentIndex(index)
            else:
                combo = QtWidgets.QComboBox()
                combo.setObjectName('comboBox_site'+str(i))
                combo.addItem(a)
                self.tableWidget_atm.setCellWidget(i, j, combo)
                
    def get_atom_combo(self):
        data = []
        j = self.atom_table['atom']
        for i in range(self.tableWidget_atm.rowCount()):
            widget = self.tableWidget_atm.cellWidget(i, j)
            text = widget.currentText() if (widget is not None) else ''
            data.append(text)
        return np.array(data)
    
    def get_ion_combo(self):
        data = []
        j = self.atom_table['ion']
        for i in range(self.tableWidget_atm.rowCount()):
            widget = self.tableWidget_atm.cellWidget(i, j)
            text = widget.currentText() if (widget is not None) else ''
            data.append(text)
        return np.array(data)

    def index_changed_atom(self, slot):
        j = self.atom_table['atom']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = self.tableWidget_atm.cellWidget(i, j)
            combo.currentIndexChanged.connect(slot)
            
    def index_changed_ion(self, slot):
        j = self.atom_table['ion']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = self.tableWidget_atm.cellWidget(i, j)
            combo.currentIndexChanged.connect(slot)
    
    def add_site_check(self):
        j = self.atom_table['active']
        for i in range(self.tableWidget_atm.rowCount()):
            check = QtWidgets.QCheckBox()
            check.setObjectName('checkBox_site'+str(i))
            check.setCheckState(QtCore.Qt.Checked) 
            self.tableWidget_atm.setCellWidget(i, j, check)
    
    def check_clicked_site(self, slot):
        j = self.atom_table['active']
        for i in range(self.tableWidget_atm.rowCount()):
            check = self.tableWidget_atm.cellWidget(i, j)
            check.clicked.connect(slot)
            
    def change_site_check(self):
        n_atm = 0
        k, l = self.atom_table['active'], self.unit_table['site']
        for i in range(self.tableWidget_atm.rowCount()):
            check = self.tableWidget_atm.cellWidget(i, k)
            site = self.tableWidget_atm.indexAt(check.pos()).row()
            for j in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(j, l).text())-1
                if (site == s):
                    if check.isChecked():
                        self.tableWidget_CIF.setRowHidden(j, False)
                        n_atm += 1
                    else:
                        self.tableWidget_CIF.setRowHidden(j, True)
        return n_atm
    
    def item_changed_atom_site_table(self, slot):
        self.tableWidget_atm.itemChanged.connect(slot)
                
    # ---
    
    def get_experiment_table_row_count(self):
        return self.tableWidget_exp.rowCount()
        
    def create_experiment_table(self):
        self.tableWidget_exp.setRowCount(3)
        self.tableWidget_exp.setColumnCount(4)
        
        horiz_lbl = ['step','size','min','max']
        self.tableWidget_exp.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['h','k','l']
        self.tableWidget_exp.setVerticalHeaderLabels(vert_lbl)
        
    def clear_experiment_table(self):
        try: self.tableWidget_exp.disconnect() 
        except Exception: pass  
        self.tableWidget_exp.clearContents()
        self.tableWidget_exp.setRowCount(0)
        self.tableWidget_exp.setColumnCount(0)
        
    def format_experiment_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        for i in range(self.tableWidget_exp.rowCount()):
            for j in range(self.tableWidget_exp.columnCount()):
                item = self.tableWidget_exp.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    if (j == 0): item.setFlags(flags)

        delegate = SizeIntDelegate(self.tableWidget_exp)
        self.tableWidget_exp.setItemDelegateForColumn(1, delegate)
        delegate = StandardDoubleDelegate(self.tableWidget_exp)
        self.tableWidget_exp.setItemDelegateForColumn(2, delegate)
        self.tableWidget_exp.setItemDelegateForColumn(3, delegate)
        
        horiz_hdr = self.tableWidget_exp.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)

    def get_experiment_binning_h(self):       
        i = 0
        step = float(self.tableWidget_exp.item(i, 0).text()) 
        size = int(self.tableWidget_exp.item(i, 1).text()) 
        mininum = float(self.tableWidget_exp.item(i, 2).text())
        maximum = float(self.tableWidget_exp.item(i, 3).text())
        return step, size, mininum, maximum
                
    def get_experiment_binning_k(self):       
        i = 1
        step = float(self.tableWidget_exp.item(i, 0).text()) 
        size = int(self.tableWidget_exp.item(i, 1).text()) 
        mininum = float(self.tableWidget_exp.item(i, 2).text())
        maximum = float(self.tableWidget_exp.item(i, 3).text())
        return step, size, mininum, maximum
    
    def get_experiment_binning_l(self):       
        i = 2
        step = float(self.tableWidget_exp.item(i, 0).text())
        size = int(self.tableWidget_exp.item(i, 1).text()) 
        mininum = float(self.tableWidget_exp.item(i, 2).text())
        maximum = float(self.tableWidget_exp.item(i, 3).text())
        return step, size, mininum, maximum
    
    def set_experiment_binning_h(self, size, minimum, maximum):
        i = 0
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_exp.setItem(i, 0, item)

        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_exp.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_exp.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_exp.setItem(i, 3, item)  
        
    def set_experiment_binning_k(self, size, minimum, maximum):
        i = 1
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_exp.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_exp.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_exp.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_exp.setItem(i, 3, item)  
        
    def set_experiment_binning_l(self, size, minimum, maximum):
        i = 2
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_exp.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_exp.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_exp.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_exp.setItem(i, 3, item)  
    
    def set_experiment_table_item(self, i, j, value):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        item = QtWidgets.QTableWidgetItem(str(value))
        item.setTextAlignment(alignment)
        self.tableWidget_exp.setItem(i, j, item)
                    
    def item_changed_experiment_table(self, slot):
        self.tableWidget_exp.itemChanged.connect(slot)
        
    def block_experiment_table_signals(self):
        self.tableWidget_exp.blockSignals(True)

    def unblock_experiment_table_signals(self):
        self.tableWidget_exp.blockSignals(False)
        
    def enable_cropbin_signals(self, visible):
        self.comboBox_rebin_h.setEnabled(visible)
        self.comboBox_rebin_k.setEnabled(visible)
        self.comboBox_rebin_l.setEnabled(visible)
        self.checkBox_centered_h.setEnabled(visible)
        self.checkBox_centered_k.setEnabled(visible)
        self.checkBox_centered_l.setEnabled(visible)
        self.lineEdit_min_h.setEnabled(visible)
        self.lineEdit_min_k.setEnabled(visible)
        self.lineEdit_min_l.setEnabled(visible)
        self.lineEdit_max_h.setEnabled(visible)
        self.lineEdit_max_k.setEnabled(visible)
        self.lineEdit_max_l.setEnabled(visible)
        self.pushButton_punch.setEnabled(visible)
        self.pushButton_reset_punch.setEnabled(visible)
        self.pushButton_reset.setEnabled(visible)
        self.pushButton_reset_h.setEnabled(visible)
        self.pushButton_reset_k.setEnabled(visible)
        self.pushButton_reset_l.setEnabled(visible)
        
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        for i in range(self.tableWidget_exp.rowCount()):
            for j in range(self.tableWidget_exp.columnCount()):
                item = self.tableWidget_exp.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    if visible:
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)
                    else:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)                        
                        
    def set_rebin_combo_h(self, steps, sizes):
        for step, size in zip(steps, sizes):            
            parameters = 'h-step : {}, h-size: {}'.format(step,size)
            self.comboBox_rebin_h.addItem(parameters)
 
    def set_rebin_combo_k(self, steps, sizes):
        for step, size in zip(steps, sizes):            
            parameters = 'k-step : {}, k-size: {}'.format(step,size)
            self.comboBox_rebin_k.addItem(parameters)
            
    def set_rebin_combo_l(self, steps, sizes):
        for step, size in zip(steps, sizes):            
            parameters = 'l-step : {}, l-size: {}'.format(step,size)
            self.comboBox_rebin_l.addItem(parameters)
            
    def get_rebin_combo_h(self):
        text = self.comboBox_rebin_h.currentText().split(':')[-1]
        if (text != ''): return int(text)
    
    def get_rebin_combo_k(self):
        text = self.comboBox_rebin_k.currentText().split(':')[-1]
        if (text != ''): return int(text)  
        
    def get_rebin_combo_l(self):
        text = self.comboBox_rebin_l.currentText().split(':')[-1]
        if (text != ''): return int(text)
            
    def clear_rebin_combo_h(self):
        self.comboBox_rebin_h.clear()
 
    def clear_rebin_combo_k(self):
        self.comboBox_rebin_k.clear()

    def clear_rebin_combo_l(self):
        self.comboBox_rebin_l.clear()
        
    def index_changed_combo_h(self, slot):
        self.comboBox_rebin_h.currentIndexChanged.connect(slot)
        
    def index_changed_combo_k(self, slot):
        self.comboBox_rebin_k.currentIndexChanged.connect(slot)
        
    def index_changed_combo_l(self, slot):
        self.comboBox_rebin_l.currentIndexChanged.connect(slot)
        
    def block_changed_combo_h(self, block):
        self.comboBox_rebin_h.blockSignals(block)
        
    def block_changed_combo_k(self, block):
        self.comboBox_rebin_k.blockSignals(block)
        
    def block_changed_combo_l(self, block):
        self.comboBox_rebin_l.blockSignals(block)
                   
    def centered_h_checked(self):
        return self.checkBox_centered_h.isChecked()
    
    def centered_k_checked(self):
        return self.checkBox_centered_k.isChecked()
    
    def centered_l_checked(self):
        return self.checkBox_centered_l.isChecked()
    
    def clicked_centered_h(self, slot):
        self.checkBox_centered_h.clicked.connect(slot)

    def clicked_centered_k(self, slot):
        self.checkBox_centered_k.clicked.connect(slot)

    def clicked_centered_l(self, slot):
        self.checkBox_centered_l.clicked.connect(slot)
                
    def set_min_h(self, value):
        self.lineEdit_min_h.setText(str(value))

    def set_min_k(self, value):
        self.lineEdit_min_k.setText(str(value))
        
    def set_min_l(self, value):
        self.lineEdit_min_l.setText(str(value))
    
    def set_max_h(self, value):
        self.lineEdit_max_h.setText(str(value))
        
    def set_max_k(self, value):
        self.lineEdit_max_k.setText(str(value))
        
    def set_max_l(self, value):
        self.lineEdit_max_l.setText(str(value))
        
    def get_min_h(self):
        return float(self.lineEdit_min_h.text())
    
    def get_min_k(self):
        return float(self.lineEdit_min_k.text())
    
    def get_min_l(self):
        return float(self.lineEdit_min_l.text())
    
    def get_max_h(self):
        return float(self.lineEdit_max_h.text())
    
    def get_max_k(self):
        return float(self.lineEdit_max_k.text())
    
    def get_max_l(self):
        return float(self.lineEdit_max_l.text())
    
    def finished_editing_min_h(self, slot):
        self.lineEdit_min_h.editingFinished.connect(slot)
 
    def finished_editing_min_k(self, slot):
        self.lineEdit_min_k.editingFinished.connect(slot)
        
    def finished_editing_min_l(self, slot):
        self.lineEdit_min_l.editingFinished.connect(slot)
        
    def finished_editing_max_h(self, slot):
        self.lineEdit_max_h.editingFinished.connect(slot)
 
    def finished_editing_max_k(self, slot):
        self.lineEdit_max_k.editingFinished.connect(slot)
        
    def finished_editing_max_l(self, slot):
        self.lineEdit_max_l.editingFinished.connect(slot)
 
    def validate_crop_h(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_min_h.setValidator(validator)
        self.lineEdit_max_h.setValidator(validator)

    def validate_crop_k(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_min_k.setValidator(validator)
        self.lineEdit_max_k.setValidator(validator)
        
    def validate_crop_l(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_min_l.setValidator(validator)
        self.lineEdit_max_l.setValidator(validator)
        
    def set_slice_h(self, value):
        self.lineEdit_slice_h.setText(str(value))

    def set_slice_k(self, value):
        self.lineEdit_slice_k.setText(str(value))

    def set_slice_l(self, value):
        self.lineEdit_slice_l.setText(str(value))
        
    def block_slices(self):
        self.lineEdit_slice_h.blockSignals(True)
        self.lineEdit_slice_k.blockSignals(True)
        self.lineEdit_slice_l.blockSignals(True)
        
    def unblock_slices(self):
        self.lineEdit_slice_h.blockSignals(False)
        self.lineEdit_slice_k.blockSignals(False)
        self.lineEdit_slice_l.blockSignals(False)
        
    def get_slice_h(self):
        text = self.lineEdit_slice_h.text()
        if (text != ''): return float(text) 

    def get_slice_k(self):
        text = self.lineEdit_slice_k.text()
        if (text != ''): return float(text) 
        
    def get_slice_l(self):
        text = self.lineEdit_slice_l.text()
        if (text != ''): return float(text) 
        
    def validate_slice_h(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_slice_h.setValidator(validator)
        
    def validate_slice_k(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_slice_k.setValidator(validator)
        
    def validate_slice_l(self, minimum, maximum):
        validator = QtGui.QDoubleValidator(minimum, maximum, 4)
        self.lineEdit_slice_l.setValidator(validator)
        
    def finished_editing_slice_h(self, slot):
        self.lineEdit_slice_h.editingFinished.connect(slot)
        
    def finished_editing_slice_k(self, slot):
        self.lineEdit_slice_k.editingFinished.connect(slot)
        
    def finished_editing_slice_l(self, slot):
        self.lineEdit_slice_l.editingFinished.connect(slot)
        
    def set_min_exp(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_min_exp.setText('{:1.4e}'.format(value))
 
    def set_max_exp(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_max_exp.setText('{:1.4e}'.format(value))
        
    def get_min_exp(self):
        return float(self.lineEdit_min_exp.text())

    def get_max_exp(self):
        return float(self.lineEdit_max_exp.text())
    
    def finished_editing_min_exp(self, slot):
        self.lineEdit_min_exp.editingFinished.connect(slot)

    def finished_editing_max_exp(self, slot):
        self.lineEdit_max_exp.editingFinished.connect(slot)
        
    def validate_min_exp(self):
        maximum = float(self.lineEdit_max_exp.text())
        validator = QtGui.QDoubleValidator(np.finfo(float).min, maximum, 4)
        self.lineEdit_min_exp.setValidator(validator)
        
    def validate_max_exp(self):
        minimum = float(self.lineEdit_min_exp.text())
        validator = QtGui.QDoubleValidator(minimum, np.finfo(float).max, 4)
        self.lineEdit_max_exp.setValidator(validator)
    
    def get_plot_exp(self):
        index = self.comboBox_plot_exp.currentIndex()    
        return self.comboBox_plot_exp.itemText(index)
    
    def get_norm_exp(self):
        index = self.comboBox_norm_exp.currentIndex()    
        return self.comboBox_norm_exp.itemText(index)
    
    def get_plot_exp_canvas(self):
        return self.canvas_exp_h, self.canvas_exp_k, self.canvas_exp_l
    
    def clear_plot_exp_canvas(self):
        self.canvas_exp_h.figure.clear()
        self.canvas_exp_k.figure.clear()
        self.canvas_exp_l.figure.clear()
        self.canvas_exp_cb.figure.clear()
        self.canvas_exp_h.draw()
        self.canvas_exp_k.draw()
        self.canvas_exp_l.draw()
        self.canvas_exp_cb.draw()
        self.lineEdit_min_exp.setText('')       
        self.lineEdit_max_exp.setText('')    

    def index_changed_plot_exp(self, slot):
        self.comboBox_plot_exp.currentIndexChanged.connect(slot)
        
    def index_changed_norm_exp(self, slot):
        self.comboBox_norm_exp.currentIndexChanged.connect(slot)
        
    def get_centering(self):
        return self.comboBox_centering.currentText()
    
    def set_centering(self, centering):
        match = QtCore.Qt.MatchFixedString
        index = self.comboBox_centering.findText(centering, match)
        self.comboBox_centering.setCurrentIndex(index)
    
    def get_radius_h(self):
        return float(self.lineEdit_radius_h.text())
    
    def get_radius_k(self):
        return float(self.lineEdit_radius_k.text())
    
    def get_radius_l(self):
        return float(self.lineEdit_radius_l.text())
    
    def get_outlier(self):
        return float(self.lineEdit_outlier.text())
    
    def get_punch(self):
        return self.comboBox_punch.currentText()
    
    def button_clicked_punch(self, slot):
        self.pushButton_punch.clicked.connect(slot)
        
    def button_clicked_reset_punch(self, slot):
        self.pushButton_reset_punch.clicked.connect(slot)
        
    def button_clicked_reset(self, slot):
        self.pushButton_reset.clicked.connect(slot)
        
    def button_clicked_reset_h(self, slot):
        self.pushButton_reset_h.clicked.connect(slot)
        
    def button_clicked_reset_k(self, slot):
        self.pushButton_reset_k.clicked.connect(slot)
        
    def button_clicked_reset_l(self, slot):
        self.pushButton_reset_l.clicked.connect(slot)
    
    def open_dialog_nxs(self):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '.', 
                                                        'NeXus files *.nxs',
                                                        'NumPy files *.npz',
                                                        options=options)
        return filename            
    
    def button_clicked_NXS(self, slot):
        self.pushButton_load_NXS.clicked.connect(slot)
        
    def enable_load_NXS(self, visible):
        self.pushButton_load_NXS.setEnabled(visible)
        
    def save_intensity_exp(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        
        return filename
            
    def button_clicked_save_intensity_exp(self, slot):
        self.pushButton_save_intensity_exp.clicked.connect(slot)
        
    # ---
    
    def batch_checked(self):
        return self.checkBox_batch.isChecked()
    
    def clicked_batch(self, slot):
        self.checkBox_batch.stateChanged.connect(slot)
        
    def enable_runs(self, visible):
        self.lineEdit_runs.setEnabled(visible)
        
    def enable_disorder_mag(self, visible):
        self.checkBox_mag.setEnabled(visible)
        
    def clicked_disorder_mag(self, slot):
        self.checkBox_mag.clicked.connect(slot)

    def clicked_disorder_occ(self, slot):
        self.checkBox_occ.clicked.connect(slot)
        
    def clicked_disorder_dis(self, slot):
        self.checkBox_dis.clicked.connect(slot)
        
    def set_disorder_mag(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_mag.setCheckState(check)
        
    def set_disorder_occ(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_occ.setCheckState(check)
        
    def set_disorder_dis(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_dis.setCheckState(check)

    def get_disorder_mag(self):
        return self.checkBox_mag.isChecked()
    
    def get_disorder_occ(self):
        return self.checkBox_occ.isChecked()
    
    def get_disorder_dis(self):
        return self.checkBox_dis.isChecked()
    
    def get_prefactor(self):
        return float(self.lineEdit_prefactor.text())
 
    def get_constant(self):
        return float(self.lineEdit_tau.text()) 
 
    def get_order(self):
        return int(self.lineEdit_order.text())
    
    def get_centering_ref(self):
        return self.comboBox_centering_ref.currentText()
    
    def set_centering_ref(self, centering):
        match = QtCore.Qt.MatchFixedString
        index = self.comboBox_centering_ref.findText(centering, match)
        self.comboBox_centering_ref.setCurrentIndex(index)
    
    def get_filter_ref_h(self):
        return int(self.lineEdit_filter_ref_h.text())
    
    def get_filter_ref_k(self):
        return int(self.lineEdit_filter_ref_k.text())

    def get_filter_ref_l(self):
        return int(self.lineEdit_filter_ref_l.text())
    
    def get_progress(self):
        return int(self.progressBar_ref.value())
    
    def set_progress(self, progress):
        self.progressBar_ref.setValue(progress)
        
    def get_run(self):
        return int(self.lineEdit_run.text())
        
    def set_run(self, batch):
        self.lineEdit_run.setText(str(batch))
     
    def set_chi_sq(self, chi_sq):
        self.lineEdit_chi_sq.setText('{:1.4e}'.format(chi_sq))
        
    def enable_refinement(self, visible):
        self.pushButton_run.setEnabled(visible)
        
    def enable_reset_refinement(self, visible):
        self.pushButton_reset_run.setEnabled(visible)
        
    def enable_continue_refinement(self, visible):
        self.pushButton_continue.setEnabled(visible)
        
    def get_runs(self):
        return int(self.lineEdit_runs.text())
    
    def set_runs(self, runs):
        self.lineEdit_runs.setText(str(runs))
        
    def get_cycles(self):
        return int(self.lineEdit_cycles.text())
    
    def set_cycles(self, cycles):
        self.lineEdit_cycles.setText(str(cycles))
    
    def clicked_run(self, slot):
        self.pushButton_run.clicked.connect(slot)
    
    def clicked_stop(self, slot):
        self.pushButton_stop.clicked.connect(slot)
        
    def clicked_reset(self, slot):
        self.pushButton_reset_run.clicked.connect(slot)
        
    def clicked_continue(self, slot):
        self.pushButton_continue.clicked.connect(slot)
        
    def get_filter_h(self):
        return float(self.lineEdit_filter_ref_h.text())
 
    def get_filter_k(self):
        return float(self.lineEdit_filter_ref_k.text())
    
    def get_filter_l(self):
        return float(self.lineEdit_filter_ref_l.text())
    
    def set_min_ref(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_min_ref.setText('{:1.4e}'.format(value))

    def set_max_ref(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_max_ref.setText('{:1.4e}'.format(value))
        
    def get_min_ref(self):
        return float(self.lineEdit_min_ref.text())

    def get_max_ref(self):
        return float(self.lineEdit_max_ref.text())
    
    def finished_editing_min_ref(self, slot):
        self.lineEdit_min_ref.editingFinished.connect(slot)

    def finished_editing_max_ref(self, slot):
        self.lineEdit_max_ref.editingFinished.connect(slot)
        
    def validate_min_ref(self):
        maximum = float(self.lineEdit_max_ref.text())
        validator = QtGui.QDoubleValidator(np.finfo(float).min, maximum, 4)
        self.lineEdit_min_ref.setValidator(validator)
        
    def validate_max_ref(self):
        minimum = float(self.lineEdit_min_ref.text())
        validator = QtGui.QDoubleValidator(minimum, np.finfo(float).max, 4)
        self.lineEdit_max_ref.setValidator(validator)
    
    def get_plot_ref(self):
        index = self.comboBox_plot_ref.currentIndex()    
        return self.comboBox_plot_ref.itemText(index)
            
    def get_norm_ref(self):
        index = self.comboBox_norm_ref.currentIndex()    
        return self.comboBox_norm_ref.itemText(index)
    
    def get_plot_ref_canvas(self):
        return self.canvas_ref
    
    def clear_plot_ref_canvas(self):
        self.canvas_ref.figure.clear()
        self.canvas_ref.draw()
        self.lineEdit_min_ref.setText('')       
        self.lineEdit_max_ref.setText('')    
        self.lineEdit_chi_sq.setText('') 
        
    def index_changed_plot_ref(self, slot):
        self.comboBox_plot_ref.currentIndexChanged.connect(slot)
        
    def index_changed_norm_ref(self, slot):
        self.comboBox_norm_ref.currentIndexChanged.connect(slot)
        
    def set_slice(self, value):
        self.lineEdit_slice.setText(str(value))
        
    def get_slice(self):
        text = self.lineEdit_slice.text()
        if (text != ''): return float(text) 
        
    def get_slice_hkl(self):
        index = self.comboBox_slice.currentIndex()    
        return self.comboBox_slice.itemText(index)
    
    def index_changed_slice_hkl(self, slot):
        self.comboBox_slice.currentIndexChanged.connect(slot)
        
    def finished_editing_slice(self, slot):
        self.lineEdit_slice.editingFinished.connect(slot)
        
    def get_plot_chi_sq_canvas(self):
        return self.canvas_chi_sq
    
    def clear_plot_chi_sq_canvas(self):
        self.canvas_chi_sq.figure.clear()
        self.canvas_chi_sq.draw()
        
    def get_plot_top_chi_sq(self,):
        return self.comboBox_plot_top_chi_sq.currentText()

    def get_plot_bottom_chi_sq(self):
        return self.comboBox_plot_bottom_chi_sq.currentText()
        
    def index_changed_plot_top_chi_sq(self, slot):
        self.comboBox_plot_top_chi_sq.currentIndexChanged.connect(slot)

    def index_changed_plot_bottom_chi_sq(self, slot):
        self.comboBox_plot_bottom_chi_sq.currentIndexChanged.connect(slot)
        
    def fixed_moment_check(self):
        return self.checkBox_fixed_moment.isChecked()

    def fixed_composition_check(self):
        return self.checkBox_fixed_composition.isChecked()
        
    def fixed_displacement_check(self):
        return self.checkBox_fixed_displacement.isChecked()
    
    def save_intensity_ref(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        return filename
            
    def save_chi_sq(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        return filename
               
    def button_clicked_save_intensity_ref(self, slot):
        self.pushButton_save_intensity_ref.clicked.connect(slot)
        
    def button_clicked_save_chi_sq(self, slot):
        self.pushButton_save_chi_sq.clicked.connect(slot)
        
    # ---
        
    def batch_checked_1d(self):
        return self.checkBox_batch_corr_1d.isChecked()
    
    def clicked_batch_1d(self, slot):
        self.checkBox_batch_corr_1d.stateChanged.connect(slot)
        
    def enable_runs_1d(self, visible):
        self.lineEdit_runs_corr_1d.setEnabled(visible)
        
    def set_runs_1d(self, runs):
        self.lineEdit_runs_corr_1d.setText(str(runs))
        
    def get_runs_1d(self):
        return int(self.lineEdit_runs_corr_1d.text())
        
    def batch_checked_3d(self):
        return self.checkBox_batch_corr_3d.isChecked()
    
    def clicked_batch_3d(self, slot):
        self.checkBox_batch_corr_3d.stateChanged.connect(slot)
        
    def enable_runs_3d(self, visible):
        self.lineEdit_runs_corr_3d.setEnabled(visible)
        
    def set_runs_3d(self, runs):
        self.lineEdit_runs_corr_3d.setText(str(runs))
        
    def get_runs_3d(self):
        return int(self.lineEdit_runs_corr_3d.text())
      
    def set_correlations_1d_type(self):
        selection = self.comboBox_correlations_1d.currentIndex()    
        data = self.comboBox_correlations_1d.itemData(selection)
        self.comboBox_plot_1d.clear()
        for t in data: self.comboBox_plot_1d.addItem(t)    

    def set_correlations_3d_type(self):
        selection = self.comboBox_correlations_3d.currentIndex()    
        data = self.comboBox_correlations_3d.itemData(selection)
        self.comboBox_plot_3d.clear()
        for t in data: self.comboBox_plot_3d.addItem(t)
        
    def index_changed_correlations_1d(self, slot):
        self.comboBox_correlations_1d.currentIndexChanged.connect(slot)
        
    def index_changed_correlations_3d(self, slot):
        self.comboBox_correlations_3d.currentIndexChanged.connect(slot)
        
    def get_correlations_1d(self):
        return self.comboBox_correlations_1d.currentText()
    
    def get_correlations_3d(self):
        return self.comboBox_correlations_3d.currentText()
    
    def get_plot_1d(self):
        return self.comboBox_plot_1d.currentText()
    
    def get_plot_3d(self):
        return self.comboBox_plot_3d.currentText()  
    
    def index_changed_plot_1d(self, slot):
        self.comboBox_plot_1d.currentIndexChanged.connect(slot)
        
    def index_changed_plot_3d(self, slot):
        self.comboBox_plot_3d.currentIndexChanged.connect(slot)

    def get_norm_1d(self):
        return self.comboBox_norm_1d.currentText()
    
    def get_norm_3d(self):
        return self.comboBox_norm_3d.currentText()
    
    def index_changed_norm_1d(self, slot):
        self.comboBox_norm_1d.currentIndexChanged.connect(slot)
        
    def index_changed_norm_3d(self, slot):
        self.comboBox_norm_3d.currentIndexChanged.connect(slot)

    def get_fract_1d(self):
        return float(self.lineEdit_fract_1d.text())
    
    def get_fract_3d(self):
        return float(self.lineEdit_fract_3d.text())
    
    def get_tol_1d(self):
        return float(self.lineEdit_tol_1d.text())
    
    def get_tol_3d(self):
        return float(self.lineEdit_tol_3d.text())
 
    def get_plot_1d_canvas(self):
        return self.canvas_1d
    
    def get_plot_3d_canvas(self):
        return self.canvas_3d
    
    def get_average_1d(self):
        return self.checkBox_average_1d.isChecked()
    
    def get_average_3d(self):
        return self.checkBox_average_3d.isChecked()

    def enable_calculate_1d(self, visible):
        self.pushButton_calculate_1d.setEnabled(visible)
        
    def enable_calculate_3d(self, visible):
        self.pushButton_calculate_3d.setEnabled(visible)
        
    def button_clicked_calculate_1d(self, slot):
        self.pushButton_calculate_1d.clicked.connect(slot)
        
    def button_clicked_calculate_3d(self, slot):
        self.pushButton_calculate_3d.clicked.connect(slot)
        
    def clear_plot_1d_canvas(self):
        self.canvas_1d.figure.clear()
        self.canvas_1d.draw()
        
    def clear_plot_3d_canvas(self):
        self.canvas_3d.figure.clear()
        self.canvas_3d.draw()
        
    def get_symmetrize(self):
        return self.comboBox_laue_corr.currentText()
        
    def set_symmetrize(self, laue):
        match = QtCore.Qt.MatchFixedString
        index = self.comboBox_laue_corr.findText(laue, match)
        self.comboBox_laue_corr.setCurrentIndex(index)

    def create_pairs_1d_table(self, n_pairs):
        self.tableWidget_pairs_1d.setRowCount(n_pairs)
        self.tableWidget_pairs_1d.setColumnCount(3)
        
        horiz_lbl = 'atom,pair,active'
        horiz_lbl = horiz_lbl.split(',')
        self.tableWidget_pairs_1d.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['{}'.format(s+1) for s in range(n_pairs)]
        self.tableWidget_pairs_1d.setVerticalHeaderLabels(vert_lbl)
        
    def create_pairs_3d_table(self, n_pairs):
        self.tableWidget_pairs_3d.setRowCount(n_pairs)
        self.tableWidget_pairs_3d.setColumnCount(3)
        
        horiz_lbl = 'atom,pair,active'
        horiz_lbl = horiz_lbl.split(',')
        self.tableWidget_pairs_3d.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['{}'.format(s+1) for s in range(n_pairs)]
        self.tableWidget_pairs_3d.setVerticalHeaderLabels(vert_lbl)
     
    def clear_pairs_1d_table(self):
        self.tableWidget_pairs_1d.clearContents()
        self.tableWidget_pairs_1d.setRowCount(0)
        self.tableWidget_pairs_1d.setColumnCount(0)
        
    def clear_pairs_3d_table(self):
        self.tableWidget_pairs_3d.clearContents()
        self.tableWidget_pairs_3d.setRowCount(0)
        self.tableWidget_pairs_3d.setColumnCount(0)
        
    def format_pairs_1d_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        
        self.tableWidget_pairs_1d.setSpan(0, 0, 1, 2)
        self.tableWidget_pairs_1d.item(0, 0).setTextAlignment(alignment)
        self.tableWidget_pairs_1d.item(0, 0).setFlags(flags)
        
        for i in range(1,self.tableWidget_pairs_1d.rowCount()):
            for j in range(self.tableWidget_pairs_1d.columnCount()):
                item = self.tableWidget_pairs_1d.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    item.setFlags(flags)
                                
        horiz_hdr = self.tableWidget_pairs_1d.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
            
    def format_pairs_3d_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        
        self.tableWidget_pairs_3d.setSpan(0, 0, 1, 2)
        self.tableWidget_pairs_3d.item(0, 0).setTextAlignment(alignment)
        self.tableWidget_pairs_3d.item(0, 0).setFlags(flags)
                                       
        for i in range(1,self.tableWidget_pairs_3d.rowCount()):
            for j in range(self.tableWidget_pairs_3d.columnCount()):
                item = self.tableWidget_pairs_3d.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    item.setFlags(flags)
                                
        horiz_hdr = self.tableWidget_pairs_3d.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        
    def clear_pairs_1d_table_table(self):
        self.tableWidget_pairs_1d.clearContents()
        self.tableWidget_pairs_1d.setRowCount(0)
        self.tableWidget_pairs_1d.setColumnCount(0) 
        
    def clear_pairs_3d_table_table(self):
        self.tableWidget_pairs_3d.clearContents()
        self.tableWidget_pairs_3d.setRowCount(0)
        self.tableWidget_pairs_3d.setColumnCount(0) 
                
    def get_pairs_1d_table_row_count(self):
        return self.tableWidget_pairs_1d.rowCount()
                
    def get_pairs_3d_table_row_count(self):
        return self.tableWidget_pairs_3d.rowCount()
 
    def get_pairs_1d_table_col_count(self):
        return self.tableWidget_pairs_1d.columnCount()
    
    def get_pairs_3d_table_col_count(self):
        return self.tableWidget_pairs_3d.columnCount()
 
    def get_pairs_1d_table_row(self, i):
        atom = self.tableWidget_pairs_1d.item(i, 0)
        pair = self.tableWidget_pairs_1d.item(i, 1)
        active = self.tableWidget_pairs_1d.cellWidget(i, 2).isChecked()
        if (atom is not None): atom = atom.text()
        if (pair is not None): pair = pair.text()
        return atom, pair, active
    
    def get_pairs_3d_table_row(self, i):
        atom = self.tableWidget_pairs_3d.item(i, 0)
        pair = self.tableWidget_pairs_3d.item(i, 1)
        active = self.tableWidget_pairs_3d.cellWidget(i, 2).isChecked()
        if (atom is not None): atom = atom.text()
        if (pair is not None): pair = pair.text()
        return atom, pair, active
 
    def set_pairs_1d_table_row(self, data, i):
        item = QtWidgets.QTableWidgetItem(data[0])
        self.tableWidget_pairs_1d.setItem(i, 0, item)
        item = QtWidgets.QTableWidgetItem(data[1])
        self.tableWidget_pairs_1d.setItem(i, 1, item)        
        check = QtWidgets.QCheckBox()
        check.setObjectName('checkBox_pairs_1d_'+str(i))
        check.setCheckState(QtCore.Qt.Checked) 
        self.tableWidget_pairs_1d.setCellWidget(i, 2, check)
    
    def set_pairs_3d_table_row(self, data, i):
        item = QtWidgets.QTableWidgetItem(data[0])
        self.tableWidget_pairs_3d.setItem(i, 0, item)
        item = QtWidgets.QTableWidgetItem(data[1])
        self.tableWidget_pairs_3d.setItem(i, 1, item)        
        check = QtWidgets.QCheckBox()
        check.setObjectName('checkBox_pairs_3d_'+str(i))
        check.setCheckState(QtCore.Qt.Checked) 
        self.tableWidget_pairs_3d.setCellWidget(i, 2, check)
 
    def enable_pairs_1d(self, state):  
        visible = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        for i in range(self.tableWidget_pairs_1d.rowCount()):
            check = self.tableWidget_pairs_1d.cellWidget(i, 2)
            check.setCheckState(QtCore.Qt.Checked) 
            check.setEnabled(visible)
            
    def enable_pairs_3d(self, state):  
        visible = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        for i in range(self.tableWidget_pairs_3d.rowCount()):
            check = self.tableWidget_pairs_3d.cellWidget(i, 2)
            check.setCheckState(QtCore.Qt.Checked) 
            check.setEnabled(visible)
    
    def check_clicked_pairs_1d(self, slot):
        for i in range(self.tableWidget_pairs_1d.rowCount()):
            check = self.tableWidget_pairs_1d.cellWidget(i, 2)
            check.clicked.connect(slot)
            
    def check_clicked_pairs_3d(self, slot):
        for i in range(self.tableWidget_pairs_3d.rowCount()):
            check = self.tableWidget_pairs_3d.cellWidget(i, 2)
            check.clicked.connect(slot)
 
    def average_1d_checked(self):
        return self.checkBox_average_1d.isChecked()
    
    def average_3d_checked(self):
        return self.checkBox_average_3d.isChecked()
    
    def get_h(self):
        return int(self.lineEdit_plane_h.text())
    
    def get_k(self):
        return int(self.lineEdit_plane_k.text())
    
    def get_l(self):    
        return int(self.lineEdit_plane_l.text())
    
    def get_d(self):
        return float(self.lineEdit_plane_d.text())
    
    def set_h(self, h):
        self.lineEdit_plane_h.setText(str(h))
        
    def set_k(self, k):
        self.lineEdit_plane_k.setText(str(k))
        
    def set_l(self, l):
        self.lineEdit_plane_l.setText(str(l))
        
    def set_d(self, d):
        self.lineEdit_plane_d.setText(str(d))
    
    def finished_editing_h(self, slot):
        self.lineEdit_plane_h.editingFinished.connect(slot)
            
    def finished_editing_k(self, slot):
        self.lineEdit_plane_k.editingFinished.connect(slot)
            
    def finished_editing_l(self, slot):
        self.lineEdit_plane_l.editingFinished.connect(slot)
            
    def finished_editing_d(self, slot):
        self.lineEdit_plane_d.editingFinished.connect(slot)
        
    def save_correlations_1d(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        return filename
            
    def save_correlations_3d(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        return filename
            
    def button_clicked_save_1d(self, slot):
        self.pushButton_save_1d.clicked.connect(slot)
        
    def button_clicked_save_3d(self, slot):
        self.pushButton_save_3d.clicked.connect(slot)
               
    # ---
    
    def enable_disorder_mag_recalc(self, visible):
        self.checkBox_mag_recalc.setEnabled(visible)
        
    def enable_disorder_occ_recalc(self, visible):
        self.checkBox_occ_recalc.setEnabled(visible)
        
    def enable_disorder_dis_recalc(self, visible):
        self.checkBox_dis_recalc.setEnabled(visible)
        
    def enable_disorder_struct_recalc(self, visible):
        self.checkBox_struct_recalc.setEnabled(visible)
        
    def clicked_disorder_mag_recalc(self, slot):
        self.checkBox_mag_recalc.clicked.connect(slot)

    def clicked_disorder_occ_recalc(self, slot):
        self.checkBox_occ_recalc.clicked.connect(slot)
        
    def clicked_disorder_dis_recalc(self, slot):
        self.checkBox_dis_recalc.clicked.connect(slot)
        
    def clicked_disorder_struct_recalc(self, slot):
        self.checkBox_struct_recalc.clicked.connect(slot)
        
    def set_disorder_mag_recalc(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_mag_recalc.setCheckState(check)
        
    def set_disorder_occ_recalc(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_occ_recalc.setCheckState(check)
        
    def set_disorder_dis_recalc(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_dis_recalc.setCheckState(check)
        
    def set_disorder_struct_recalc(self, state):
        check = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.checkBox_struct_recalc.setCheckState(check)

    def get_disorder_mag_recalc(self):
        return self.checkBox_mag_recalc.isChecked()
    
    def get_disorder_occ_recalc(self):
        return self.checkBox_occ_recalc.isChecked()
    
    def get_disorder_dis_recalc(self):
        return self.checkBox_dis_recalc.isChecked()
    
    def get_disorder_struct_recalc(self):
        return self.checkBox_struct_recalc.isChecked()
        
    def create_recalculation_table(self, dh, nh, min_h, max_h, 
                                         dk, nk, min_k, max_k, 
                                         dl, nl, min_l, max_l):
        
        self.tableWidget_calc.setRowCount(3)
        self.tableWidget_calc.setColumnCount(5)
        
        data = [[dh, nh, min_h, max_h, 0.0], 
                [dk, nk, min_k, max_k, 0.0], 
                [dl, nl, min_l, max_l, 0.0]]
        
        for i in range(self.tableWidget_calc.rowCount()):
            for j in range(self.tableWidget_calc.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(data[i][j]))
                self.tableWidget_calc.setItem(i, j, item)
        
        lbl = 'step,size,min,max,filter'    
        lbl = lbl.split(',')
        self.tableWidget_calc.setHorizontalHeaderLabels(lbl)
        
        lbl = 'h,k,l'
        lbl = lbl.split(',')
        self.tableWidget_calc.setVerticalHeaderLabels(lbl)
        
    def clear_recalculation_table(self):
        try: self.tableWidget_calc.disconnect() 
        except Exception: pass  
        self.tableWidget_calc.clearContents()
        self.tableWidget_calc.setRowCount(0)
        self.tableWidget_calc.setColumnCount(0)
        
    def format_recalculation_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        stretch = QtWidgets.QHeaderView.Stretch
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        for i in range(self.tableWidget_calc.rowCount()):
            for j in range(self.tableWidget_calc.columnCount()):
                item = self.tableWidget_calc.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    if (j == 0): item.setFlags(flags)

        delegate = SizeIntDelegate(self.tableWidget_calc)
        self.tableWidget_calc.setItemDelegateForColumn(1, delegate)
        delegate = StandardDoubleDelegate(self.tableWidget_calc)
        self.tableWidget_calc.setItemDelegateForColumn(2, delegate)
        self.tableWidget_calc.setItemDelegateForColumn(3, delegate)
        delegate = PositiveDoubleDelegate(self.tableWidget_calc)
        self.tableWidget_calc.setItemDelegateForColumn(4, delegate)
        
        horiz_hdr = self.tableWidget_calc.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        
    def get_recalculation_binning_h(self):       
        i = 0
        text = self.tableWidget_calc.item(i, 0).text()
        step = 0 if text == '-' else float(text)
        size = int(self.tableWidget_calc.item(i, 1).text()) 
        mininum = float(self.tableWidget_calc.item(i, 2).text())
        maximum = float(self.tableWidget_calc.item(i, 3).text())
        return step, size, mininum, maximum
    
    def get_recalculation_binning_k(self):       
        i = 1
        text = self.tableWidget_calc.item(i, 0).text()
        step = 0 if text == '-' else float(text)
        size = int(self.tableWidget_calc.item(i, 1).text()) 
        mininum = float(self.tableWidget_calc.item(i, 2).text())
        maximum = float(self.tableWidget_calc.item(i, 3).text())
        return step, size, mininum, maximum
    
    def get_recalculation_binning_l(self):       
        i = 2
        text = self.tableWidget_calc.item(i, 0).text()
        step = 0 if text == '-' else float(text)
        size = int(self.tableWidget_calc.item(i, 1).text()) 
        mininum = float(self.tableWidget_calc.item(i, 2).text())
        maximum = float(self.tableWidget_calc.item(i, 3).text())
        return step, size, mininum, maximum
    
    def set_recalculation_binning_h(self, size, minimum, maximum):
        i = 0
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_calc.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_calc.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_calc.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_calc.setItem(i, 3, item)
        
    def set_recalculation_binning_k(self, size, minimum, maximum):
        i = 1
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_calc.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_calc.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_calc.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_calc.setItem(i, 3, item)
        
    def set_recalculation_binning_l(self, size, minimum, maximum):
        i = 2
        text = '-' if size <= 1 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_calc.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_calc.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_calc.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_calc.setItem(i, 3, item)
        
    def get_recalculation_filter(self):       
        sigma_h = float(self.tableWidget_calc.item(0, 4).text())
        sigma_k = float(self.tableWidget_calc.item(1, 4).text())
        sigma_l = float(self.tableWidget_calc.item(2, 4).text())
        return sigma_h, sigma_k, sigma_l
        
    def block_recalculation_table_signals(self):
        self.tableWidget_calc.blockSignals(True)

    def unblock_recalculation_table_signals(self):
        self.tableWidget_calc.blockSignals(False)
        
    def item_changed_recalculation_table(self, slot):
        self.tableWidget_calc.itemChanged.connect(slot)
        
    def get_recalculation_table_row_count(self):
        return self.tableWidget_calc.rowCount()
    
    def get_unit_recalculation_col_count(self):
        return self.tableWidget_calc.columnCount()
    
    def batch_checked_calc(self):
        return self.checkBox_batch_calc.isChecked()
    
    def clicked_batch_calc(self, slot):
        self.checkBox_batch_calc.stateChanged.connect(slot)
        
    def enable_runs_calc(self, visible):
        self.lineEdit_runs_calc.setEnabled(visible)
        
    def set_runs_calc(self, runs):
        self.lineEdit_runs_calc.setText(str(runs))
        
    def get_runs_calc(self):
        return int(self.lineEdit_runs_calc.text())
 
    def get_order_calc(self):
        return int(self.lineEdit_order_calc.text())
    
    def get_laue(self):
        return self.comboBox_laue.currentText()
    
    def set_laue(self, laue):
        index = self.comboBox_laue.findText(laue, QtCore.Qt.MatchFixedString)
        self.comboBox_laue.setCurrentIndex(index)
        
    def get_norm_calc(self):
        index = self.comboBox_norm_calc.currentIndex()    
        return self.comboBox_norm_calc.itemText(index)
        
    def index_changed_norm_calc(self, slot):
        self.comboBox_norm_calc.currentIndexChanged.connect(slot)
    
    def set_min_calc(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_min_calc.setText('{:1.4e}'.format(value))

    def set_max_calc(self, value):
        if str(value) == '--': value = 0
        self.lineEdit_max_calc.setText('{:1.4e}'.format(value))
        
    def get_min_calc(self):
        return float(self.lineEdit_min_calc.text())

    def get_max_calc(self):
        return float(self.lineEdit_max_calc.text())
    
    def finished_editing_min_calc(self, slot):
        self.lineEdit_min_calc.editingFinished.connect(slot)

    def finished_editing_max_calc(self, slot):
        self.lineEdit_max_calc.editingFinished.connect(slot)
    
    def validate_min_calc(self):
        maximum = float(self.lineEdit_max_exp.text())
        validator = QtGui.QDoubleValidator(np.finfo(float).min, maximum, 4)
        self.lineEdit_min_calc.setValidator(validator)
        
    def validate_max_calc(self):
        minimum = float(self.lineEdit_min_exp.text())
        validator = QtGui.QDoubleValidator(minimum, np.finfo(float).max, 4)
        self.lineEdit_max_calc.setValidator(validator)
    
    def get_axes(self):
        return self.comboBox_axes.currentText()
    
    def get_plot_calc_canvas(self):
        return self.canvas_calc
    
    def clear_canvas_calc_canvas(self):
        self.canvas_calc.figure.clear()
        self.canvas_calc.draw()
        self.lineEdit_min_calc.setText('')
        self.lineEdit_max_calc.setText('')
    
    def button_clicked_calc(self, slot):
        self.pushButton_calc.clicked.connect(slot)
        
    def enable_recalculation(self, visible):
        self.pushButton_calc.setEnabled(visible)
        
    def get_type_recalc(self):
        return self.lineEdit_type.text()
    
    def set_type_recalc(self, value):
        self.lineEdit_type.setText(str(value))
        
    def set_slice_calc(self, value):
        self.lineEdit_slice_calc.setText(str(value))
    
    def get_slice_calc(self):
        text = self.lineEdit_slice_calc.text()
        if (text != ''): return float(text) 
        
    def get_slice_hkl_calc(self):
        index = self.comboBox_slice_calc.currentIndex()    
        return self.comboBox_slice_calc.itemText(index)
        
    def index_changed_slice_hkl_calc(self, slot):
        self.comboBox_slice_calc.currentIndexChanged.connect(slot)
        
    def finished_editing_slice_calc(self, slot):
        self.lineEdit_slice_calc.editingFinished.connect(slot)
        
    def get_centering_calc(self):
        return self.comboBox_centering_calc.currentText()
    
    def set_centering_calc(self, centering):
        match = QtCore.Qt.MatchFixedString
        index = self.comboBox_centering_calc.findText(centering, match)
        self.comboBox_centering_calc.setCurrentIndex(index)
    
    def clear_atom_site_recalculation_table(self):
        self.tableWidget_recalc.clearContents()
        self.tableWidget_recalc.setRowCount(0)
        self.tableWidget_recalc.setColumnCount(0)
        
    def format_atom_site_recalculation_table(self):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        stretch = QtWidgets.QHeaderView.Stretch
        resize = QtWidgets.QHeaderView.ResizeToContents

        for i in range(self.tableWidget_recalc.rowCount()):
            for j in range(self.tableWidget_recalc.columnCount()-1):
                item = self.tableWidget_recalc.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    item.setFlags(flags)
                    
        horiz_hdr = self.tableWidget_recalc.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        horiz_hdr.setSectionResizeMode(1, resize)
    
    def create_atom_site_recalculation_table(self, atom, occupancy, Uiso, mu):
        alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        stretch = QtWidgets.QHeaderView.Stretch
        resize = QtWidgets.QHeaderView.ResizeToContents
                    
        n_site = len(atom)
        
        self.tableWidget_recalc.setRowCount(n_site)
        self.tableWidget_recalc.setColumnCount(5)
        
        data = [atom, occupancy, Uiso, mu]
        
        for i in range(self.tableWidget_recalc.rowCount()):
            for j in range(self.tableWidget_recalc.columnCount()-1):
                item = QtWidgets.QTableWidgetItem(str(data[j][i]))
                self.tableWidget_recalc.setItem(i, j, item)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
                    item.setFlags(flags)
            check = QtWidgets.QCheckBox()
            check.setObjectName('checkBox_atom_site_recalculation_'+str(i))
            check.setCheckState(QtCore.Qt.Checked) 
            self.tableWidget_recalc.setCellWidget(i, 4, check)
                    
        horiz_lbl = 'atom,occupancy,Uiso,mu,active'    
        horiz_lbl = horiz_lbl.split(',')
        self.tableWidget_recalc.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['{}'.format(s+1) for s in range(n_site)]
        self.tableWidget_recalc.setVerticalHeaderLabels(vert_lbl)
        
        horiz_hdr = self.tableWidget_recalc.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        horiz_hdr.setSectionResizeMode(1, resize)
        
    def get_atom_site_recalculation_row_count(self):
        return self.tableWidget_recalc.rowCount()
 
    def get_active_atom_site(self):
        data = []
        for i in range(self.tableWidget_recalc.rowCount()):
            active = self.tableWidget_recalc.cellWidget(i, 4).isChecked()
            data.append(active)
        return np.array(data)
    
    def save_intensity_calc(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder, 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        return filename
            
    def button_clicked_save_calc(self, slot):
        self.pushButton_save_calc.clicked.connect(slot)
        
    # ---
    
    def save_CIF(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
                      
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder,
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options)
        return filename
    
    def save_correlations_CSV(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
                      
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder,
                                                        'CSV files *.csv',
                                                        options=options)
        return filename
    
    def save_correlations_VTK(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
                      
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder,
                                                        'VTK files *.vtm',
                                                        options=options)
        return filename
    
    def save_VTK(self, folder='.'):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
                      
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', 
                                                        folder,
                                                        'VTK files *.vtm',
                                                        options=options)
        return filename        
    
    def button_clicked_save_CIF(self, slot):
        self.pushButton_save_CIF.clicked.connect(slot)
    
    def button_clicked_save_dis_CIF(self, slot):
        self.pushButton_save_CIF_dis.clicked.connect(slot)
 
    def button_clicked_save_CSV(self, slot):
        self.pushButton_save_CSV_correlations.clicked.connect(slot) 

    def button_clicked_save_VTK(self, slot):
        self.pushButton_save_VTK_correlations.clicked.connect(slot)
        
    def button_clicked_save_recalc_VTK(self, slot):
        self.pushButton_save_VTK.clicked.connect(slot)