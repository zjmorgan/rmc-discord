#!/ur/bin/env/python3

from PyQt5 import QtWidgets, QtGui, QtCore, uic

import os
import sys

import numpy as np

from disorder.graphical.utilities import FractionalDelegate
from disorder.graphical.utilities import StandardDoubleDelegate
from disorder.graphical.utilities import SizeIntDelegate

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
        
        self.lineEdit_lat.setEnabled(False)
     
        self.lineEdit_n_atm.setEnabled(False)
        self.lineEdit_n.setEnabled(False)
        self.lineEdit_space_group.setEnabled(False)
        self.lineEdit_space_group_hm.setEnabled(False)
        
        self.lineEdit_nu.setValidator(QtGui.QIntValidator(1, 32))
        self.lineEdit_nv.setValidator(QtGui.QIntValidator(1, 32))
        self.lineEdit_nw.setValidator(QtGui.QIntValidator(1, 32))
        
        self.comboBox_centering.addItem('P')
        self.comboBox_centering.addItem('I')
        self.comboBox_centering.addItem('F')
        self.comboBox_centering.addItem('A')
        self.comboBox_centering.addItem('B')
        self.comboBox_centering.addItem('C')
        self.comboBox_centering.addItem('R')
        
        self.comboBox_punch.addItem('Box')
        self.comboBox_punch.addItem('Ellipsoid')
        
        self.comboBox_plot_exp.addItem('Intensity')
        self.comboBox_plot_exp.addItem('Error')

        self.comboBox_norm_exp.addItem('Linear')
        self.comboBox_norm_exp.addItem('Logarithmic')
        
        self.clear_application()
        
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
        
    def new_triggered(self, slot): 
        self.actionNew.triggered.connect(slot)    
        
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
        
        self.tableWidget_CIF.setRowCount(0)
        self.tableWidget_CIF.setColumnCount(0)
        
        self.tableWidget_atm.setRowCount(0)
        self.tableWidget_atm.setColumnCount(0)
        
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

        self.tableWidget_exp.setRowCount(0)
        self.tableWidget_exp.setColumnCount(0)
                
        self.lineEdit_cycles.setText('10')
       
        self.lineEdit_filter_ref_h.setText('0.0')
        self.lineEdit_filter_ref_k.setText('0.0')
        self.lineEdit_filter_ref_l.setText('0.0')

        self.lineEdit_runs.setText('1')
        self.lineEdit_runs.setEnabled(False)
        self.checkBox_batch.setChecked(False)
        
        self.checkBox_fixed_moment.setChecked(True)
        self.checkBox_fixed_composition.setChecked(True)
        self.checkBox_fixed_displacement.setChecked(True)
        
        self.lineEdit_order.setText('2')
       
        self.lineEdit_slice.setText('0.0')
        
        self.lineEdit_prefactor.setText('%1.2e' % 1e+4)
        self.lineEdit_tau.setText('%1.2e' % 1e-3)
        
        self.lineEdit_min_ref.setText('')       
        self.lineEdit_max_ref.setText('')    

        self.lineEdit_chi_sq.setText('')    
                        
        self.tableWidget_pairs_1d.setRowCount(0)
        self.tableWidget_pairs_1d.setColumnCount(0)
        
        self.tableWidget_pairs_3d.setRowCount(0)
        self.tableWidget_pairs_3d.setColumnCount(0)
                        
        self.tableWidget_calc.setRowCount(0)
        self.tableWidget_calc.setColumnCount(0)
        
        self.lineEdit_order_calc.setText('2')
       
        self.lineEdit_slice_calc.setText('0.0')
        
        self.lineEdit_min_calc.setText('')       
        self.lineEdit_max_calc.setText('')
        
    def save_as_triggered(self, slot):
        self.actionSave_As.triggered.connect(slot)

    def save_triggered(self, slot):
        self.actionSave.triggered.connect(slot)
        
    def open_dialog_save(self):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file',
                                                        '.', 
                                                        'ini files *.ini',
                                                        options=options)   
        
        return filename
        
    def save_widgets(self, filename):
        settings = QtCore.QSettings(filename, QtCore.QSettings.IniFormat)
        save_gui(self, settings)

    def open_triggered(self, slot):
        self.actionOpen.triggered.connect(slot)
        
    def open_dialog_load(self):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file',
                                                        '.', 
                                                        'ini files *.ini',
                                                        options=options)   

        return filename
        
    def load_widgets(self, filename):
        settings = QtCore.QSettings(filename, QtCore.QSettings.IniFormat)
        load_gui(self, settings)

    def exit_triggered(self, slot):
        self.actionExit.triggered.connect(slot)

    def close_application(self):
        choice = QtWidgets.QMessageBox.question(self, 
                                               'Quit?', 
                                               'Are you sure?',
                                               QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.Yes)
        
        return choice == QtWidgets.QMessageBox.Yes
        
    def get_every_site(self):
        return self.get_every_unit_cell_table_col(self.unit_table['site'])
    
    def get_every_atom(self):
        return self.get_every_unit_cell_table_col(self.unit_table['atom'])
    
    def get_every_isotope(self):
        return self.get_every_unit_cell_table_col(self.unit_table['isotope'])
    
    def get_every_ion(self):
        return self.get_every_unit_cell_table_col(self.unit_table['ion'])

    def get_every_occupancy(self):
        return self.get_every_unit_cell_table_col(self.unit_table['occupancy'])

    def get_every_Uiso(self):
        return self.get_every_unit_cell_table_col(self.unit_table['Uiso'])
    
    def get_every_U11(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U11'])
    
    def get_every_U22(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U22'])
    
    def get_every_U33(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U33'])
    
    def get_every_U23(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U23'])
    
    def get_every_U13(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U13'])
    
    def get_every_U12(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U12'])
    
    def get_every_U1(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U1'])
    
    def get_every_U2(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U2'])
    
    def get_every_U3(self):
        return self.get_every_unit_cell_table_col(self.unit_table['U3'])

    def get_every_mu(self):
        return self.get_every_unit_cell_table_col(self.unit_table['mu'])
    
    def get_every_mu1(self):
        return self.get_every_unit_cell_table_col(self.unit_table['mu1'])
    
    def get_every_mu2(self):
        return self.get_every_unit_cell_table_col(self.unit_table['mu2'])
    
    def get_every_mu3(self):
        return self.get_every_unit_cell_table_col(self.unit_table['mu3'])
   
    def get_every_g(self):
        return self.get_every_unit_cell_table_col(self.unit_table['g'])
    
    def get_every_u(self):
        return self.get_every_unit_cell_table_col(self.unit_table['u'])
    
    def get_every_v(self):
        return self.get_every_unit_cell_table_col(self.unit_table['v'])
    
    def get_every_w(self):
        return self.get_every_unit_cell_table_col(self.unit_table['w'])
    
    def get_every_operator(self):
        return self.get_every_unit_cell_table_col(self.unit_table['operator'])

    def get_every_magnetic_operator(self):
        return self.get_every_unit_cell_table_col(self.unit_table['moment'])    
    
    # ---

    def get_site(self):
        return self.get_unit_cell_table_col(self.unit_table['site'])

    def get_atom(self):
        return self.get_unit_cell_table_col(self.unit_table['atom'])
    
    def get_isotope(self):
        return self.get_unit_cell_table_col(self.unit_table['isotope'])
    
    def get_ion(self):
        return self.get_unit_cell_table_col(self.unit_table['ion'])

    def get_occupancy(self):
        return self.get_unit_cell_table_col(self.unit_table['occupancy'])

    def get_Uiso(self):
        return self.get_unit_cell_table_col(self.unit_table['Uiso'])
    
    def get_U11(self):
        return self.get_unit_cell_table_col(self.unit_table['U11'])
    
    def get_U22(self):
        return self.get_unit_cell_table_col(self.unit_table['U22'])
    
    def get_U33(self):
        return self.get_unit_cell_table_col(self.unit_table['U33'])
    
    def get_U23(self):
        return self.get_unit_cell_table_col(self.unit_table['U23'])
    
    def get_U13(self):
        return self.get_unit_cell_table_col(self.unit_table['U13'])
    
    def get_U12(self):
        return self.get_unit_cell_table_col(self.unit_table['U12'])
    
    def get_U1(self):
        return self.get_unit_cell_table_col(self.unit_table['U1'])
    
    def get_U2(self):
        return self.get_unit_cell_table_col(self.unit_table['U2'])
    
    def get_U3(self):
        return self.get_unit_cell_table_col(self.unit_table['U3'])

    def get_mu(self):
        return self.get_unit_cell_table_col(self.unit_table['mu'])
    
    def get_mu1(self):
        return self.get_unit_cell_table_col(self.unit_table['mu1'])
    
    def get_mu2(self):
        return self.get_unit_cell_table_col(self.unit_table['mu2'])
    
    def get_mu3(self):
        return self.get_unit_cell_table_col(self.unit_table['mu3'])
   
    def get_g(self):
        return self.get_unit_cell_table_col(self.unit_table['g'])
    
    def get_u(self):
        return self.get_unit_cell_table_col(self.unit_table['u'])
    
    def get_v(self):
        return self.get_unit_cell_table_col(self.unit_table['v'])
    
    def get_w(self):
        return self.get_unit_cell_table_col(self.unit_table['w'])
        
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
        self.lineEdit_a.setText(str(a))
        self.lineEdit_b.setText(str(b))
        self.lineEdit_c.setText(str(c))
        self.lineEdit_alpha.setText(str(np.rad2deg(alpha)))
        self.lineEdit_beta.setText(str(np.rad2deg(beta)))
        self.lineEdit_gamma.setText(str(np.rad2deg(gamma)))
        
    def set_lattice(self, lat):
        self.lineEdit_lat.setText(lat)
        
    def set_space_group(self, group, hm):
        self.lineEdit_space_group.setText(str(group))
        self.lineEdit_space_group_hm.setText(hm)
        
    def open_dialog_cif(self):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file', 
                                                        '.', 
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options)
        
        return filename
    
    def button_clicked_CIF(self, slot):
        self.pushButton_load_CIF.clicked.connect(slot)
        
    def get_every_unit_cell_table_col(self, j):       
        data = []
        for i in range(self.tableWidget_CIF.rowCount()):
            item = self.tableWidget_CIF.item(i, j)
            if (item is not None):
                data.append(str(item.text()))
            else:
                data.append(str(''))
        return np.array(data)
                
    def get_unit_cell_table_col(self, j):
        data = []
        for i in range(self.tableWidget_CIF.rowCount()):
            if (not self.tableWidget_CIF.isRowHidden(i)):
                data.append(str(self.tableWidget_CIF.item(i, j).text()))
        return np.array(data)
    
    def get_atom_site_table_col(self, j):
        data = []
        for i in range(self.tableWidget_atm.rowCount()):
            if (not self.tableWidget_atm.isRowHidden(i)):
                data.append(str(self.tableWidget_atm.item(i, j).text()))
        return np.array(data)
    
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
        
    def create_experiment_table(self):
        self.tableWidget_exp.setRowCount(3)
        self.tableWidget_exp.setColumnCount(4)
        
        horiz_lbl = ['step','size','min','max']
        self.tableWidget_exp.setHorizontalHeaderLabels(horiz_lbl)
        
        vert_lbl = ['h','k','l']
        self.tableWidget_exp.setVerticalHeaderLabels(vert_lbl)
        
    def format_experimet_table(self):
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
        text = '-' if size == 0 else np.round((maximum-minimum)/(size-1),4)
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
        text = '-' if size == 0 else np.round((maximum-minimum)/(size-1),4)
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
        text = '-' if size == 0 else np.round((maximum-minimum)/(size-1),4)
        item = QtWidgets.QTableWidgetItem(str(text))
        self.tableWidget_exp.setItem(i, 0, item)
        
        item = QtWidgets.QTableWidgetItem(str(size))
        self.tableWidget_exp.setItem(i, 1, item)   
        item = QtWidgets.QTableWidgetItem(str(minimum))
        self.tableWidget_exp.setItem(i, 2, item)  
        item = QtWidgets.QTableWidgetItem(str(maximum))
        self.tableWidget_exp.setItem(i, 3, item)  
    
    def set_experiment_table_item(self, i, j, value):
        item = QtWidgets.QTableWidgetItem(str(value))
        self.tableWidget_exp.setItem(i, j, item)
        
    def item_changed_experiment_table(self, slot):
        self.tableWidget_exp.itemChanged.connect(slot)
        
    def block_experiment_table_signals(self):
        self.tableWidget_exp.blockSignals(True)

    def unblock_experiment_table_signals(self):
        self.tableWidget_exp.blockSignals(False)
        
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
        if (text != ''):
            return int(text)
    
    def get_rebin_combo_k(self):
        text = self.comboBox_rebin_k.currentText().split(':')[-1]
        if (text != ''):
            return int(text)  
        
    def get_rebin_combo_l(self):
        text = self.comboBox_rebin_l.currentText().split(':')[-1]
        if (text != ''):
            return int(text)
            
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
        self.lineEdit_min_exp.setText('{:1.4e}'.format(value))
 
    def set_max_exp(self, value):
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
        return self.canvas_exp
    
    def index_changed_plot_exp(self, slot):
        self.comboBox_norm_exp.currentIndexChanged.connect(slot)
        
    def index_changed_norm_exp(self, slot):
        self.comboBox_norm_exp.currentIndexChanged.connect(slot)
    
    def open_dialog_nxs(self):
        options = QtWidgets.QFileDialog.Option()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
            
        filename, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file', 
                                                        '.', 
                                                        'NeXus files *.nxs',
                                                        options=options)
        
        return filename            
    
    def button_clicked_NXS(self, slot):
        self.pushButton_load_NXS.clicked.connect(slot)