#!/ur/bin/env/python3

from PyQt5 import QtWidgets, QtGui, QtCore, uic

import os
import sys

import numpy as np

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
        
        self.lineEdit_nu.setText('1')
        self.lineEdit_nv.setText('1')
        self.lineEdit_nw.setText('1')
        
        self.unit_table = {'site': 0, 'atom': 1, 'isotope': 2, 'ion': 3,
                           'occupancy': 4, 'Uiso': 5,
                           'U11': 6, 'U22': 7, 'U33': 8,
                           'U23': 9, 'U13': 10, 'U12': 11,
                           'mu': 12, 'mu1': 13, 'mu2': 14, 'mu3': 15, 'g': 16,
                           'u': 17, 'v': 18, 'w': 19}
        
        self.atom_table = {'atom': 0, 'ion': 1, 'occupancy': 2, 'Uiso': 3,
                           'U11': 4, 'U22': 5, 'U33': 6, 
                           'U23': 7, 'U13': 8, 'U12': 9,
                           'mu': 10, 'mu1': 11, 'mu2': 12, 'mu3': 13, 'g': 14,
                           'u': 15, 'v': 16, 'w': 17, '': 18}
        
    def get_every_site(self):
        j = self.unit_table['site']
        data = []
        for i in range(self.tableWidget_CIF.rowCount()):
            data.append(str(self.tableWidget_CIF.item(i, j).text()))
        return np.array(data)
    
    def get_site(self):
        return self.get_unit_cell_table_col(self.unit_table['site'])

    def get_atm(self):
        return self.get_unit_cell_table_col(self.unit_table['atm'])
    
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
               float(self.lineEdit_alpha.text()), \
               float(self.lineEdit_beta.text()), \
               float(self.lineEdit_gamma.text())
        
    def set_lattice_parameters(self, a, b, c, alpha, beta, gamma):
        self.lineEdit_a.setText(str(a))
        self.lineEdit_b.setText(str(b))
        self.lineEdit_c.setText(str(c))
        self.lineEdit_alpha.setText(str(alpha))
        self.lineEdit_beta.setText(str(beta))
        self.lineEdit_gamma.setText(str(gamma))
        
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
            widget_item = QtWidgets.QTableWidgetItem(str(data[i]))
            self.tableWidget_CIF.setItem(i, j, widget_item)
            
    def set_atom_site_table_col(self, data, j):
        for i in range(self.tableWidget_atm.rowCount()):
            widget_item = QtWidgets.QTableWidgetItem(str(data[i]))
            self.tableWidget_atm.setItem(i, j, widget_item)
            
    def set_unit_cell_site(self, data):
        self.set_unit_cell_table_col(data, self.unit_table['site'])

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
        
    def set_atom_site_occupancy(self, data):
        self.set_atom_site_table_col(data, self.atom_table['occupancy'])
        
    def set_atom_site_Uiso(self, data):
        self.set_atom_site_table_col(data, self.atom_table['Uiso'])
        
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
        
    def set_atom_site_mu(self, data):
        self.set_atom_site_table_col(data, self.atom_table['mu'])
        
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
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        for i in range(self.tableWidget_atm.rowCount()):
            for j in range(self.tableWidget_atm.columnCount()):
                item = self.tableWidget_atm.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setTextAlignment(alignment)
        
        horiz_hdr = self.tableWidget_atm.horizontalHeader()
        horiz_hdr.setSectionResizeMode(stretch)
        horiz_hdr.setSectionResizeMode(self.atom_table['occupancy'], resize)
        
        frac_coords = [self.atom_table['u'], 
                       self.atom_table['v'], 
                       self.atom_table['w']]
        
        for i in range(self.tableWidget_atm.rowCount()):
            for j in frac_coords:
                item = self.tableWidget_atm.item(i, j)
                if (item is not None and item.text() != ''):
                    item.setFlags(flags)

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
            cols += ['occupancy','u','v','w','']
        elif (parameters == 'Structural parameters'):
            cols = ['U11','U22','U33','U23','U13','U12']
        else:
            cols = ['ion','mu1','mu2','mu3','g','']
            
        show = [self.atom_table[key] for key in cols]         
        for i in range(len(self.atom_table)):
            if i in show:
                self.tableWidget_atm.setColumnHidden(i, False)
            else:
                self.tableWidget_atm.setColumnHidden(i, True)
                
    def show_unit_cell_table_cols(self):
        index = self.comboBox_parameters.currentIndex()    
        parameters = self.comboBox_parameters.itemText(index)
        
        cols = ['site', 'atom']
        if (parameters == 'Site parameters'):
            cols += ['isotope','occupancy']
        elif (parameters == 'Structural parameters'):
            cols += ['isotope','Uiso']
        else:
            cols += ['ion','mu']
        cols += ['u','v','w']
            
        show = [self.unit_table[key] for key in cols]         
        for i in range(len(self.unit_table)):
            if i in show:
                self.tableWidget_CIF.setColumnHidden(i, False)
            else:
                self.tableWidget_CIF.setColumnHidden(i, True)
                
    def unit_site_col(self, col):
        key = list(self.atom_table.keys())[col]
        
        if key in ['U11', 'U22', 'U33', 'U23', 'U13', 'U12']: key = 'Uiso'
        if key in ['mu1', 'mu2', 'mu3', 'g']: key = 'mu'
                
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
    
    def index_changed_type(self, slot):
        self.comboBox_type.currentIndexChanged.connect(slot)
        
    def get_type(self):
        index = self.comboBox_type.currentIndex()    
        return self.comboBox_type.itemText(index)
        
    def add_item_magnetic(self):
        self.comboBox_parameters.addItem('Magnetic parameters')

    def remove_item_magnetic(self):
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
            
    def set_atom_combo(self, atm):
        j = self.atom_table['atom']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = self.tableWidget_atm.cellWidget(i, j)
            index = combo.findText(atm[i], QtCore.Qt.MatchFixedString)
            if (index >= 0):
                 combo.setCurrentIndex(index)
            else:
                index = combo.findText(atm[i][:2], QtCore.Qt.MatchStartsWith)
                if (index >= 0):
                    combo.setCurrentIndex(index)
                else:
                    index = combo.findText(atm[i][0], QtCore.Qt.MatchStartsWith)
                    if (index >= 0):
                        combo.setCurrentIndex(index)
 
    def set_ion_combo(self, atm):
        j = self.atom_table['ion']
        for i in range(self.tableWidget_atm.rowCount()):
            combo = self.tableWidget_atm.cellWidget(i, j)
            index = combo.findText(atm[i], QtCore.Qt.MatchFixedString)
            if (index >= 0):
                 combo.setCurrentIndex(index)
            else:
                index = combo.findText(atm[i][:2], QtCore.Qt.MatchStartsWith)
                if (index >= 0):
                    combo.setCurrentIndex(index)
                else:
                    index = combo.findText(atm[i][0], QtCore.Qt.MatchStartsWith)
                    if (index >= 0):
                        combo.setCurrentIndex(index)                       
        
#    def 
#    
#        for i in range(ind.size):
#            combo = QtWidgets.QComboBox()
#            combo.setObjectName('comboBox_site'+str(i))
#            for t in data:
#                combo.addItem(t)
#            self.tableWidget_atm.setCellWidget(i, 0, combo)
#            index = combo.findText(atm[ind[i]], QtCore.Qt.MatchFixedString)
#            if (index >= 0):
#                 combo.setCurrentIndex(index)
#            else:
#                index = combo.findText(atm[ind[i]][:2], 
#                                       QtCore.Qt.MatchStartsWith)
#                if (index >= 0):
#                    combo.setCurrentIndex(index)
#                else:
#                    index = combo.findText(atm[ind[i]][0], 
#                                           QtCore.Qt.MatchStartsWith)
#                    if (index >= 0):
#                        combo.setCurrentIndex(index)                    
#            combo.currentIndexChanged.connect(self.combo_change_site)
#            atm[ind[i]] = combo.currentText()
#            
#            mag_atm = j0_atm[j0_atm == atm[ind[i]]]
#            mag_ion = j0_ion[j0_atm == atm[ind[i]]]
#
#            combo = QtWidgets.QComboBox()
#            combo.setObjectName('comboBox_ion'+str(i))
#            for j in range(mag_ion.size):
#                combo.addItem(mag_atm[j]+mag_ion[j])
#            if (mag_ion.size == 0):
#                combo.addItem('None')
            
#    def combo_change_ion(self):
#        
#        combo = self.sender()
#        index = self.tableWidget_atm.indexAt(combo.pos())
#        
#        site = index.row()
# 
#        atom = str(combo.currentText())
#        ion = atom.lstrip(numbers).lstrip(letters)
#        
#        for i in range(self.tableWidget_CIF.rowCount()):
#            s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
#            if (site == s):
#                self.tableWidget_CIF.setItem(i, 3, 
#                                             QtWidgets.QTableWidgetItem(ion))
#                self.tableWidget_CIF.item(i, 3).setTextAlignment(alignment)
    
    def add_site_check(self):
        for i in range(self.tableWidget_atm.rowCount()):
            check = QtWidgets.QCheckBox()
            check.setObjectName('checkBox_site'+str(i))
            check.setCheckState(QtCore.Qt.Checked) 
            self.tableWidget_atm.setCellWidget(i, 18, check)
    
    def check_clicked_site(self, slot):
        for i in range(self.tableWidget_atm.rowCount()):
            check = self.tableWidget_atm.cellWidget(i, 18)
            check.clicked.connect(slot)
            
    def change_site_check(self):
        n_atm = 0
        for i in range(self.tableWidget_atm.rowCount()):
            check = self.tableWidget_atm.cellWidget(i, 18)
            site = self.tableWidget_atm.indexAt(check.pos()).row()
            for j in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(j, 0).text())-1
                if (site == s):
                    if check.isChecked():
                        self.tableWidget_CIF.setRowHidden(j, False)
                        n_atm += 1
                    else:
                        self.tableWidget_CIF.setRowHidden(j, True)
        return n_atm                    
