import sys
import os
import traceback

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.style as mplstyle
mplstyle.use('fast')

from PyQt5 import QtWidgets, QtGui, QtCore, uic

import inspect
import warnings

from distutils.util import strtobool
from nexusformat.nexus import nxload

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
    
import pyvista as pv

from shutil import copyfile

# import pstats, cProfile
# import time
# import datetime

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
matplotlib.rcParams['mathtext.cal'] = 'sans'
matplotlib.rcParams['mathtext.rm'] = 'sans'
matplotlib.rcParams['mathtext.sf'] = 'sans'
matplotlib.rcParams['mathtext.tt'] = 'monospace'

_ROOT = os.path.abspath(os.path.dirname(__file__))

path = 'graphical'
sys.path.append(os.path.join(_ROOT, path))

# sys.path.append('.')

from disorder.diffuse import experimental, space, scattering, refinement
from disorder.diffuse import magnetic, occupational, displacive, monocrystal
from disorder.material import crystal, symmetry, tables

scattering.parallelism()

import disorder.correlation.functions as correlations

pm = '+-'
numbers = '0123456789'
letters = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'

bc_keys = tables.bc.keys()
bc_keys = np.array([s for s in bc_keys])

bc_atm = np.array([s.lstrip(numbers) for s in bc_keys])
bc_nuc = np.array([s.rstrip(letters) for s in bc_keys])
        
sort = np.lexsort(np.array((bc_nuc,bc_atm)))        
bc_keys = np.array(bc_keys)[sort]

j0_keys = tables.j0.keys()
j0_keys = np.array([s for s in j0_keys])

j0_atm = np.array([s.rstrip(pm).rstrip(numbers) for s in j0_keys])
j0_ion = np.array([s.lstrip(letters) for s in j0_keys])
        
sort = np.lexsort(np.array((j0_ion,j0_atm)))        
j0_keys = np.array(j0_keys)[sort]

X_keys = tables.X.keys()
X_keys = np.array([s for s in X_keys])

X_atm = np.array([s.rstrip(pm).rstrip(numbers) for s in X_keys])
X_ion = np.array([s.lstrip(letters) for s in X_keys])
        
sort = np.lexsort(np.array((X_ion,X_atm)))        
X_keys = np.array(X_keys)[sort]

alignment = int(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
options = QtWidgets.QFileDialog.Options()
options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
from matplotlib import ticker
from matplotlib.ticker import Locator
    
class MinorSymLogLocator(Locator):
    def __init__(self, linthresh, nints=10):
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        dmlower = majorlocs[1]-majorlocs[0]
        dmupper = majorlocs[-1]-majorlocs[-2]

        if (majorlocs[0] != 0. and 
            ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or 
             (dmlower == self.linthresh and majorlocs[0] < 0))):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        if (majorlocs[-1] != 0. and 
            ((np.abs(majorlocs[-1]) != self.linthresh 
              and dmupper > self.linthresh) or 
             (dmupper == self.linthresh and majorlocs[-1] > 0))):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        minorlocs = []

        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i]-majorlocs[i-1]
            if abs(majorlocs[i-1]+majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals-1.

            minorstep = majorstep/ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))
    
def save_gui(ui, settings):
    
    for name, obj in inspect.getmembers(ui):
        
        if isinstance(obj, QtWidgets.QComboBox):
            name = obj.objectName()
            index = obj.currentIndex()
            text = obj.itemText(index)
            settings.setValue(name, text)

        if isinstance(obj, QtWidgets.QLineEdit):
            name = obj.objectName()
            value = obj.text()
            settings.setValue(name, value)
            
        if isinstance(obj, QtWidgets.QProgressBar):
            name = obj.objectName()
            value = obj.value()
            settings.setValue(name, value)

        if isinstance(obj, QtWidgets.QCheckBox):
            name = obj.objectName()
            state = obj.isChecked()
            settings.setValue(name, state)     
            
        if isinstance(obj, QtWidgets.QTabWidget):
            name = obj.objectName()            
            index = obj.currentIndex()
            settings.setValue(name, index) 
            
        if isinstance(obj, QtWidgets.QTableWidget):
            name = obj.objectName()
            data = QtCore.QByteArray()
            stream = QtCore.QDataStream(data, QtCore.QIODevice.WriteOnly)
            rowCount = obj.rowCount()
            columnCount = obj.columnCount()
            stream.writeInt(rowCount)
            stream.writeInt(columnCount)
            for row in range(rowCount):
                stream.writeQString(obj.verticalHeaderItem(row).text())
            for col in range(columnCount):
                stream.writeQString(obj.horizontalHeaderItem(col).text())
            for row in range(rowCount):
                for col in range(columnCount):
                    if (obj.item(row, col) is not None):
                        stream.writeQString(obj.item(row, col).text())
                    else:
                        obj_cell = obj.cellWidget(row, col)
                        if isinstance(obj_cell, QtWidgets.QComboBox):
                            name_cell = obj_cell.objectName()
                            index_cell = obj_cell.currentIndex()
                            text_cell = obj_cell.itemText(index_cell)
                            settings.setValue(name_cell, text_cell)
                        if isinstance(obj_cell, QtWidgets.QCheckBox):
                            name_cell = obj_cell.objectName()
                            state_cell = obj_cell.isChecked()
                            settings.setValue(name_cell, state_cell)
                        stream.writeQString(name_cell)
            settings.setValue('{}/data'.format(name), data)

def load_gui(ui, settings):

    for name, obj in inspect.getmembers(ui):
                
        if isinstance(obj, QtWidgets.QComboBox):
            index = obj.currentIndex()
            name = obj.objectName()
            value = str(settings.value(name))  
            if (value == ''):
                continue
            index = obj.findText(value)
            if (index == -1):
                obj.insertItems(0,[value])
                index = obj.findText(value)
                obj.setCurrentIndex(index)
            else:
                obj.setCurrentIndex(index)    

        if isinstance(obj, QtWidgets.QLineEdit):
            name = obj.objectName()
            try:
                value = str(settings.value(name))
                obj.setText(value)
            except:
                pass
            
        if isinstance(obj, QtWidgets.QProgressBar):
            name = obj.objectName()
            try:
                value = np.int(settings.value(name))
                obj.setValue(value)
            except:
                pass
            
        if isinstance(obj, QtWidgets.QCheckBox):
            name = obj.objectName()
            try:
                value = bool(strtobool(settings.value(name)))
                obj.setChecked(value)
            except:
                pass
                
        if isinstance(obj, QtWidgets.QTabWidget):
            name = obj.objectName()          
            try:
                index = np.int(settings.value(name))
                obj.setCurrentIndex(index) 
            except:
                pass
 
        if isinstance(obj, QtWidgets.QTableWidget):
            name = obj.objectName()
            data = settings.value('{}/data'.format(name))
            if (not data):
                continue
            stream = QtCore.QDataStream(data, QtCore.QIODevice.ReadOnly)
            rowCount = stream.readInt()
            columnCount = stream.readInt()
            obj.setRowCount(rowCount)
            obj.setColumnCount(columnCount)
            for row in range(rowCount):
                cellText = str(stream.readQString())
                if (cellText):
                    obj.setVerticalHeaderItem(row, 
                                              QtWidgets.QTableWidgetItem(
                                              cellText))
            for col in range(columnCount):
                cellText = str(stream.readQString())
                if (cellText):
                    obj.setHorizontalHeaderItem(col, 
                                                QtWidgets.QTableWidgetItem(
                                                cellText))
            for row in range(rowCount):
                for col in range(columnCount):
                    cellText = str(stream.readQString())
                    if ('comboBox' in cellText):
                        combo = QtWidgets.QComboBox()
                        combo.setObjectName(cellText)         
                        obj.setCellWidget(row, col, combo)
                        combo.addItem(str(settings.value(
                                              cellText)))
                    elif ('checkBox' in cellText):
                        check = QtWidgets.QCheckBox()
                        check.setObjectName(cellText)
                        obj.setCellWidget(row, col, check)  
                        try:
                            value = bool(strtobool(settings.value(cellText)))
                            check.setChecked(value)
                        except:
                            pass
                    else:
                        obj.setItem(row, col, 
                                    QtWidgets.QTableWidgetItem(cellText))
                        
class WorkerSignals(QtCore.QObject):

    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

class Worker(QtCore.QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        self.kwargs['callback'] = self.signals.progress        

    @QtCore.pyqtSlot()
    def run(self):
  
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()       
        
qtCreatorFile_MainWindow = os.path.join(_ROOT, 'graphical/mainwindow.ui')
qtCreatorFile_Dialog = os.path.join(_ROOT, 'graphical/dialog.ui')
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile_MainWindow)
Ui_Dialog, QtBaseClass = uic.loadUiType(qtCreatorFile_Dialog)

class Dialog(QtWidgets.QDialog, Ui_Dialog):
    
    def __init__(self):
        
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
    def start_progress(self):
        self.ui.plainTextEdit_progress.clear()
        self.show()
        
    def update_progress(self, text):
        self.ui.plainTextEdit_progress.setPlainText(text)
        self.ui.plainTextEdit_progress.repaint()
        
    def stop_progress(self):
        self.hide()
    
class Window(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        icon = os.path.join(_ROOT, 'graphical/logo.png')
                
        self.setWindowIcon(QtGui.QIcon(icon))

        self.pushButton_load_CIF.clicked.connect(self.load_CIF)
        self.pushButton_save_CIF.clicked.connect(self.save_CIF)
        
        self.comboBox_type.addItem('Neutron', bc_keys.tolist())
        self.comboBox_type.addItem('X-ray', X_keys.tolist())
        self.comboBox_type.currentIndexChanged.connect(self.change_type)
        
        self.comboBox_parameters.addItem('Site parameters')
        self.comboBox_parameters.addItem('Structural parameters')
        self.comboBox_parameters.addItem('Magnetic parameters')
        self.comboBox_parameters.currentIndexChanged.connect(
                                                        self.change_parameters)
        
        self.tabWidget_disorder.setTabEnabled(0, True)
        self.tabWidget_disorder.setTabEnabled(1, True)
        self.tabWidget_disorder.setTabEnabled(2, True)
        
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
        
        self.lineEdit_nu.editingFinished.connect(self.supercell_n)
        self.lineEdit_nv.editingFinished.connect(self.supercell_n)
        self.lineEdit_nw.editingFinished.connect(self.supercell_n)
             
        self.pushButton_load_NXS.clicked.connect(self.load_NXS)
        self.pushButton_reset.clicked.connect(self.reset_hkl)

        self.checkBox_centered_h.stateChanged.connect(self.centered_integer)      
        self.checkBox_centered_k.stateChanged.connect(self.centered_integer)      
        self.checkBox_centered_l.stateChanged.connect(self.centered_integer)      
        
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
        
        self.comboBox_plot_exp.currentIndexChanged.connect(
                                                     self.replot_intensity_exp)
        self.comboBox_norm_exp.currentIndexChanged.connect(
                                                     self.replot_intensity_exp)
        
        self.lineEdit_radius_h.setText('0')
        self.lineEdit_radius_k.setText('0')
        self.lineEdit_radius_l.setText('0')
        self.lineEdit_outlier.setText('1.5')
        
        self.pushButton_punch.clicked.connect(self.punch)
        self.pushButton_reset_punch.clicked.connect(self.reset_punch)

        self.pushButton_reset_h.clicked.connect(self.reset_h)
        self.pushButton_reset_k.clicked.connect(self.reset_k)
        self.pushButton_reset_l.clicked.connect(self.reset_l)
        
        self.comboBox_rebin_h.currentIndexChanged.connect(self.change_rebin_h)
        self.comboBox_rebin_k.currentIndexChanged.connect(self.change_rebin_k)
        self.comboBox_rebin_l.currentIndexChanged.connect(self.change_rebin_l)
        
        self.lineEdit_min_h.editingFinished.connect(self.change_crop_min_h)
        self.lineEdit_min_k.editingFinished.connect(self.change_crop_min_k)
        self.lineEdit_min_l.editingFinished.connect(self.change_crop_min_l)

        self.lineEdit_max_h.editingFinished.connect(self.change_crop_max_h)
        self.lineEdit_max_k.editingFinished.connect(self.change_crop_max_k)
        self.lineEdit_max_l.editingFinished.connect(self.change_crop_max_l)
        
        self.lineEdit_slice_h.editingFinished.connect(self.change_slice_h)
        self.lineEdit_slice_k.editingFinished.connect(self.change_slice_k)
        self.lineEdit_slice_l.editingFinished.connect(self.change_slice_l)

        self.lineEdit_min_exp.editingFinished.connect(self.change_min_exp)
        self.lineEdit_max_exp.editingFinished.connect(self.change_max_exp)
        
        self.lineEdit_radius_h.editingFinished.connect(self.change_radius_h)
        self.lineEdit_radius_k.editingFinished.connect(self.change_radius_k)
        self.lineEdit_radius_l.editingFinished.connect(self.change_radius_l)

        self.lineEdit_outlier.editingFinished.connect(self.change_outlier)

        self.pushButton_save_intensity_exp.clicked.connect(
                                                       self.save_intensity_exp)
        
        self.lineEdit_cycles.setText('10')
        self.lineEdit_cycles.editingFinished.connect(self.change_cycles)
       
        self.lineEdit_filter_ref_h.setText('0.0')
        self.lineEdit_filter_ref_k.setText('0.0')
        self.lineEdit_filter_ref_l.setText('0.0')

        self.lineEdit_runs.setText('1')
        self.lineEdit_runs.setEnabled(False)
        self.lineEdit_runs.editingFinished.connect(self.change_runs)
        
        self.lineEdit_run.setEnabled(False)
        self.lineEdit_run.setText('0')

        self.lineEdit_order.setText('2')
      
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
        
        self.comboBox_slice.setCurrentIndex(2)
        self.lineEdit_slice.setText('0.0')
        
        self.comboBox_slice.currentIndexChanged.connect(
                                                     self.replot_intensity_ref)
        self.lineEdit_slice.editingFinished.connect(self.replot_intensity_ref)
       
        self.comboBox_slice.currentIndexChanged.connect(self.change_slice)
        self.lineEdit_slice.editingFinished.connect(self.change_slice)

        self.lineEdit_prefactor.setText('%1.2e' % 1e+4)
        self.lineEdit_tau.setText('%1.2e' % 1e-3)
        
        self.lineEdit_prefactor.editingFinished.connect(self.change_prefactor)
        self.lineEdit_tau.editingFinished.connect(self.change_tau)

        self.comboBox_plot_top_chi_sq.addItem('Accepted')
        self.comboBox_plot_top_chi_sq.addItem('Rejected')
        self.comboBox_plot_top_chi_sq.addItem('Temperature')
        self.comboBox_plot_top_chi_sq.addItem('Energy')
        self.comboBox_plot_top_chi_sq.addItem('Chi-squared')
        self.comboBox_plot_top_chi_sq.addItem('Scale factor')
        self.comboBox_plot_top_chi_sq.setCurrentIndex(0)
        
        self.comboBox_plot_bottom_chi_sq.addItem('Accepted')
        self.comboBox_plot_bottom_chi_sq.addItem('Rejected')
        self.comboBox_plot_bottom_chi_sq.addItem('Temperature')
        self.comboBox_plot_bottom_chi_sq.addItem('Energy')
        self.comboBox_plot_bottom_chi_sq.addItem('Chi-squared')
        self.comboBox_plot_bottom_chi_sq.addItem('Scale factor')
        self.comboBox_plot_bottom_chi_sq.setCurrentIndex(1)
        
        self.comboBox_plot_ref.addItem('Calculated')
        self.comboBox_plot_ref.addItem('Experimental')
        self.comboBox_plot_ref.addItem('Error')
        
        self.comboBox_norm_ref.addItem('Linear')
        self.comboBox_norm_ref.addItem('Logarithmic')

        self.lineEdit_chi_sq.setEnabled(False)

        self.magnetic = False
        self.occupational = False
        self.displacive = False
        self.restart = False
        self.started = False
        self.allocated = False
        self.stop = False
        self.running = False
        self.batch = 0
        self.iteration = 0
        self.progress = 0
        self.progressBar_ref.setValue(self.progress)
                
        self.pushButton_run.clicked.connect(self.run_refinement)
        self.pushButton_stop.clicked.connect(self.stop_refinement)
        self.pushButton_reset_run.clicked.connect(self.reset_refinement)
        
        self.tabWidget_disorder.currentChanged.connect(self.disorder_select)
        
        self.comboBox_plot_top_chi_sq.currentIndexChanged.connect(
                                                    self.plot_chi_sq)
        self.comboBox_plot_bottom_chi_sq.currentIndexChanged.connect(
                                                    self.plot_chi_sq)
        
        self.pushButton_save_chi_sq.clicked.connect(self.save_intensity_chi_sq)
        self.pushButton_save_intensity_ref.clicked.connect(
                                                       self.save_intensity_ref)
        
        self.comboBox_slice.currentIndexChanged.connect(self.change_slice)
        self.lineEdit_slice.editingFinished.connect(self.change_slice)

        self.comboBox_plot_ref.currentIndexChanged.connect(
                                                     self.replot_intensity_ref)
        self.comboBox_norm_ref.currentIndexChanged.connect(
                                                     self.replot_intensity_ref)

        self.lineEdit_min_ref.editingFinished.connect(self.change_min_ref)
        self.lineEdit_max_ref.editingFinished.connect(self.change_max_ref)
        
        self.lineEdit_filter_ref_h.editingFinished.connect(self.change_sigma_h)
        self.lineEdit_filter_ref_k.editingFinished.connect(self.change_sigma_k)
        self.lineEdit_filter_ref_l.editingFinished.connect(self.change_sigma_l)
        
        self.checkBox_batch.stateChanged.connect(self.check_batch)      
        self.lineEdit_runs.editingFinished.connect(self.change_runs)
        
        vectors = ['Correlation', 'Collinearity']
        scalars = ['Correlation']
        
        self.comboBox_correlations_1d.addItem('Moment', vectors)
        self.comboBox_correlations_1d.addItem('Occupancy', scalars)
        self.comboBox_correlations_1d.addItem('Displacement', vectors)
        self.comboBox_correlations_1d.setCurrentIndex(1)   
        self.comboBox_correlations_1d.currentIndexChanged.connect(
                                                           self.change_type_1d)

        self.comboBox_correlations_3d.addItem('Moment', vectors)
        self.comboBox_correlations_3d.addItem('Occupancy', scalars)
        self.comboBox_correlations_3d.addItem('Displacement', vectors) 
        self.comboBox_correlations_3d.setCurrentIndex(1)   
        self.comboBox_correlations_3d.currentIndexChanged.connect(
                                                           self.change_type_3d)      
        
        self.comboBox_plot_1d.addItem('Correlation')

        self.comboBox_norm_1d.addItem('Linear')
        self.comboBox_norm_1d.addItem('Logarithmic')

        self.comboBox_plot_3d.addItem('Correlation')

        self.comboBox_norm_3d.addItem('Linear')
        self.comboBox_norm_3d.addItem('Logarithmic')
        
        self.comboBox_plot_1d.currentIndexChanged.connect(self.update_plot_1d)
        self.comboBox_norm_1d.currentIndexChanged.connect(self.update_plot_1d)

        self.comboBox_plot_3d.currentIndexChanged.connect(self.update_plot_3d)
        self.comboBox_norm_3d.currentIndexChanged.connect(self.update_plot_3d)
        
        self.lineEdit_fract_1d.setText('0.125')
        self.lineEdit_fract_3d.setText('0.125')
        
        self.lineEdit_tol_1d.setText('1e-04')
        self.lineEdit_tol_3d.setText('1e-04')
        
        self.lineEdit_plane_h.setText('0')
        self.lineEdit_plane_k.setText('0')
        self.lineEdit_plane_l.setText('1')
        self.lineEdit_plane_d.setText('0.0')
        
        self.lineEdit_plane_h.editingFinished.connect(self.change_plane_h)
        self.lineEdit_plane_k.editingFinished.connect(self.change_plane_k)
        self.lineEdit_plane_l.editingFinished.connect(self.change_plane_l)
        self.lineEdit_plane_d.editingFinished.connect(self.change_plane_d)
        
        self.lineEdit_fract_1d.editingFinished.connect(self.change_fract_1d)
        self.lineEdit_fract_3d.editingFinished.connect(self.change_fract_3d)

        self.lineEdit_tol_1d.editingFinished.connect(self.change_tol_1d)
        self.lineEdit_tol_3d.editingFinished.connect(self.change_tol_3d)
        
        self.checkBox_batch_corr_1d.stateChanged.connect(
                                                      self.check_batch_corr_1d)
        self.lineEdit_runs_corr_1d.setText('1')
        self.lineEdit_runs_corr_1d.setEnabled(False)
        self.lineEdit_runs_corr_1d.editingFinished.connect(
                                                      self.change_runs_corr_1d)
        
        self.checkBox_batch_corr_3d.stateChanged.connect(
                                                      self.check_batch_corr_3d)
        self.lineEdit_runs_corr_3d.setText('1')
        self.lineEdit_runs_corr_3d.setEnabled(False)
        self.lineEdit_runs_corr_3d.editingFinished.connect(
                                                      self.change_runs_corr_3d)
        
        self.pushButton_calculate_1d.clicked.connect(
                                                self.calculate_correlations_1d)      
        self.pushButton_calculate_3d.clicked.connect(
                                                self.calculate_correlations_3d)
        
        self.checkBox_average_1d.stateChanged.connect(
                                              self.recalculate_correlations_1d)      
        self.checkBox_average_3d.stateChanged.connect(
                                              self.recalculate_correlations_3d)      
        self.checkBox_symmetrize.stateChanged.connect(
                                              self.recalculate_correlations_3d)      

        self.pushButton_save_1d.clicked.connect(self.save_1d)
        self.pushButton_save_3d.clicked.connect(self.save_3d)
        
        self.pushButton_save_CIF_dis.clicked.connect(self.save_CIF_dis)
        
        self.pushButton_save_CSV_correlations.clicked.connect(
                                                    self.save_CSV_correlations)                            
        self.pushButton_save_VTK_correlations.clicked.connect(
                                                    self.save_VTK_correlations)
        
        self.comboBox_axes.addItem('(h00), (0k0), (00l)')
        self.comboBox_axes.addItem('(hh0), (-kk0), (00l)')
        
        self.tabWidget_calc.setTabEnabled(0, True)
        self.tabWidget_calc.setTabEnabled(1, True)
        self.tabWidget_calc.setTabEnabled(2, True)
        
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
 
        self.checkBox_batch_calc.stateChanged.connect(self.check_batch_calc)      
        self.lineEdit_runs_calc.setText('1')
        self.lineEdit_runs_calc.setEnabled(False)
        self.lineEdit_runs_calc.editingFinished.connect(self.change_runs_calc) 
        
        self.tabWidget_disorder.currentChanged.connect(self.check_params_calc)      
        self.comboBox_laue.currentIndexChanged.connect(self.check_params_calc)      
        self.comboBox_axes.currentIndexChanged.connect(self.check_params_calc)      

        self.lineEdit_order_calc.setText('2')
        
        self.comboBox_slice_calc.addItem('h =')
        self.comboBox_slice_calc.addItem('k =')
        self.comboBox_slice_calc.addItem('l =')
        
        self.comboBox_slice_calc.setCurrentIndex(2)
        self.lineEdit_slice_calc.setText('0.0')
        
        self.comboBox_norm_calc.addItem('Linear')
        self.comboBox_norm_calc.addItem('Logarithmic')
        
        self.pushButton_calc.clicked.connect(self.recalculate_intensity)
        
        self.pushButton_save_calc.clicked.connect(self.save_intensity_calc)
        
        self.comboBox_slice_calc.currentIndexChanged.connect(
                                                        self.change_slice_calc)
        self.lineEdit_slice_calc.editingFinished.connect(
                                                        self.change_slice_calc)

        self.comboBox_norm_calc.currentIndexChanged.connect(
                                                    self.replot_intensity_calc)

        self.lineEdit_min_calc.editingFinished.connect(self.change_min_calc)
        self.lineEdit_max_calc.editingFinished.connect(self.change_max_calc)
        
        self.actionNew.triggered.connect(self.file_new)        
        self.actionSave_As.triggered.connect(self.file_save_as)
        self.actionSave.triggered.connect(self.file_save)
        self.actionOpen.triggered.connect(self.file_open)
        self.actionExit.triggered.connect(self.close_application)
        
        self.threadpool = QtCore.QThreadPool()
        
        self.popup_progress_dialog = Dialog()
        
        self.checkBox_mag = QtWidgets.QCheckBox()
        self.checkBox_mag.setObjectName('checkBox_mag')
        self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
        self.checkBox_mag.clicked.connect(self.disorder_check)
        self.tabWidget_disorder.tabBar().setTabButton(0,
                                                    QtWidgets.QTabBar.LeftSide, 
                                                      self.checkBox_mag)
        
        self.checkBox_occ = QtWidgets.QCheckBox()
        self.checkBox_occ.setObjectName('checkBox_occ')
        self.checkBox_occ.setCheckState(QtCore.Qt.Checked)
        self.checkBox_occ.clicked.connect(self.disorder_check)
        self.tabWidget_disorder.tabBar().setTabButton(1,
                                                    QtWidgets.QTabBar.LeftSide, 
                                                      self.checkBox_occ)
        
        self.checkBox_dis = QtWidgets.QCheckBox()
        self.checkBox_dis.setObjectName('checkBox_dis')
        self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)
        self.checkBox_dis.clicked.connect(self.disorder_check)
        self.tabWidget_disorder.tabBar().setTabButton(2,
                                                    QtWidgets.QTabBar.LeftSide, 
                                                      self.checkBox_dis)
        
    def disorder_check(self):
    
        sender = self.sender()
        
        if (sender.objectName() == 'checkBox_mag'):
            if (self.checkBox_mag.isChecked()):
                self.checkBox_mag.setCheckState(QtCore.Qt.Checked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)
            else:
                self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Checked)    
                self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)    
        elif (sender.objectName() == 'checkBox_occ'):
            if (self.checkBox_occ.isChecked()):
                self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Checked)    
                self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)  
            else:
                self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Checked)    
                self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)    
        elif (sender.objectName() == 'checkBox_dis'):
            if (self.checkBox_dis.isChecked()):
                self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Unchecked)    
                self.checkBox_dis.setCheckState(QtCore.Qt.Checked)  
            else:
                self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
                self.checkBox_occ.setCheckState(QtCore.Qt.Checked)    
                self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)  
                             
    def update_calc(self, item):
                
        i, j = item.row(), item.column()
        
        try :
            item = self.tableWidget_calc.item(i, j).text()
            
            self.tableWidget_calc.blockSignals(True)
            
            if (j == 1):
                try:
                    size = np.int(item)
                except:
                    size = 1
                if (size < 1):
                    size = 1
                minimum = np.float(self.tableWidget_calc.item(i, 2).text())
                maximum = np.float(self.tableWidget_calc.item(i, 3).text())
                if (size == 1):
                    step = '-'
                else:
                    step = (maximum-minimum)/(size-1)
                self.tableWidget_calc.setItem(i, 0, 
                                         QtWidgets.QTableWidgetItem(str(step)))
                self.tableWidget_calc.setItem(i, 1, 
                                         QtWidgets.QTableWidgetItem(str(size)))
            elif (j == 2):
                try:
                    minimum = np.float(item)
                except:
                    minimum = 0
                size = np.float(self.tableWidget_calc.item(i, 1).text())  
                maximum = np.float(self.tableWidget_calc.item(i, 3).text())
                if (minimum > maximum):
                    minimum = maximum
                if (size == 1):
                    step = '-'
                else:
                    step = (maximum-minimum)/(size-1)
                self.tableWidget_calc.setItem(i, 0, 
                                         QtWidgets.QTableWidgetItem(str(step)))
                self.tableWidget_calc.setItem(i, 2, 
                                      QtWidgets.QTableWidgetItem(str(minimum)))
            elif (j == 3):
                try:
                    maximum = np.float(item)
                except:
                    maximum = 0
                size = np.float(self.tableWidget_calc.item(i, 1).text())  
                minimum = np.float(self.tableWidget_calc.item(i, 2).text())
                if (maximum < minimum):
                    maximum = minimum
                if (size == 1):
                    step = '-'
                else:
                    step = (maximum-minimum)/(size-1)
                self.tableWidget_calc.setItem(i, 0, 
                                         QtWidgets.QTableWidgetItem(str(step)))
                self.tableWidget_calc.setItem(i, 3, 
                                      QtWidgets.QTableWidgetItem(str(maximum)))
            elif (j == 4):
                try:
                    sigma = np.float(item)
                except:
                    sigma = 0.
                if (sigma < 0):
                    sigma = 0.       
                self.tableWidget_calc.setItem(i, 4, 
                                        QtWidgets.QTableWidgetItem(str(sigma)))
                    
            try:
                if (j <= 3 and j > 0):
                    param = self.recalc_params[i+3*(j-1)]
                    if (j == 1):
                        test = size
                    elif (j == 2):
                        test = minimum
                    elif (j == 3):
                        test = maximum
                    if (np.isclose(param,test)):
                        self.changed_params = False
                    else:
                        self.changed_params = True
            except:
                self.changed_params = True
                    
            for j in range(5):
                self.tableWidget_calc.item(i, j).setTextAlignment(alignment)
                
            self.tableWidget_calc.blockSignals(False)
            
        except:
            
            pass
        
    def plot_intensity_calc(self, data):
        
        index = self.comboBox_norm_calc.currentIndex()    
        norm = self.comboBox_norm_calc.itemText(index)
        
        index_hkl = self.comboBox_slice_calc.currentIndex()    
        hkl = self.comboBox_slice_calc.itemText(index_hkl)
        
        slice_hkl = np.float(self.lineEdit_slice_calc.text())
                
        try:
            step_h = np.float(self.tableWidget_calc.item(0, 0).text())
        except:
            step_h = 0
        size_h = np.int(np.float(self.tableWidget_calc.item(0, 1).text()))

        min_h = np.float(self.tableWidget_calc.item(0, 2).text())
        max_h = np.float(self.tableWidget_calc.item(0, 3).text())

        try:
            step_k = np.float(self.tableWidget_calc.item(1, 0).text())
        except:
            step_k = 0                
        size_k = np.int(np.float(self.tableWidget_calc.item(1, 1).text()))

        min_k = np.float(self.tableWidget_calc.item(1, 2).text())
        max_k = np.float(self.tableWidget_calc.item(1, 3).text())

        try:
            step_l = np.float(self.tableWidget_calc.item(2, 0).text())
        except:
            step_l = 0                   
        size_l = np.int(np.float(self.tableWidget_calc.item(2, 1).text()))

        min_l = np.float(self.tableWidget_calc.item(2, 2).text())
        max_l = np.float(self.tableWidget_calc.item(2, 3).text())
        
        vmin = np.float(self.lineEdit_min_calc.text())
        vmax = np.float(self.lineEdit_max_calc.text())
        
        if (norm == 'Logarithmic'):
            normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            normalize = colors.Normalize(vmin=vmin, vmax=vmax)
            
        index = self.comboBox_axes.currentIndex()    
        
        if (index == 1):
            T = np.array([[1, -1,  0],
                          [1,  1,  0],
                          [0,  0,  1]])*1.
            
        pm = ['', '+', '-']
                    
        fig = self.canvas_calc.figure
        fig.clear()   
        
        ax = fig.add_subplot(111)
        
        ax.set_aspect(1.)
        
        if (hkl == 'h ='):
                
            if (size_h == 1 or np.isclose(step_h, 0)):
                ih = 0
                h = min_h
            else:
                ih = np.int(np.round((slice_hkl-min_h)/step_h))
                h = np.round(min_h+step_h*ih,4)

            dk = np.float(max_k-min_k)/size_k
            dl = np.float(max_l-min_l)/size_l
            extents_h = [min_k-dk/2, max_k+dk/2, min_l-dl/2, max_l+dl/2]
        
            im = ax.imshow(data[ih,:,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_h)
            
            if (index == 0):
                ax.set_title(r'$('+str(h)+',k,l)$', fontsize='small') 
                ax.set_xlabel(r'$(0k0)$', fontsize='small')
                ax.set_ylabel(r'$(00l)$', fontsize='small')
            else:        
                if (np.isclose(h,0)):
                    ax.set_title(r'$(-k,k,l)$', fontsize='small') 
                else:
                    ax.set_title(r'$(-k'+pm[np.sign(h)]+str(abs(h))\
                                   +',k'+pm[np.sign(h)]+str(abs(h))\
                                   +'l)$', fontsize='small')                    
                ax.set_xlabel(r'$(\bar{k}k0)$', fontsize='small')
                ax.set_ylabel(r'$(00l)$', fontsize='small')
                
            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[1,1]/self.B[2,2],self.B[1,2]/self.B[2,2],0],
                          [self.B[2,1]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                          [0,0,1]])
    
            if (index == 1):
                Q = np.eye(3)
                Q[0:2,0:2] = T.T[1:3:2,1:3:2].copy()
                
                N = np.dot(M.T,M)
                
                scale_trans = N[1,1].copy()
                
                N[0:2] /= scale_trans
                N[0:2] /= scale_trans
                
                M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
                
                rot = T[2,2]/T[1,1]*(np.linalg.norm(T[1])/np.linalg.norm(T[2]))
            else:
                rot = 1
                
            scale_rot = M[1,1].copy()
            
            M[0,1] /= scale_rot
            M[0,0] /= scale_rot
            M[1,1] /= scale_rot
    
            scale = M[0,0]
            
            M[0,1] /= scale
            M[0,0] /= scale
            
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_l,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale/rot)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_h[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_h[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])
            
        elif (hkl == 'k ='):
            
            if (size_k == 1 or np.isclose(step_k, 0)):
                ik = 0
                k = min_k
            else:
                ik = np.int(np.round((slice_hkl-min_k)/step_k))
                k = np.round(min_k+step_k*ik,4)  

            dh = np.float(max_h-min_h)/size_h
            dl = np.float(max_l-min_l)/size_l
            extents_k = [min_h-dh/2, max_h+dh/2, min_l-dl/2, max_l+dl/2]
            
            im = ax.imshow(data[:,ik,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_k)
        
            
            if (index == 0):
                ax.set_title(r'$(h,'+str(k)+',l)$', fontsize='small')
                ax.set_xlabel(r'$(h00)$', fontsize='small')
                ax.set_ylabel(r'$(00l)$', fontsize='small')
            else:     
                if (np.isclose(k,0)):
                    ax.set_title(r'$(h,h,l)$', fontsize='small')
                else:
                    ax.set_title(r'$(h'+pm[np.sign(k)]+str(abs(k))\
                                 +',-h'+pm[np.sign(k)]+str(abs(k))\
                                   +'l)$', fontsize='small')    
                ax.set_xlabel(r'$(hh0)$', fontsize='small')
                ax.set_ylabel(r'$(00l)$', fontsize='small')
                
            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[0,0]/self.B[2,2],self.B[0,2]/self.B[2,2],0],
                          [self.B[2,0]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                          [0,0,1]])
    
            if (index == 1):
                Q = np.eye(3)
                Q[0:2,0:2] = T.T[0:3:2,0:3:2].copy()
                
                N = np.dot(M.T,M)
                
                scale_trans = N[1,1].copy()
                
                N[0:2] /= scale_trans
                N[0:2] /= scale_trans
                
                M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
                
                rot = T[2,2]/T[0,0]*(np.linalg.norm(T[0])/np.linalg.norm(T[2]))
            else:
                rot = 1
                
            scale_rot = M[1,1].copy()
            
            M[0,1] /= scale_rot
            M[0,0] /= scale_rot
            M[1,1] /= scale_rot
            
            scale = M[0,0]
            
            M[0,1] /= scale
            M[0,0] /= scale
            
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_l,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale/rot)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_k[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_k[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])

        else:
            
            if (size_l == 1 or step_l == 0):
                il = 0
                l = min_l
            else:
                il = np.int(np.round((slice_hkl-min_l)/step_l))
                l = np.round(min_l+step_l*il,4) 

            dh = np.float(max_h-min_h)/size_h
            dk = np.float(max_k-min_k)/size_k
            extents_l = [min_h-dh/2, max_h+dh/2, min_k-dk/2, max_k+dk/2]
        
            im = ax.imshow(data[:,:,il].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_l)
            
            if (index == 0):
                ax.set_title(r'$(h,k,'+str(l)+')$', fontsize='small') 
                ax.set_xlabel(r'$(h00)$', fontsize='small')
                ax.set_ylabel(r'$(0k0)$', fontsize='small')
            else:        
                ax.set_title(r'$(h,k,'+str(l)+')$', fontsize='small') 
                ax.set_xlabel(r'$(hh0)$', fontsize='small')
                ax.set_ylabel(r'$(\bar{k}k0)$', fontsize='small')
                
            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[0,0]/self.B[1,1],self.B[0,1]/self.B[1,1],0],
                          [self.B[1,0]/self.B[1,1],self.B[1,1]/self.B[1,1],0],
                          [0,0,1]])
            
            if (index == 1):
                Q = np.eye(3)
                Q[0:2,0:2] = T.T[0:2,0:2].copy()
                
                N = np.dot(M.T,M)
                
                scale_trans = N[1,1].copy()
                
                N[0:2] /= scale_trans
                N[0:2] /= scale_trans
                
                M = np.linalg.cholesky(np.dot(Q,np.dot(N,Q.T))).T
                
                rot = T[1,1]/T[0,0]*(np.linalg.norm(T[0])/np.linalg.norm(T[1]))
            else:
                rot = 1
                
            scale_rot = M[1,1].copy()
            
            M[0,1] /= scale_rot
            M[0,0] /= scale_rot
            M[1,1] /= scale_rot
                    
            scale = M[0,0].copy()
            
            M[0,1] /= scale
            M[0,0] /= scale
                
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_k,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale/rot)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_l[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_l[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])
   
        ax.xaxis.tick_bottom()
        
        ax.minorticks_on()
        
        ax.axes.tick_params(labelsize='small')
    
        fig.tight_layout(pad=3.24)
        
        cb = fig.colorbar(im, ax=ax)
        cb.ax.minorticks_on()
        
        if (norm == 'Linear'):
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
            
        cb.ax.tick_params(labelsize='small') 

        with np.errstate(invalid='ignore'):
            self.canvas_calc.draw()
        
    def save_intensity_calc(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        
        if (name):
                
            fig = self.canvas_calc.figure
            fig.savefig(name)
        
    def change_min_calc(self):

        if (self.allocated or self.progress > 0):
            
            try:
                vmin = np.float(self.lineEdit_min_calc.text())
                
                if (vmin >= np.float(self.lineEdit_max_calc.text())):
                    raise
                    
                self.lineEdit_min_calc.setText('%1.4e' % vmin)
                
                self.plot_intensity_calc(self.I_recalc)
            except:
                self.replot_intensity_calc()
            
    def change_max_calc(self):
            
        if (self.allocated or self.progress > 0):

            try:
                vmax = np.float(self.lineEdit_max_calc.text())
                
                if (vmax <= np.float(self.lineEdit_min_calc.text())):
                    raise
                    
                self.lineEdit_max_calc.setText('%1.4e' % vmax)
               
                self.plot_intensity_calc(self.I_recalc)
            except:
                self.replot_intensity_calc()
                
    def replot_intensity_calc(self):
                
        if (self.allocated or self.progress > 0):
            try:
                self.recalculated
                self.lineEdit_min_calc.setText('%1.4e' % self.I_recalc.min())
                self.lineEdit_max_calc.setText('%1.4e' % self.I_recalc.max())
                self.plot_intensity_calc(self.I_recalc)
            except:
                pass
        
    def change_runs_calc(self):
        
        try:
            runs = np.int(self.lineEdit_runs_calc.text())
        except:
            runs = 1
            
        self.lineEdit_runs_calc.setText(str(runs))
        
        try:
            param = self.recalc_params[12]
            if (np.isclose(param,runs)):
                self.changed_params = False
            else:
                self.changed_params = True
        except:
            self.changed_params = True
        
    def check_batch_calc(self):
        
        if (self.checkBox_batch_calc.isChecked()):
            self.lineEdit_runs_calc.setEnabled(True)
        else:
            self.lineEdit_runs_calc.setEnabled(False)
            self.lineEdit_runs_calc.setText('1')
            
        try:
            param = self.recalc_params[11]
            if (np.isclose(param,self.checkBox_batch_calc.isChecked())):
                self.changed_params = False
            else:
                self.changed_params = True
        except:
            self.changed_params = True
            
    def check_params_calc(self):
        
        index = self.comboBox_axes.currentIndex()
        laue_index = self.comboBox_laue.currentIndex()    
            
        try:
            param_laue = self.recalc_params[9]
            param_ind = self.recalc_params[10]

            if (np.isclose(param_laue,laue_index) and \
                np.isclose(param_ind,index)):
                self.changed_params = False
            else:
                self.changed_params = True
        except:
            self.changed_params = True
        
    def change_slice_calc(self):
 
        if (self.allocated):

            index_hkl = self.comboBox_slice_calc.currentIndex()    
            hkl = self.comboBox_slice_calc.itemText(index_hkl)
                
            slice_hkl = np.float(self.lineEdit_slice_calc.text())
                    
            if (hkl == 'h = '):
                size_h = np.float(self.tableWidget_calc.item(0, 1).text())
                min_h = np.float(self.tableWidget_calc.item(0, 2).text())
                max_h = np.float(self.tableWidget_calc.item(0, 3).text())
                if (size_h > 1):
                    step_h = (max_h-min_h)/(size_h-1)
                    if (slice_hkl < min_h):
                        h = min_h
                    elif (slice_hkl > max_h):
                        h = max_h
                    else:
                        ih = np.int(np.round((slice_hkl-min_h)/step_h))
                        h = np.round(min_h+step_h*ih, 4)
                else:
                    h = min_h
                self.lineEdit_slice_calc.setText(str(h))
            elif (hkl == 'k = '):
                size_k = np.float(self.tableWidget_calc.item(1, 1).text())
                min_k = np.float(self.tableWidget_calc.item(1, 2).text())
                max_k = np.float(self.tableWidget_calc.item(1, 3).text())
                if (size_k > 1):
                    step_k = (max_k-min_k)/(size_k-1)
                    if (slice_hkl < min_k):
                        k = min_k
                    elif (slice_hkl > max_k):
                        k = max_k
                    else:
                        ik = np.int(np.round((slice_hkl-min_k)/step_k))
                        k = np.round(min_k+step_k*ik, 4)
                else:
                    k = min_k
                self.lineEdit_slice_calc.setText(str(k))
            else:
                size_l = np.float(self.tableWidget_calc.item(2, 1).text())
                min_l = np.float(self.tableWidget_calc.item(2, 2).text())
                max_l = np.float(self.tableWidget_calc.item(2, 3).text())   
                if (size_l > 1):
                    step_l = (max_l-min_l)/(size_l-1)
                    if (slice_hkl < min_l):
                        l = min_l
                    elif (slice_hkl > max_l):
                        l = max_l
                    else:
                        il = np.int(np.round((slice_hkl-min_l)/step_l))
                        l = np.round(min_l+step_l*il, 4)  
                else:
                    l = min_l
                self.lineEdit_slice_calc.setText(str(l))
        
            self.replot_intensity_calc()
            
    def recalculate_intensity_thread(self, callback):

        if (self.tableWidget_calc.rowCount() or self.progress > 0):

            if (self.changed_params):
                            
                if (self.allocated == False):
                    self.preprocess_supercell()
                                                    
                batch = self.checkBox_batch_calc.isChecked()
                
                runs = np.int(self.lineEdit_runs_calc.text())
                
                # ---
                
                nh = np.int(self.tableWidget_calc.item(0, 1).text())
                nk = np.int(self.tableWidget_calc.item(1, 1).text())
                nl = np.int(self.tableWidget_calc.item(2, 1).text())
                
                twins = np.zeros((1,3,3))
                variants = np.array([1.])
                
                twins[0,:,:] = np.eye(3)
                                            
                index = self.comboBox_axes.currentIndex()
                
                if (index == 0):
                    T = np.eye(3)
                else:                
                    T = np.array([[1, -1,  0],
                                  [1,  1,  0],
                                  [0,  0,  1]])*1.
                
                min_h = np.float(self.tableWidget_calc.item(0, 2).text())
                min_k = np.float(self.tableWidget_calc.item(1, 2).text())
                min_l = np.float(self.tableWidget_calc.item(2, 2).text())
                
                max_h = np.float(self.tableWidget_calc.item(0, 3).text())
                max_k = np.float(self.tableWidget_calc.item(1, 3).text())
                max_l = np.float(self.tableWidget_calc.item(2, 3).text())
                
                h_range = [min_h, max_h]
                k_range = [min_k, max_k]
                l_range = [min_l, max_l]
                
                laue_index = self.comboBox_laue.currentIndex()    
                laue = self.comboBox_laue.itemText(laue_index)
                                
                self.recalc_params = [nh,nk,nl,
                                      min_h,min_k,min_l,
                                      max_h,max_k,max_l,
                                      laue_index,index,batch,runs]
                
                self.intensity = np.zeros((nh,nk,nl))
                            
                indices, \
                inverses, \
                operators, \
                Nu, \
                Nv, \
                Nw = crystal.reduced(h_range,
                                     k_range,
                                     l_range,
                                     nh,
                                     nk,
                                     nl,
                                     self.nu,
                                     self.nv,
                                     self.nw,
                                     T=T,
                                     folder=self.folder, 
                                     filename=self.filename,
                                     symmetry=laue)
                
                lauesym = symmetry.operators(invert=True)
                
                symmetries = list(lauesym.keys())
                
                symop = [11,1]
                
                for count, sym in enumerate(symmetries):
                    if (np.array([operators[p] in lauesym.get(sym) \
                        for p in range(operators.shape[0])]).all() and \
                        len(lauesym.get(sym)) == operators.shape[0]):
                        
                        symop = [count,len(lauesym.get(sym))]
                    
                if (self.displacive):
                    
                    p = np.int(self.lineEdit_order_calc.text())
    
                    coeffs = displacive.coefficients(p)
                    
                    start = (np.cumsum(displacive.number(np.arange(p+1)))
                          -  displacive.number(np.arange(p+1)))[::2]
                    end = np.cumsum(displacive.number(np.arange(p+1)))[::2]
                    
                    even = []
                    for k in range(len(end)):
                        even += range(start[k], end[k])
                    even = np.array(even)
                            
                    nuclear = ['P', 'I', 'F', 'R', 'C', 'A', 'B']
                                        
                    cntr = np.argwhere([x in self.centering \
                                        for x in nuclear])[0][0]
                    
                    cntr += 1
                        
                # ---
                    
                for run in range(runs):               
    
                    if (batch):
                        r = '-'+str(run)
                    else:
                        r = ''  
                        
                    if (self.magnetic):
                        
                        Sx = np.load(self.fname+'-calculated-spin-x'+r+'.npy')
                        Sy = np.load(self.fname+'-calculated-spin-y'+r+'.npy')
                        Sz = np.load(self.fname+'-calculated-spin-z'+r+'.npy')
                        
                        I_calc = monocrystal.magnetic(Sx, 
                                                      Sy, 
                                                      Sz, 
                                                      self.ux, 
                                                      self.uy, 
                                                      self.uz, 
                                                      self.atm,
                                                      h_range,
                                                      k_range,
                                                      l_range,
                                                      indices,
                                                      symop,
                                                      T,
                                                      self.B,
                                                      self.R,
                                                      twins,
                                                      variants,
                                                      nh,
                                                      nk,
                                                      nl,
                                                      self.nu,
                                                      self.nv,
                                                      self.nw,
                                                      Nu,
                                                      Nv,
                                                      Nw,
                                                      self.g)
                                                
                    elif (self.occupational):
                            
                        A_r = np.load(self.fname\
                                           +'-calculated-composition'+r+'.npy')
                        
                        I_calc = monocrystal.occupational(A_r, 
                                                          self.occupancy, 
                                                          self.ux, 
                                                          self.uy, 
                                                          self.uz, 
                                                          self.atm,
                                                          h_range,
                                                          k_range,
                                                          l_range,
                                                          indices,
                                                          symop,
                                                          T,
                                                          self.B,
                                                          self.R,
                                                          twins,
                                                          variants,
                                                          nh,
                                                          nk,
                                                          nl,
                                                          self.nu,
                                                          self.nv,
                                                          self.nw,
                                                          Nu,
                                                          Nv,
                                                          Nw)
        
                    else:
                        
                        Ux = np.load(self.fname\
                                     +'-calculated-displacement-x'+r+'.npy')
                        Uy = np.load(self.fname\
                                     +'-calculated-displacement-y'+r+'.npy')
                        Uz = np.load(self.fname\
                                     +'-calculated-displacement-z'+r+'.npy')
                            
                        U_r = displacive.products(Ux, Uy, Uz, p)
                                
                        I_calc = monocrystal.displacive(U_r, 
                                                        coeffs,
                                                        self.ux, 
                                                        self.uy, 
                                                        self.uz, 
                                                        self.atm,
                                                        h_range,
                                                        k_range,
                                                        l_range,
                                                        indices,
                                                        symop,
                                                        T,
                                                        self.B,
                                                        self.R,
                                                        twins,
                                                        variants,
                                                        nh,
                                                        nk,
                                                        nl,
                                                        self.nu,
                                                        self.nv,
                                                        self.nw,
                                                        Nu,
                                                        Nv,
                                                        Nw,
                                                        p,
                                                        even,
                                                        cntr)
    
                self.intensity[:,:,:] += I_calc[inverses].reshape(nh,nk,nl)
                            
                self.intensity /= runs*operators.shape[0]
                
                self.recalculated = True
                
                self.changed_params = False
                            
            if (self.recalculated):
                
                sigma_h = np.float(self.tableWidget_calc.item(0, 4).text())
                sigma_k = np.float(self.tableWidget_calc.item(1, 4).text())
                sigma_l = np.float(self.tableWidget_calc.item(2, 4).text())
    
                sigma = [sigma_h, sigma_k, sigma_l]
                
                self.I_recalc = space.blurring(self.intensity, sigma)
                
    def recalculate_intensity_thread_complete(self):
        
        self.replot_intensity_calc()
        
        self.pushButton_calc.setEnabled(True)        
                                                                        
    def recalculate_intensity(self):
        
        self.pushButton_calc.setEnabled(False)
        
        self.canvas_calc.figure.clear()
        with np.errstate(invalid='ignore'):
            self.canvas_calc.draw()
            
        worker = Worker(self.recalculate_intensity_thread)
        worker.signals.finished.connect(
                                    self.recalculate_intensity_thread_complete)
        
        self.threadpool.start(worker)
        
    def save_VTK_correlations(self):
                    
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'VTK files *.vtm',
                                                        options=options)
                
        if (name):
            
            blocks = pv.MultiBlock()
            
            selection = self.comboBox_correlations_3d.currentIndex()    
            data = self.comboBox_correlations_3d.itemData(selection)
                        
            if (self.checkBox_average_3d.isChecked()):  
                points = np.column_stack((self.dx,self.dy,self.dz))
                for t in data:
                    blocks[t] = pv.PolyData(points)
                    if (t == 'Correlation'):
                        array = self.S_corr3d
                    else:
                        array = self.S_coll3d
                    blocks[t].point_arrays[t] = array
            else:
                for i in range(self.tableWidget_pairs_3d.rowCount()):
                    atom = self.tableWidget_pairs_3d.item(i, 0).text()                
                    if (atom == 'self-correlation'):
                        label = '0'
                    else:
                        pair = self.tableWidget_pairs_3d.item(i, 1).text()  
                        label = atom+'_'+pair
                    mask = self.atm_pair3d == label
                    for t in data:
                        points = np.column_stack((self.dx[mask],
                                                  self.dy[mask],
                                                  self.dz[mask]))
                        blocks[t+'-'+label] = pv.PolyData(points)
                        if (t == 'Correlation'):
                            array = self.S_corr3d[mask]
                        else:
                            array = self.S_coll3d[mask]
                        blocks[t+'-'+label].point_arrays[t+'-'+label] = array
                        
            blocks.save(name, binary=False)
            
    def save_CSV_correlations(self):
                      
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'CSV files *.csv',
                                                        options=options)

        if (name):        
            
            selection = self.comboBox_correlations_3d.currentIndex()    
            data = self.comboBox_correlations_3d.itemData(selection)
            
            if (self.checkBox_average_1d.isChecked()):  
                if (len(data) > 1):
                    np.savetxt(name, 
                               np.column_stack((self.d, 
                                                self.S_corr, 
                                                self.S_coll)), 
                               delimiter=',', 
                               fmt='%s',
                               header='d,corr,coll')
                else:
                    np.savetxt(name, 
                               np.column_stack((self.d, 
                                                self.S_corr)), 
                               delimiter=',', 
                               fmt='%s',
                               header='d,corr')
            else:
                if (len(data) > 1):
                    np.savetxt(name, 
                               np.column_stack((self.d, 
                                                self.S_corr, 
                                                self.S_coll, 
                                                self.atm_pair)), 
                               delimiter=',', 
                               fmt='%s',
                               header='d,corr,coll,pair')
                else:
                    np.savetxt(name, 
                               np.column_stack((self.d, 
                                                self.S_corr, 
                                                self.atm_pair)), 
                               delimiter=',', 
                               fmt='%s',
                               header='d,corr,pair')
                               
    def save_CIF_dis(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options)
                
        if (name):
            
            try:
                crystal.magnetic(self.Sx, 
                                 self.Sy, 
                                 self.Sz, 
                                 self.rx, 
                                 self.ry, 
                                 self.rz, 
                                 self.nu,
                                 self.nv,
                                 self.nw,
                                 self.atm, 
                                 self.A,
                                 folder=self.folder, 
                                 filename=self.filename,
                                 xlim=[0,self.nu],
                                 ylim=[0,self.nv],
                                 zlim=[0,self.nw])
                os.rename(self.folder+self.filename.replace('.cif','.mcif'), 
                          name.replace('.cif','.mcif'))
            except:
                pass

            try:
                delta_r = ((self.A_r.reshape(self.A_r.size
                        // self.n_atm,self.n_atm)+1)*self.occupancy).flatten()
                crystal.occupational(delta_r, 
                                     self.rx, 
                                     self.ry, 
                                     self.rz, 
                                     self.nu,
                                     self.nv,
                                     self.nw,
                                     self.atm, 
                                     self.A,
                                     folder=self.folder, 
                                     filename=self.filename,
                                     xlim=[0,self.nu],
                                     ylim=[0,self.nv],
                                     zlim=[0,self.nw])
                os.rename(self.folder+self.filename.replace('.cif','-occ.cif'), 
                          name)
            except:
                pass
            
            try:
                crystal.displacive(self.Ux, 
                                   self.Uy, 
                                   self.Uz, 
                                   self.rx, 
                                   self.ry, 
                                   self.rz, 
                                   self.nu,
                                   self.nv,
                                   self.nw,
                                   self.atm, 
                                   self.A,
                                   folder=self.folder, 
                                   filename=self.filename,
                                   xlim=[0,self.nu],
                                   ylim=[0,self.nv],
                                   zlim=[0,self.nw])   
                os.rename(self.folder+self.filename.replace('.cif','-dis.cif'), 
                          name)
            except:
                pass
            
    def change_plane_h(self):
        
        try:
            plane = np.int(self.lineEdit_plane_h.text())
        except:
            plane = 1
            
        self.lineEdit_plane_h.setText(str(plane))
        self.update_plot_3d()
        
    def change_plane_k(self):
        
        try:
            plane = np.int(self.lineEdit_plane_k.text())
        except:
            plane = 1
            
        self.lineEdit_plane_k.setText(str(plane))
        self.update_plot_3d()
        
    def change_plane_l(self):
        
        try:
            plane = np.int(self.lineEdit_plane_l.text())
        except:
            plane = 1
            
        self.lineEdit_plane_l.setText(str(plane))
        self.update_plot_3d()
        
    def change_plane_d(self):
        
        try:
            plane = np.float(self.lineEdit_plane_d.text())
        except:
            plane = 0.0
            
        self.lineEdit_plane_d.setText(str(plane))
        self.update_plot_3d()
        
    def change_fract_3d(self):
        
        try:
            fract = np.float(self.lineEdit_fract_3d.text())
        except:
            fract = 0.125
            
        if (fract <= 0 or fract >= 1):
            fract = 0.125
            
        self.lineEdit_fract_3d.setText(str(fract))
        self.recalculate_correlations_3d()
        
    def change_fract_1d(self):
        
        try:
            fract = np.float(self.lineEdit_fract_1d.text())
        except:
            fract = 0.125
            
        if (fract <= 0 or fract >= 1):
            fract = 0.125
            
        self.lineEdit_fract_1d.setText(str(fract))
        self.recalculate_correlations_1d()
        
    def change_tol_3d(self):
        
        try:
            tol = np.float(self.lineEdit_tol_3d.text())
        except:
            tol = 1e-4
            
        if (tol <= 0 or tol >= 1):
            tol = 1e-4
            
        tol = 10.**(np.round(np.log10(tol)).astype(int))
        self.lineEdit_tol_3d.setText('%1.0e' % tol)
        self.recalculate_correlations_3d()
        
    def change_tol_1d(self):
        
        try:
            tol = np.float(self.lineEdit_tol_1d.text())
        except:
            tol = 1e-4
            
        if (tol <= 0 or tol >= 1):
            tol = 1e-4
            
        tol = 10.**(np.round(np.log10(tol)).astype(int))
        self.lineEdit_tol_1d.setText('%1.0e' % tol)
        self.recalculate_correlations_1d()
       
    def change_type_3d(self):
                
        selection = self.comboBox_correlations_3d.currentIndex()    
        data = self.comboBox_correlations_3d.itemData(selection)
        
        self.comboBox_plot_3d.blockSignals(True)        
        self.comboBox_plot_3d.clear()
        for t in data:                   
            self.comboBox_plot_3d.addItem(t)           
        self.comboBox_plot_3d.blockSignals(False)    
        
    def change_type_1d(self):
                
        selection = self.comboBox_correlations_1d.currentIndex()    
        data = self.comboBox_correlations_1d.itemData(selection)

        self.comboBox_plot_1d.blockSignals(True)        
        self.comboBox_plot_1d.clear()
        for t in data:                   
            self.comboBox_plot_1d.addItem(t)            
        self.comboBox_plot_1d.blockSignals(False) 
        
    def update_plot_3d(self):
        
        if (self.allocated or self.progress > 0):
            
            try:
                self.atm_pair3d
                self.plot_3d()
            except:
                pass
            
    def save_3d(self):
             
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
                       
        if (name):
            
            fig = self.canvas_3d.figure
            fig.savefig(name)
            
    def plot_3d(self):
        
        disorder_index = self.comboBox_correlations_3d.currentIndex()    
        disorder = self.comboBox_correlations_3d.itemText(disorder_index)
        
        correlation_index = self.comboBox_plot_3d.currentIndex()    
        correlation = self.comboBox_plot_3d.itemText(correlation_index)
        
        index = self.comboBox_norm_3d.currentIndex()    
        norm = self.comboBox_norm_3d.itemText(index)
        
        if (correlation == 'Correlation'):
            data = self.S_corr3d
            vmin = -1.0
            cmap = plt.cm.bwr
        else:
            data = self.S_coll3d
            vmin = 0.0
            cmap = plt.cm.binary
            
        if (norm == 'Logarithmic'):
            normalize = colors.SymLogNorm(linthresh=0.1, 
                                          linscale=1.0*(1-10**-1),
                                          base=10,
                                          vmin=vmin, 
                                          vmax=1.0)
        else:
            normalize = colors.Normalize(vmin=vmin, vmax=1.0)
            
        self.mask_plane()
        
        fig = self.canvas_3d.figure
        fig.clear()   
        
        ax = fig.add_subplot(111)
                
        if (self.checkBox_average_3d.isChecked()):   
            s = ax.scatter(self.D0, 
                           self.D1, 
                           c=data[self.plane], 
                           norm=normalize, 
                           cmap=cmap)
        else:
            for i in range(self.tableWidget_pairs_3d.rowCount()):
                if (self.tableWidget_pairs_3d.cellWidget(i, 2).isChecked()):
                    atom = self.tableWidget_pairs_3d.item(i, 0).text()                
                    if (atom == 'self-correlation'):
                        mask = self.atm_pair3d[self.plane] == '0'
                    else:
                        pair = self.tableWidget_pairs_3d.item(i, 1).text()  
                        mask = self.atm_pair3d[self.plane] == atom+'_'+pair
                    s = ax.scatter(self.D0[mask], 
                                   self.D1[mask], 
                                   c=data[self.plane][mask], 
                                   norm=normalize, 
                                   cmap=cmap)
                    
        try:
            s
        except:
            s = ax.scatter(0, 0, c=0, norm=normalize, cmap=cmap)            

        cb = fig.colorbar(s, format='%.1f')
        cb.ax.minorticks_on()
        
        if (norm == 'Logarithmic'):
            cb.locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=10)
            cb.update_ticks()
            if (correlation == 'Correlation'):
                minorticks = np.concatenate((s.norm(np.linspace(-1, -0.1, 11)), 
                                             s.norm(np.linspace(-0.1, 0.1, 21)), 
                                             s.norm(np.linspace(0.1, 1, 11))))
                cb.ax.yaxis.set_ticks(2*minorticks-1, minor=True)
            else:
                minorticks = np.concatenate((s.norm(np.linspace(0, 0.1, 11)), 
                                             s.norm(np.linspace(0.1, 1, 11))))
                cb.ax.yaxis.set_ticks(minorticks, minor=True)

        ax.set_aspect(1.0)
        
        h = np.int(self.lineEdit_plane_h.text())
        k = np.int(self.lineEdit_plane_k.text())
        l = np.int(self.lineEdit_plane_l.text())
        d = np.float(self.lineEdit_plane_d.text())
        
        scale = np.gcd.reduce([h, k, l])
        
        if (scale != 0):
            h, k, l = np.array([h, k, l]) // scale

        if (h >= 0):
            H = str(h)
        else:
            H = r'\bar\{'+str(np.abs(h))+'}'
        if (k >= 0):
            K = str(k)
        else:
            K = r'\bar{'+str(np.abs(k))+'}'
        if (l >= 0):
            L = str(l)
        else:
            L = r'\bar{'+str(np.abs(l))+'}'
            
        ax.set_title(r'$('+H+K+L+')\cdot[uvw]='+str(d)+'$', fontsize='small')

        ax.minorticks_on()
        
        ax.set_aspect(self.cor_aspect)
        
        uvw = np.array(['u','v','w'])
        
        var0 = np.repeat(uvw[np.argwhere(np.isclose(self.proj0,1))[0][0]],3)
        var1 = np.repeat(uvw[np.argwhere(np.isclose(self.proj1,1))[0][0]],3)
        
        coeff0 = np.round(self.proj0,4).astype(str)
        coeff1 = np.round(self.proj1,4).astype(str)
        
        for c in range(3):
            if (np.isclose(np.float(coeff0[c]),0)):
                coeff0[c] = '0'
                var0[c] = ''
            elif (np.isclose(np.float(coeff0[c]),1)):
                coeff0[c] = ''
            elif (np.isclose(np.float(coeff0[c]),-1)):
                coeff0[c] = '-'
                
            if (np.isclose(np.float(coeff1[c]),0)):
                coeff1[c] = '0'            
                var1[c] = ''            
            elif (np.isclose(np.float(coeff1[c]),1)):
                coeff1[c] = ''
            elif (np.isclose(np.float(coeff1[c]),-1)):
                coeff1[c] = '-'
                
        ax.set_xlabel(r'$['+coeff0[0]+var0[0]+','\
                           +coeff0[1]+var0[1]+','\
                           +coeff0[2]+var0[2]+']$')
                    
        ax.set_ylabel(r'$['+coeff1[0]+var1[0]+','\
                           +coeff1[1]+var1[1]+','\
                           +coeff1[2]+var1[2]+']$')

        if (correlation == 'Correlation'):
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
            if (disorder == 'Moment'):
                label = r'$\langle|\mathbf{S}(\mathbf{0})'\
                        r'\cdot\mathbf{S}(\mathbf{r})|^2\rangle$'
            elif (disorder == 'Occupancy'):
                label = r'$\langle|\sigma(\mathbf{0})'\
                        r'\cdot\sigma(\mathbf{r})|^2\rangle$'
            else:
                label = r'$\langle|\hat{\mathbf{u}}(\mathbf{0})'\
                        r'\cdot\hat{\mathbf{u}}(\mathbf{r})|^2\rangle$'
                        
        cb.set_label(label)

        ax.axes.tick_params(labelsize='small')
        
        fig.tight_layout(pad=3.24)
               
        with np.errstate(invalid='ignore'):
            self.canvas_3d.draw()
        
    def mask_plane(self):
        
        h = np.float(self.lineEdit_plane_h.text())
        k = np.float(self.lineEdit_plane_k.text())
        l = np.float(self.lineEdit_plane_l.text())
        d = np.float(self.lineEdit_plane_d.text())

        tol = np.float(self.lineEdit_tol_3d.text())
        
        B = self.B
                
        hx, hy, hz = np.dot(B, [h,k,l])
            
        if (not np.isclose(hx**2+hy**2+hz**2,0)):
            
            nx, ny, nz = [hx,hy,hz]/np.linalg.norm([hx,hy,hz])
            
            Px, Py, Pz = np.cross([0,0,1], [nx,ny,nz])
            P = np.linalg.norm([Px,Py,Pz])
            
            if (np.isclose(P,0)):
                Px, Py, Pz = np.cross([0,1,0], [nx,ny,nz])
                P = np.linalg.norm([Px,Py,Pz])            
            elif (np.isclose(np.max([Px,Py,Pz]),0)):
                Px, Py, Pz = np.cross([1,0,0], [nx,ny,nz])
                P = np.linalg.norm([Px,Py,Pz])
                
            px, py, pz = Px/P, Py/P, Pz/P
    
            Qx, Qy, Qz = np.cross([nx,ny,nz], [px,py,pz])
            Q = np.linalg.norm([Qx,Qy,Qz])                          
    
            qx, qy, qz = Qx/Q, Qy/Q, Qz/Q

            self.plane = np.isclose(hx*self.dx+hy*self.dy+hz*self.dz,
                                    d,
                                    rtol=tol)
            
            A = self.A
            
            A_inv = np.linalg.inv(A)
             
            pu, pv, pw = np.dot(A_inv, [px,py,pz])
            qu, qv, qw = np.dot(A_inv, [qx,qy,qz])
            
            proj0 = np.array([pu,pv,pw])
            proj1 = np.array([qu,qv,qw])
                                    
            scale_D0 = proj0.max()
            scale_D1 = proj1.max()
            
            self.proj0 = proj0/scale_D0
            self.proj1 = proj1/scale_D1
            
            self.cor_aspect = scale_D0/scale_D1
          
            self.D0 = (px*self.dx[self.plane]\
                    +  py*self.dy[self.plane]\
                    +  pz*self.dz[self.plane])*scale_D0
            
            self.D1 = (qx*self.dx[self.plane]\
                    +  qy*self.dy[self.plane]\
                    +  qz*self.dz[self.plane])*scale_D1
                
    def update_plot_1d(self):
        
        if (self.allocated or self.progress > 0):
            
            try:
                self.atm_pair
                self.plot_1d()
            except:
                pass
                        
    def save_1d(self):
             
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        
        if (name):
            
            fig = self.canvas_1d.figure
            fig.savefig(name)

    def plot_1d(self):
        
        disorder_index = self.comboBox_correlations_1d.currentIndex()    
        disorder = self.comboBox_correlations_1d.itemText(disorder_index)
        
        correlation_index = self.comboBox_plot_1d.currentIndex()
        correlation = self.comboBox_plot_1d.itemText(correlation_index)
        
        index = self.comboBox_norm_1d.currentIndex()    
        norm = self.comboBox_norm_1d.itemText(index)
        
        if (correlation == 'Correlation'):
            data = self.S_corr
            error = self.sigma_sq_corr
        else:
            data = self.S_coll
            error = self.sigma_sq_coll
            
        fig = self.canvas_1d.figure
        fig.clear()   
        
        ax = fig.add_subplot(111)

        if (correlation == 'Correlation'):        
            ax.axhline(y=0, 
                       xmin=0, 
                       xmax=1, 
                       color='k', 
                       linestyle='-', 
                       linewidth=1)
            
        if (self.checkBox_average_1d.isChecked()):   
            ax.scatter(self.d, data, marker='o', clip_on=False, zorder=50)
        else:
            count = 0
            for i in range(self.tableWidget_pairs_1d.rowCount()):
                if (self.tableWidget_pairs_1d.cellWidget(i, 2).isChecked()):
                    atom = self.tableWidget_pairs_1d.item(i, 0).text()
                    if (atom == r'self-correlation'):
                        mask = self.atm_pair == '0'
                        label = atom
                    else:
                        pair = self.tableWidget_pairs_1d.item(i, 1).text()  
                        mask = self.atm_pair == atom+'_'+pair
                        atom0 = atom.strip(pm).strip(numbers)
                        atom1 = pair.strip(pm).strip(numbers)
                        pre0, post0 = atom.split(atom0)
                        pre1, post1 = pair.split(atom1)
                        label = r'$^{{{}}}${}$^{{{}}}$-'\
                                r'$^{{{}}}${}$^{{{}}}$'.format(pre0,
                                                               atom0,
                                                               post0,
                                                               pre1,
                                                               atom1,
                                                               post1)
                    ax.errorbar(self.d[mask], 
                                data[mask], 
                                yerr=1.96*np.sqrt(error[mask]),
                                fmt='o', 
                                ls='none',
                                clip_on=False, 
                                zorder=50, 
                                label=label)
                    count += 1
            if (count > 0):
                ax.legend(loc='best', 
                          frameon=True, 
                          fancybox=True, 
                          fontsize='small')        
            #ax.legend(loc='center left', 
            #          bbox_to_anchor=(1, 0.5), 
            #          frameon=True, 
            #          fancybox=True, 
            #          fontsize='small')              
        ax.minorticks_on()
       
        if (norm == 'Logarithmic'):
            ax.set_yscale('symlog', linthresh=0.1, linscale=(1-10**-1))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_locator(MinorSymLogLocator(1e-1))
        
        x1,x2,y1,y2 = ax.axis()
        ax.set_xlim([0,x2])

        if (correlation == 'Correlation'):                
            ax.set_ylim([-1,1])
        else:
            ax.set_ylim([0,1])
            
        ax.set_xlabel(r'$r$ [Å]')
        
        if (correlation == 'Correlation'):
            if (disorder == 'Moment'):
                label = r'$\langle\mathbf{S}(0)\cdot\mathbf{S}(r)\rangle$'
            elif (disorder == 'Occupancy'):
                label = r'$\langle\sigma(0)\cdot\sigma(r)\rangle$'
            else:
                label = r'$\langle\hat{\mathbf{u}}(0)'\
                        r'\cdot\hat{\mathbf{u}}(r)\rangle$'
        else:
            if (disorder == 'Moment'):
                label = r'$\langle|\mathbf{S}(0)\cdot\mathbf{S}(r)|^2\rangle$'
            elif (disorder == 'Occupancy'):
                label = r'$\langle|\sigma(0)\cdot\sigma(r)|^2\rangle$'
            else:
                label = r'$\langle|\hat{\mathbf{u}}(0)'\
                        r'\cdot\hat{\mathbf{u}}(r)|^2\rangle$'            

        ax.set_ylabel(label)
        ax.axes.tick_params(labelsize='small')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout(pad=3.24)
               
        with np.errstate(invalid='ignore'):
            self.canvas_1d.draw()
    
    def recalculate_correlations_3d_thread_complete(self):
                
        self.pushButton_calculate_3d.setEnabled(True)
        
        self.recreate_table_3d() 
                                
        self.plot_3d()
        
    def calculate_correlations_3d_thread_complete(self):
                
        self.pushButton_calculate_3d.setEnabled(True)
        
        self.recreate_table_3d() 
                        
        self.plot_3d()
        
    def recalculate_correlations_3d(self):
        
        if (self.tableWidget_pairs_3d.rowCount() > 0):
                        
            disorder_index = self.comboBox_correlations_3d.currentIndex()    
            disorder = self.comboBox_correlations_3d.itemText(disorder_index)
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):
                
                self.pushButton_calculate_3d.setEnabled(False)
                                            
                worker = Worker(self.calculate_correlations_3d_thread)
                worker.signals.finished.connect(
                              self.recalculate_correlations_3d_thread_complete)
                
                self.threadpool.start(worker) 
            
    def calculate_correlations_3d(self):
        
        if (self.tableWidget_pairs_3d.rowCount() == 0):
            
            disorder_index = self.comboBox_correlations_3d.currentIndex()    
            disorder = self.comboBox_correlations_3d.itemText(disorder_index)
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):

                self.pushButton_calculate_3d.setEnabled(False)
                                            
                worker = Worker(self.calculate_correlations_3d_thread)
                worker.signals.finished.connect(
                                self.calculate_correlations_3d_thread_complete)
                
                self.threadpool.start(worker) 
        
    def calculate_correlations_3d_thread(self, callback):
                
        disorder_index = self.comboBox_correlations_3d.currentIndex()    
        disorder = self.comboBox_correlations_3d.itemText(disorder_index)
 
        fract = np.float(self.lineEdit_fract_3d.text())
        tol = np.float(self.lineEdit_tol_3d.text())
        
        runs = np.int(self.lineEdit_runs_corr_3d.text())
                   
        for run in range(runs):               

            if (self.checkBox_batch_corr_3d.isChecked()):
                r = '-'+str(run)
            else:
                r = ''  
        
            if (disorder == 'Moment'):
                
                Sx = np.load(self.fname+'-calculated-spin-x'+r+'.npy')
                Sy = np.load(self.fname+'-calculated-spin-y'+r+'.npy')
                Sz = np.load(self.fname+'-calculated-spin-z'+r+'.npy')
                                    
                S_corr3d, \
                S_coll3d, \
                S_corr3d_, \
                S_coll3d_, \
                self.dx, \
                self.dy, \
                self.dz, \
                self.atm_pair3d = correlations.radial3d(Sx, 
                                                        Sy, 
                                                        Sz, 
                                                        self.rx, 
                                                        self.ry, 
                                                        self.rz, 
                                                        self.atms, 
                                                        fract=fract, 
                                                        tol=tol,
                                                        period=(self.A,
                                                                self.nu,
                                                                self.nv,
                                                                self.nw,
                                                                self.n_atm))
                    
            elif (disorder == 'Occupancy'):
                
                A_r = np.load(self.fname\
                                   +'-calculated-composition'+r+'.npy')
                                    
                S_corr3d, \
                S_corr3d_, \
                self.dx, \
                self.dy, \
                self.dz, \
                self.atm_pair3d = correlations.parameter3d(A_r, 
                                                           self.rx, 
                                                           self.ry, 
                                                           self.rz, 
                                                           self.atms, 
                                                           fract=fract, 
                                                           tol=tol,
                                                           period=(self.A,
                                                                   self.nu,
                                                                   self.nv,
                                                                   self.nw,
                                                                   self.n_atm))    
                
            elif (disorder == 'Displacement'):
                
                Ux = np.load(self.fname\
                             +'-calculated-displacement-x'+r+'.npy')
                Uy = np.load(self.fname\
                             +'-calculated-displacement-y'+r+'.npy')
                Uz = np.load(self.fname\
                             +'-calculated-displacement-z'+r+'.npy')
                                
                S_corr3d, \
                S_coll3d, \
                S_corr3d_, \
                S_coll3d_, \
                self.dx, \
                self.dy, \
                self.dz, \
                self.atm_pair3d = correlations.radial3d(Ux, 
                                                        Uy, 
                                                        Uz, 
                                                        self.rx, 
                                                        self.ry, 
                                                        self.rz, 
                                                        self.atms, 
                                                        fract=fract, 
                                                        tol=tol,
                                                        period=(self.A,
                                                                self.nu,
                                                                self.nv,
                                                                self.nw,
                                                                self.n_atm))
            
            if (run == 0):
                
                s_corr3d = np.zeros((S_corr3d.shape[0],runs))
                s_corr3d_ = np.zeros((S_corr3d_.shape[0],runs))
                
                if (disorder != 'Occupancy'):
                    
                    s_coll3d = np.zeros((S_coll3d.shape[0],runs))
                    s_coll3d_ = np.zeros((S_coll3d_.shape[0],runs))
                
            s_corr3d[:,run] = S_corr3d.copy()
            s_corr3d_[:,run] = S_corr3d_.copy()

            if (disorder != 'Occupancy'):
                
                s_coll3d[:,run] = S_coll3d.copy()
                s_coll3d_[:,run] = S_coll3d_.copy()
     
        self.S_corr3d = np.mean(s_corr3d, axis=1)
        self.S_corr3d_ = np.mean(s_corr3d_, axis=1)
        
        self.sigma_sq_corr3d = np.std(s_corr3d, axis=1)**2/runs
        self.sigma_sq_corr3d_ = np.std(s_corr3d_, axis=1)**2/runs
        
        if (disorder != 'Occupancy'):
            
            self.S_coll3d = np.mean(s_coll3d, axis=1)
            self.S_coll3d_ = np.mean(s_coll3d_, axis=1)
            
            self.sigma_sq_coll3d = np.std(s_coll3d, axis=1)**2/runs
            self.sigma_sq_coll3d_ = np.std(s_coll3d_, axis=1)**2/runs
            
        if (self.checkBox_symmetrize.isChecked()):

            if (disorder == 'Occupancy'):

                self.S_corr3d, \
                self.S_corr3d_, \
                self.sigma_sq_corr3d, \
                self.sigma_sq_corr3d_, \
                self.dx, \
                self.dy, \
                self.dz, \
                self.atm_pair3d = crystal.symmetrize((self.S_corr3d, 
                                                      self.S_corr3d_,
                                                      self.sigma_sq_corr3d,
                                                      self.sigma_sq_corr3d_),
                                                      self.dx, 
                                                      self.dy, 
                                                      self.dz, 
                                                      self.atm_pair3d, 
                                                      self.A, 
                                                      folder=self.folder, 
                                                      filename=self.filename, 
                                                      tol=tol)
            else:
                                    
                self.S_corr3d, \
                self.S_coll3d, \
                self.S_corr3d_, \
                self.S_coll3d_, \
                self.sigma_sq_corr3d, \
                self.sigma_sq_coll3d, \
                self.sigma_sq_corr3d_, \
                self.sigma_sq_coll3d_, \
                self.dx, \
                self.dy, \
                self.dz, \
                self.atm_pair3d = crystal.symmetrize((self.S_corr3d, 
                                                      self.S_coll3d, 
                                                      self.S_corr3d_, 
                                                      self.S_coll3d_,
                                                      self.sigma_sq_corr3d, 
                                                      self.sigma_sq_coll3d, 
                                                      self.sigma_sq_corr3d_, 
                                                      self.sigma_sq_coll3d_),
                                                      self.dx, 
                                                      self.dy, 
                                                      self.dz, 
                                                      self.atm_pair3d, 
                                                      self.A, 
                                                      folder=self.folder, 
                                                      filename=self.filename, 
                                                      tol=tol)

        if (self.checkBox_average_3d.isChecked()):
            
            if (disorder == 'Occupancy'):
            
                self.S_corr3d, \
                self.S_corr3d_, \
                self.sigma_sq_corr3d, \
                self.sigma_sq_corr3d_, \
                self.dx, \
                self.dy, \
                self.dz = crystal.average3d((self.S_corr3d, 
                                             self.S_corr3d_,
                                             self.sigma_sq_corr3d,
                                             self.sigma_sq_corr3d_),
                                             self.dx, 
                                             self.dy, 
                                             self.dz, 
                                             tol=tol)

            else:
                    
                self.S_corr3d, \
                self.S_coll3d, \
                self.S_corr3d_, \
                self.S_coll3d_,\
                self.sigma_sq_corr3d, \
                self.sigma_sq_coll3d, \
                self.sigma_sq_corr3d_, \
                self.sigma_sq_coll3d_, \
                self.dx, \
                self.dy, \
                self.dz = crystal.average3d((self.S_corr3d, 
                                             self.S_coll3d, 
                                             self.S_corr3d_, 
                                             self.S_coll3d_,
                                             self.sigma_sq_corr3d, 
                                             self.sigma_sq_coll3d, 
                                             self.sigma_sq_corr3d_, 
                                             self.sigma_sq_coll3d_),
                                             self.dx, 
                                             self.dy, 
                                             self.dz, 
                                             tol=tol)
                
        
    def recreate_table_3d(self):
                
        unique_pairs = np.unique(self.atm_pair3d)
        
        rows = self.tableWidget_pairs_3d.rowCount()
        
        pairs = []
        for i in range(rows):
            left = self.tableWidget_pairs_3d.cellWidget(i, 0)
            right = self.tableWidget_pairs_3d.cellWidget(i, 1)
            if (left is not None):
                if (left.text() == 'self-correlation'):
                    pairs.append('0')
                else:
                    pairs.append(left.text()+'_'+right.text())
                        
        if (rows == 0 or unique_pairs.tolist() != np.unique(pairs).tolist()):
            
            self.tableWidget_pairs_3d.setRowCount(unique_pairs.size)
            self.tableWidget_pairs_3d.setColumnCount(3)
            
            lbl = 'atom,pair, '
            lbl = lbl.split(',')
            self.tableWidget_pairs_3d.setHorizontalHeaderLabels(lbl)

            lbl = ['%d' % (s+1) for s in range(unique_pairs.size)]
            self.tableWidget_pairs_3d.setVerticalHeaderLabels(lbl)
                       
            for i in range(unique_pairs.size):
                uni = unique_pairs[i]
                if (uni == '0'):
                    uni = 'self-correlation'
                    self.tableWidget_pairs_3d.setItem(i, 0, 
                                               QtWidgets.QTableWidgetItem(uni))
                    self.tableWidget_pairs_3d.setItem(i, 1, 
                                               QtWidgets.QTableWidgetItem(' '))
                else:
                    left, right = uni.split('_')
                    self.tableWidget_pairs_3d.setItem(i, 0, 
                                              QtWidgets.QTableWidgetItem(left))
                    self.tableWidget_pairs_3d.setItem(i, 1, 
                                             QtWidgets.QTableWidgetItem(right))
                check = QtWidgets.QCheckBox()
                check.setObjectName('checkBox_pair_3d_'+str(i))
                if (self.checkBox_average_3d.isChecked()):   
                    check.setCheckState(QtCore.Qt.Unchecked)
                    check.setEnabled(False)
                else:
                    check.setCheckState(QtCore.Qt.Checked)                     
                check.clicked.connect(self.plot_3d)
                self.tableWidget_pairs_3d.setCellWidget(i, 2, check)
 
            self.tableWidget_pairs_3d.setSpan(0, 0, 1, 2)          
            self.tableWidget_pairs_3d.item(0, 0).setTextAlignment(alignment)
            self.tableWidget_pairs_3d.item(0, 0).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    
            for i in range(1, unique_pairs.size):
                for j in range(2):
                    self.tableWidget_pairs_3d.item(i, 
                                                 j).setTextAlignment(alignment)
                    self.tableWidget_pairs_3d.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
 
            self.tableWidget_pairs_3d.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
        else:
            
            sender = self.sender()
            
            for i in range(unique_pairs.size):
                check = self.tableWidget_pairs_3d.cellWidget(i, 2)
                if (self.checkBox_average_3d.isChecked()):   
                    check.setCheckState(QtCore.Qt.Unchecked)
                    check.setEnabled(False)       
                elif (sender.text() == 'Average'):
                    check.setCheckState(QtCore.Qt.Checked)  
                    check.setEnabled(True)       
                check.clicked.connect(self.plot_3d)
 
            self.tableWidget_pairs_3d.setSpan(0, 0, 1, 2)          
            self.tableWidget_pairs_3d.item(0, 0).setTextAlignment(alignment)
            self.tableWidget_pairs_3d.item(0, 0).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                
            for i in range(1, unique_pairs.size):
                for j in range(2):
                    self.tableWidget_pairs_3d.item(i, 
                                                 j).setTextAlignment(alignment)
                    self.tableWidget_pairs_3d.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
 
            self.tableWidget_pairs_3d.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
    def recalculate_correlations_1d_thread_complete(self):
                
        self.pushButton_calculate_1d.setEnabled(True)
        
        self.recreate_table_1d() 
                                
        self.plot_1d()
        
    def calculate_correlations_1d_thread_complete(self):
                
        self.pushButton_calculate_1d.setEnabled(True)
        
        self.recreate_table_1d() 
                        
        self.plot_1d()
        
    def recalculate_correlations_1d(self):
        
        if (self.tableWidget_pairs_1d.rowCount() > 0):
                        
            disorder_index = self.comboBox_correlations_1d.currentIndex()    
            disorder = self.comboBox_correlations_1d.itemText(disorder_index)
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):
                
                self.pushButton_calculate_1d.setEnabled(False)
                                            
                worker = Worker(self.calculate_correlations_1d_thread)
                worker.signals.finished.connect(
                              self.recalculate_correlations_1d_thread_complete)
                
                self.threadpool.start(worker) 
            
    def calculate_correlations_1d(self):
        
        if (self.tableWidget_pairs_1d.rowCount() == 0):
                                
            disorder_index = self.comboBox_correlations_1d.currentIndex()    
            disorder = self.comboBox_correlations_1d.itemText(disorder_index)
            
            aligned = (disorder == 'Moment' and self.magnetic) or \
                      (disorder == 'Occupancy' and self.occupational) or \
                      (disorder == 'Displacement' and self.displacive)
                       
            if (self.progress > 0 and self.allocated and aligned):
                
                self.pushButton_calculate_1d.setEnabled(False)
                                            
                worker = Worker(self.calculate_correlations_1d_thread)
                worker.signals.finished.connect(
                                self.calculate_correlations_1d_thread_complete)
                
                self.threadpool.start(worker) 
        
    def calculate_correlations_1d_thread(self, callback):
                
        disorder_index = self.comboBox_correlations_1d.currentIndex()    
        disorder = self.comboBox_correlations_1d.itemText(disorder_index)
                            
        fract = np.float(self.lineEdit_fract_1d.text())
        tol = np.float(self.lineEdit_tol_1d.text())
       
        runs = np.int(self.lineEdit_runs_corr_1d.text())
                   
        for run in range(runs):               

            if (self.checkBox_batch_corr_1d.isChecked()):
                r = '-'+str(run)
            else:
                r = ''  
        
            if (disorder == 'Moment'):
                
                Sx = np.load(self.fname+'-calculated-spin-x'+r+'.npy')
                Sy = np.load(self.fname+'-calculated-spin-y'+r+'.npy')
                Sz = np.load(self.fname+'-calculated-spin-z'+r+'.npy')
                            
                S_corr, \
                S_coll, \
                S_corr_, \
                S_coll_, \
                self.d, \
                self.atm_pair = correlations.radial(Sx, 
                                                    Sy, 
                                                    Sz, 
                                                    self.rx, 
                                                    self.ry, 
                                                    self.rz, 
                                                    self.atms, 
                                                    fract=fract,
                                                    tol=tol,
                                                    period=(self.A,
                                                            self.nu,
                                                            self.nv,
                                                            self.nw,
                                                            self.n_atm))

            elif (disorder == 'Occupancy'):
                
                A_r = np.load(self.fname\
                                   +'-calculated-composition'+r+'.npy')
                   
                S_corr, \
                S_corr_, \
                self.d, \
                self.atm_pair = correlations.parameter(A_r, 
                                                       self.rx, 
                                                       self.ry, 
                                                       self.rz, 
                                                       self.atms, 
                                                       fract=fract,
                                                       tol=tol,
                                                       period=(self.A,
                                                               self.nu,
                                                               self.nv,
                                                               self.nw,
                                                               self.n_atm))
            
            else:
                
                Ux = np.load(self.fname\
                             +'-calculated-displacement-x'+r+'.npy')
                Uy = np.load(self.fname\
                             +'-calculated-displacement-y'+r+'.npy')
                Uz = np.load(self.fname\
                             +'-calculated-displacement-z'+r+'.npy')    
            
                S_corr, \
                S_coll, \
                S_corr_, \
                S_coll_, \
                self.d, \
                self.atm_pair = correlations.radial(Ux, 
                                                    Uy, 
                                                    Uz, 
                                                    self.rx, 
                                                    self.ry, 
                                                    self.rz, 
                                                    self.atms, 
                                                    fract=fract,
                                                    tol=tol,
                                                    period=(self.A,
                                                            self.nu,
                                                            self.nv,
                                                            self.nw,
                                                            self.n_atm))
                
         
            if (run == 0):
                
                s_corr = np.zeros((S_corr.shape[0],runs))
                s_corr_ = np.zeros((S_corr_.shape[0],runs))

                if (disorder != 'Occupancy'):
                    
                    s_coll = np.zeros((S_coll.shape[0],runs))
                    s_coll_ = np.zeros((S_coll_.shape[0],runs))
                
            s_corr[:,run] = S_corr.copy()
            s_corr_[:,run] = S_corr_.copy()

            if (disorder != 'Occupancy'):
                
                s_coll[:,run] = S_coll.copy()
                s_coll_[:,run] = S_coll_.copy()
     
        self.S_corr = np.mean(s_corr, axis=1)
        self.S_corr_ = np.mean(s_corr_, axis=1)
        
        self.sigma_sq_corr = np.std(s_corr, axis=1)**2/runs
        self.sigma_sq_corr_ = np.std(s_corr_, axis=1)**2/runs
        
        if (disorder != 'Occupancy'):
            
            self.S_coll = np.mean(s_coll, axis=1)                
            self.S_coll_ = np.mean(s_coll_, axis=1)
            
            self.sigma_sq_coll = np.std(s_coll, axis=1)**2/runs                
            self.sigma_sq_coll_ = np.std(s_coll_, axis=1)**2/runs
                        
        if (self.checkBox_average_1d.isChecked()):
            
            if (disorder == 'Occupancy'):

                self.S_corr, \
                self.S_corr_, \
                self.sigma_sq_corr, \
                self.sigma_sq_corr_, \
                self.d = crystal.average((self.S_corr, 
                                          self.S_corr_,
                                          self.sigma_sq_corr,
                                          self.sigma_sq_corr_), 
                                          self.d, 
                                          tol=tol)

            else:

                self.S_corr, \
                self.S_coll, \
                self.S_corr_, \
                self.S_coll_, \
                self.sigma_sq_corr, \
                self.sigma_sq_coll, \
                self.sigma_sq_corr_, \
                self.sigma_sq_coll_, \
                self.d = crystal.average((self.S_corr, 
                                          self.S_coll,
                                          self.S_corr_,
                                          self.S_coll_,
                                          self.sigma_sq_corr,
                                          self.sigma_sq_coll,
                                          self.sigma_sq_corr_,
                                          self.sigma_sq_coll_), 
                                          self.d,
                                          tol=tol)

    def recreate_table_1d(self):
        
        unique_pairs = np.unique(self.atm_pair)
        
        rows = self.tableWidget_pairs_1d.rowCount()
        
        pairs = []
        for i in range(rows):
            left = self.tableWidget_pairs_1d.cellWidget(i, 0)
            right = self.tableWidget_pairs_1d.cellWidget(i, 1)
            if (left is not None):
                if (left.text() == 'self-correlation'):
                    pairs.append('0')
                else:
                    pairs.append(left.text()+'_'+right.text())
        
        if (rows == 0 or unique_pairs.tolist() != np.unique(pairs).tolist()):
            
            self.tableWidget_pairs_1d.setRowCount(unique_pairs.size)
            self.tableWidget_pairs_1d.setColumnCount(3)
            
            lbl = 'atom,pair, '
            lbl = lbl.split(',')
            self.tableWidget_pairs_1d.setHorizontalHeaderLabels(lbl)

            lbl = ['%d' % (s+1) for s in range(unique_pairs.size)]
            self.tableWidget_pairs_1d.setVerticalHeaderLabels(lbl)
                       
            for i in range(unique_pairs.size):
                uni = unique_pairs[i]
                if (uni == '0'):
                    uni = 'self-correlation'
                    self.tableWidget_pairs_1d.setItem(i, 0, 
                                               QtWidgets.QTableWidgetItem(uni))
                    self.tableWidget_pairs_1d.setItem(i, 1, 
                                               QtWidgets.QTableWidgetItem(' '))
                else:
                    left, right = uni.split('_')
                    self.tableWidget_pairs_1d.setItem(i, 0, 
                                              QtWidgets.QTableWidgetItem(left))
                    self.tableWidget_pairs_1d.setItem(i, 1, 
                                             QtWidgets.QTableWidgetItem(right))
                check = QtWidgets.QCheckBox()
                check.setObjectName('checkBox_pair_1d_'+str(i))
                if (self.checkBox_average_1d.isChecked()):   
                    check.setCheckState(QtCore.Qt.Unchecked)
                    check.setEnabled(False)
                else:
                    check.setCheckState(QtCore.Qt.Checked)  
                check.clicked.connect(self.plot_1d)
                self.tableWidget_pairs_1d.setCellWidget(i, 2, check)
 
            self.tableWidget_pairs_1d.setSpan(0, 0, 1, 2)          
            self.tableWidget_pairs_1d.item(0, 0).setTextAlignment(alignment)
            self.tableWidget_pairs_1d.item(0, 0).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    
            for i in range(1, unique_pairs.size):
                for j in range(2):
                    self.tableWidget_pairs_1d.item(i, 
                                                 j).setTextAlignment(alignment)
                    self.tableWidget_pairs_1d.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
 
            self.tableWidget_pairs_1d.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
        else:
            
            sender = self.sender()
            
            for i in range(unique_pairs.size):
                check = self.tableWidget_pairs_1d.cellWidget(i, 2)
                if (self.checkBox_average_1d.isChecked()):   
                    check.setCheckState(QtCore.Qt.Unchecked)
                    check.setEnabled(False)    
                elif (sender.text() == 'Average'):
                    check.setCheckState(QtCore.Qt.Checked)  
                    check.setEnabled(True)       
                check.clicked.connect(self.plot_1d)
                
            self.tableWidget_pairs_1d.setSpan(0, 0, 1, 2)          
            self.tableWidget_pairs_1d.item(0, 0).setTextAlignment(alignment)
            self.tableWidget_pairs_1d.item(0, 0).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    
            for i in range(1, unique_pairs.size):
                for j in range(2):
                    self.tableWidget_pairs_1d.item(i, 
                                                 j).setTextAlignment(alignment)
                    self.tableWidget_pairs_1d.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
 
            self.tableWidget_pairs_1d.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
            
    def change_runs_corr_1d(self):
        
        try:
            runs = np.int(self.lineEdit_runs_corr_1d.text())
        except:
            runs = 1
            
        self.lineEdit_runs_corr_1d.setText(str(runs))
        
    def check_batch_corr_1d(self):
        
        if (self.checkBox_batch_corr_1d.isChecked()):
            self.lineEdit_runs_corr_1d.setEnabled(True)
        else:
            self.lineEdit_runs_corr_1d.setEnabled(False)
            self.lineEdit_runs_corr_1d.setText('1')
            
    def change_runs_corr_3d(self):
        
        try:
            runs = np.int(self.lineEdit_runs_corr_3d.text())
        except:
            runs = 1
            
        self.lineEdit_runs_corr_3d.setText(str(runs))
        
    def check_batch_corr_3d(self):
        
        if (self.checkBox_batch_corr_3d.isChecked()):
            self.lineEdit_runs_corr_3d.setEnabled(True)
        else:
            self.lineEdit_runs_corr_3d.setEnabled(False)
            self.lineEdit_runs_corr_3d.setText('1')
                
    def change_prefactor(self):
        
        try:
            prefactor = np.float(self.lineEdit_prefactor.text())
        except:
            prefactor = 1e+4
            
        self.lineEdit_prefactor.setText('%1.2e' % prefactor)
        
        try:
            self.temperature.append(prefactor)
        except:
            self.temp = prefactor
        
    def change_tau(self):
        
        try:
            tau = np.float(self.lineEdit_tau.text())
        except:
            tau = 1e-3
            
        self.lineEdit_tau.setText('%1.2e' % tau)
        self.constant = tau
        
    def change_cycles(self):
        
        try:
            cycles = np.int(self.lineEdit_cycles.text())
        except:
            cycles = 1
            
        self.lineEdit_cycles.setText(str(cycles))
        self.reset_refinement()
        
    def change_runs(self):
        
        try:
            runs = np.int(self.lineEdit_runs.text())
        except:
            runs = 1
            
        self.lineEdit_runs.setText(str(runs))
        self.reset_refinement()
        
    def check_batch(self):
        
        if (self.checkBox_batch.isChecked()):
            self.lineEdit_runs.setEnabled(True)
        else:
            self.lineEdit_runs.setEnabled(False)
            self.lineEdit_runs.setText('1')
            
    def change_slice(self):
 
        if (self.allocated):

            index_hkl = self.comboBox_slice.currentIndex()    
            hkl = self.comboBox_slice.itemText(index_hkl)
                
            slice_hkl = np.float(self.lineEdit_slice.text())
                    
            if (hkl == 'h = '):
                step_h = np.float(self.tableWidget_exp.item(0, 0).text())
                min_h = np.float(self.tableWidget_exp.item(0, 2).text())
                max_h = np.float(self.tableWidget_exp.item(0, 3).text())
                if (slice_hkl < min_h):
                    h = min_h
                elif (slice_hkl > max_h):
                    h = max_h
                else:
                    ih = np.int(np.round((slice_hkl-min_h)/step_h))
                    h = np.round(min_h+step_h*ih, 4)
                self.lineEdit_slice.setText(str(h))
            elif (hkl == 'k = '):
                step_k = np.float(self.tableWidget_exp.item(1, 0).text())
                min_k = np.float(self.tableWidget_exp.item(1, 2).text())
                max_k = np.float(self.tableWidget_exp.item(1, 3).text())
                if (slice_hkl < min_k):
                    k = min_k
                elif (slice_hkl > max_k):
                    k = max_k
                else:
                    ik = np.int(np.round((slice_hkl-min_k)/step_k))
                    k = np.round(min_k+step_k*ik, 4)
                self.lineEdit_slice.setText(str(k))
            else:
                step_l = np.float(self.tableWidget_exp.item(2, 0).text())
                min_l = np.float(self.tableWidget_exp.item(2, 2).text())
                max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                if (slice_hkl < min_l):
                    l = min_l
                elif (slice_hkl > max_l):
                    l = max_l
                else:
                    il = np.int(np.round((slice_hkl-min_l)/step_l))
                    l = np.round(min_l+step_l*il, 4)            
                self.lineEdit_slice.setText(str(l))
        
            self.replot_intensity_ref()
        
    def plot_chi_sq(self):
        
        if (self.allocated or self.progress > 0):

            index0 = self.comboBox_plot_top_chi_sq.currentIndex()    
            plot0 = self.comboBox_plot_top_chi_sq.itemText(index0)
            
            index1 = self.comboBox_plot_bottom_chi_sq.currentIndex()    
            plot1 = self.comboBox_plot_bottom_chi_sq.itemText(index1)
            
            fig = self.canvas_chi_sq.figure
            fig.clear()   
            
            ax0 = fig.add_subplot(211)
            ax1 = fig.add_subplot(212)
            
            if (plot0 == 'Accepted'):
                ax0.semilogy(self.acc_moves, 'C0')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'Accepted $\chi^2$', fontsize='small')    
            elif (plot0 == 'Rejected'):
                ax0.semilogy(self.rej_moves, 'C0')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'Rejected $\chi^2$', fontsize='small') 
            elif (plot0 == 'Temperature'):              
                ax0.semilogy(self.temperature, 'C0')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'Temperatrue $T$', fontsize='small') 
            elif (plot0 == 'Energy'):              
                ax0.plot(self.energy, 'C0')
                ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'Energy $\Delta\chi^2$', fontsize='small') 
            elif (plot0 == 'Chi-squared'):              
                ax0.semilogy(self.chi_sq, 'C0')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'$\chi^2$', fontsize='small') 
            else:
                ax0.plot(self.scale, 'C0')
                ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                ax0.set_xlabel(r'Moves', fontsize='small')
                ax0.set_ylabel(r'Scale factor', fontsize='small') 
    
            if (plot1 == 'Accepted'):
                ax1.semilogy(self.acc_moves, 'C1')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'Accepted $\chi^2$', fontsize='small')    
            elif (plot1 == 'Rejected'):
                ax1.semilogy(self.rej_moves, 'C1')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'Rejected $\chi^2$', fontsize='small') 
            elif (plot1 == 'Temperature'):              
                ax1.semilogy(self.temperature, 'C1')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'Temperature $T$', fontsize='small') 
            elif (plot1 == 'Energy'):              
                ax1.plot(self.energy, 'C1')
                ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'Energy $\Delta\chi^2$', fontsize='small') 
            elif (plot1 == 'Chi-squared'):              
                ax1.semilogy(self.chi_sq, 'C1')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'$\chi^2$', fontsize='small') 
            else:
                ax1.plot(self.scale, 'C1')
                ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
                ax1.set_xlabel(r'Moves', fontsize='small')
                ax1.set_ylabel(r'Scale factor', fontsize='small') 
                
            ax0.axes.tick_params(labelsize='small')
            ax1.axes.tick_params(labelsize='small')
                   
            ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
            ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
            
            ax0.minorticks_on()
            ax1.minorticks_on()
            
            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
        
            #ax.set_title(r'\chi^2$', fontsize='small') 
            
            fig.tight_layout(pad=3.24)
                    
            with np.errstate(invalid='ignore'):
                self.canvas_chi_sq.draw()
        
    def save_intensity_chi_sq(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
                       
        if (name):
            
            fig = self.canvas_chi_sq.figure
            fig.savefig(name)

    def plot_intensity_ref(self, data):
        
        index = self.comboBox_norm_ref.currentIndex()    
        norm = self.comboBox_norm_ref.itemText(index)
        
        index_hkl = self.comboBox_slice.currentIndex()    
        hkl = self.comboBox_slice.itemText(index_hkl)
        
        slice_hkl = np.float(self.lineEdit_slice.text())
                
        step_h = np.float(self.tableWidget_exp.item(0, 0).text())
        size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))

        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        step_k = np.float(self.tableWidget_exp.item(1, 0).text())
        size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))

        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        step_l = np.float(self.tableWidget_exp.item(2, 0).text())
        size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))

        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        max_l = np.float(self.tableWidget_exp.item(2, 3).text())
        
        vmin = np.float(self.lineEdit_min_ref.text())
        vmax = np.float(self.lineEdit_max_ref.text())
        
        if (norm == 'Logarithmic'):
            normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            normalize = colors.Normalize(vmin=vmin, vmax=vmax)
                    
        fig = self.canvas_ref.figure
        fig.clear()   
        
        ax = fig.add_subplot(111)
        
        ax.set_aspect(1.)
        
        if (hkl == 'h ='):
                
            if (slice_hkl == 1):
                ih = 0
                h = min_h
            else:
                ih = np.int(np.round((slice_hkl-min_h)/step_h))
                h = np.round(min_h+step_h*ih,4)

            dk = np.float(max_k-min_k)/size_k
            dl = np.float(max_l-min_l)/size_l
            extents_h = [min_k-dk/2, max_k+dk/2, min_l-dl/2, max_l+dl/2]
        
            im = ax.imshow(data[ih,:,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_h)
            
            ax.set_title(r'$h='+str(h)+'$', fontsize='small') 
    
            ax.set_xlabel(r'$k$', fontsize='small')
            ax.set_ylabel(r'$l$', fontsize='small')
 
            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[1,1]/self.B[2,2],self.B[1,2]/self.B[2,2],0],
                          [self.B[2,1]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                          [0,0,1]])
            
            scale = M[0,0]
            
            M[0,1] /= scale
            M[0,0] /= scale
            
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_l,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_h[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_h[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])
            
        elif (hkl == 'k ='):
            
            if (slice_hkl == 1):
                ik = 0
                k = min_k
            else:
                ik = np.int(np.round((slice_hkl-min_k)/step_k))
                k = np.round(min_k+step_k*ik,4)  

            dh = np.float(max_h-min_h)/size_h
            dl = np.float(max_l-min_l)/size_l
            extents_k = [min_h-dh/2, max_h+dh/2, min_l-dl/2, max_l+dl/2]
            
            im = ax.imshow(data[:,ik,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_k)
        
            ax.set_title(r'$k='+str(k)+'$', fontsize='small')
            
            ax.set_xlabel(r'$h$', fontsize='small')
            ax.set_ylabel(r'$l$', fontsize='small')

            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[0,0]/self.B[2,2],self.B[0,2]/self.B[2,2],0],
                          [self.B[2,0]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                          [0,0,1]])
            
            scale = M[0,0]
            
            M[0,1] /= scale
            M[0,0] /= scale
            
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_l,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_k[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_k[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])
            
        else:
            
            if (slice_hkl == 1):
                il = 0
                l = min_l
            else:
                il = np.int(np.round((slice_hkl-min_l)/step_l))
                l = np.round(min_l+step_l*il,4) 

            dh = np.float(max_h-min_h)/size_h
            dk = np.float(max_k-min_k)/size_k
            extents_l = [min_h-dh/2, max_h+dh/2, min_k-dk/2, max_k+dk/2]
        
            im = ax.imshow(data[:,:,il].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_l)
            
            ax.set_title(r'$l='+str(l)+'$', fontsize='small') 
        
            ax.set_xlabel(r'$h$', fontsize='small') 
            ax.set_ylabel(r'$k$', fontsize='small') 
            
            trans = mtransforms.Affine2D()
            
            M = np.array([[self.B[0,0]/self.B[1,1],self.B[0,1]/self.B[1,1],0],
                          [self.B[1,0]/self.B[1,1],self.B[1,1]/self.B[1,1],0],
                          [0,0,1]])
            
            scale = M[0,0]
            
            M[0,1] /= scale
            M[0,0] /= scale
            
            trans.set_matrix(M)
            
            offset = -np.dot(M,[0,min_k,0])[0]
            
            shift = mtransforms.Affine2D().translate(offset,0)
            
            ax.set_aspect(1/scale)
            
            trans_data = trans+shift+ax.transData
            
            im.set_transform(trans_data)
            
            ext_min = np.dot(M[0:2,0:2],extents_l[0::2])
            ext_max = np.dot(M[0:2,0:2],extents_l[1::2])
            
            ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
            ax.set_ylim(ext_min[1],ext_max[1])
   
        ax.xaxis.tick_bottom()
        
        ax.minorticks_on()
        
        ax.axes.tick_params(labelsize='small')
    
        fig.tight_layout(pad=3.24)
        
        cb = fig.colorbar(im, ax=ax)
        cb.ax.minorticks_on()
        
        if (norm == 'Linear'):
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
            
        cb.ax.tick_params(labelsize='small') 

        with np.errstate(invalid='ignore'):
            self.canvas_ref.draw()
        
    def save_intensity_ref(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'pdf files *.pdf;;'+
                                                        'png files *.png',
                                                         options=options)
        
        if (name):
                
            fig = self.canvas_ref.figure
            fig.savefig(name)
        
    def change_min_ref(self):

        if (self.allocated or self.progress > 0):
            
            try:
                vmin = np.float(self.lineEdit_min_ref.text())
                
                if (vmin >= np.float(self.lineEdit_max_ref.text())):
                    raise
                    
                self.lineEdit_min_ref.setText('%1.4e' % vmin)
                
                index = self.comboBox_plot_ref.currentIndex()    
                plot = self.comboBox_plot_ref.itemText(index)
                
                if (plot == 'Calculated'):
                    self.plot_intensity_ref(self.I_obs)
                elif (plot == 'Experimental'):
                    self.plot_intensity_ref(self.I_exp)
                else:
                    self.plot_intensity_ref(self.sigma_sq_exp)
            except:
                self.replot_intensity_ref()
            
    def change_max_ref(self):
            
        if (self.allocated or self.progress > 0):

            try:
                vmax = np.float(self.lineEdit_max_ref.text())
                
                if (vmax <= np.float(self.lineEdit_min_ref.text())):
                    raise
                    
                self.lineEdit_max_ref.setText('%1.4e' % vmax)
                
                index = self.comboBox_plot_ref.currentIndex()    
                plot = self.comboBox_plot_ref.itemText(index)
                
                if (plot == 'Calculated'):
                    self.plot_intensity_ref(self.I_obs)
                elif (plot == 'Experimental'):
                    self.plot_intensity_ref(self.I_exp)
                else:
                    self.plot_intensity_ref(self.sigma_sq_exp)
            except:
                self.replot_intensity_ref()
                
    def replot_intensity_ref(self):
        
        if (self.allocated or self.progress > 0):
        
            index = self.comboBox_plot_ref.currentIndex()    
            plot = self.comboBox_plot_ref.itemText(index)
            
            if (plot == 'Calculated'):
                self.I_obs_m = np.ma.masked_less_equal(
                                              np.ma.masked_invalid(self.I_obs), 
                                              0.0, 
                                              copy=False)
                self.lineEdit_min_ref.setText('%1.4e' % self.I_obs_m.min())
                self.lineEdit_max_ref.setText('%1.4e' % self.I_obs_m.max())
                self.plot_intensity_ref(self.I_obs)
            elif (plot == 'Experimental'):
                self.I_exp_m = np.ma.masked_less_equal(
                                              np.ma.masked_invalid(self.I_exp), 
                                              0.0, 
                                              copy=False)
                self.lineEdit_min_ref.setText('%1.4e' % self.I_exp_m.min())
                self.lineEdit_max_ref.setText('%1.4e' % self.I_exp_m.max())
                self.plot_intensity_ref(self.I_exp)
            else:
                self.sigma_sq_exp_m = np.ma.masked_less_equal(
                                       np.ma.masked_invalid(self.sigma_sq_exp), 
                                       0.0, 
                                       copy=False)
                self.lineEdit_min_ref.setText('%1.4e' % 
                                              self.sigma_sq_exp_m.min())
                self.lineEdit_max_ref.setText('%1.4e' % 
                                              self.sigma_sq_exp_m.max())
                self.plot_intensity_ref(self.sigma_sq_exp)
            
    def change_sigma_h(self):
        
        if (self.tableWidget_exp.rowCount()):
        
            size_h = np.float(self.tableWidget_exp.item(0, 1).text())
            
            try:
                sigma_h = np.float(self.lineEdit_filter_ref_h.text())
                        
                if (sigma_h < 0):
                    sigma_h = 0
                elif (sigma_h > size_h):
                    sigma_h = size_h
                    
                self.lineEdit_filter_ref_h.setText(str(sigma_h))
            except:
                self.lineEdit_filter_ref_h.setText('0.0')
                
            if (self.allocated):
                
                self.filter_sigma()
                
    def change_sigma_k(self):
     
        if (self.tableWidget_exp.rowCount()):

            size_k = np.float(self.tableWidget_exp.item(1, 1).text())
             
            try:
                sigma_k = np.float(self.lineEdit_filter_ref_k.text())
                        
                if (sigma_k < 0):
                    sigma_k = 0
                elif (sigma_k > size_k):
                    sigma_k = size_k
                    
                self.lineEdit_filter_ref_k.setText(str(sigma_k))
            except:
                self.lineEdit_filter_ref_k.setText('0.0')
                
            if (self.allocated):
                
                self.filter_sigma()
                
    def change_sigma_l(self):
        
        if (self.tableWidget_exp.rowCount()):
       
            size_l = np.float(self.tableWidget_exp.item(2, 1).text())
             
            try:
                sigma_l = np.float(self.lineEdit_filter_ref_l.text())
                
                if (sigma_l < 0):
                    sigma_l = 0
                elif (sigma_l > size_l):
                    sigma_l = size_l
                    
                self.lineEdit_filter_ref_l.setText(str(sigma_l))
            except:
                self.lineEdit_filter_ref_l.setText('0.0')

            if (self.allocated):
                
                self.filter_sigma()
        
    def filter_sigma(self):
                
        sigma_h = np.float(self.lineEdit_filter_ref_h.text())       
        sigma_k = np.float(self.lineEdit_filter_ref_k.text())       
        sigma_l = np.float(self.lineEdit_filter_ref_l.text())    
        
        sigma = [sigma_h, sigma_k, sigma_l]
                
        self.v_inv =  space.gaussian(self.mask, sigma)
        
        self.boxes = space.boxblur(sigma, 3)
        
    def initialize_intensity(self):
        
        self.I_calc = np.zeros(self.Q.size, dtype=np.double)
        
        self.I_ref = self.I_obs[~self.mask].copy() # memoryview
        
        self.a_filt = np.zeros(self.mask.size, dtype=np.double)
        self.b_filt = np.zeros(self.mask.size, dtype=np.double)
        self.c_filt = np.zeros(self.mask.size, dtype=np.double)
        self.d_filt = np.zeros(self.mask.size, dtype=np.double)
        self.e_filt = np.zeros(self.mask.size, dtype=np.double)
        self.f_filt = np.zeros(self.mask.size, dtype=np.double)
        self.g_filt = np.zeros(self.mask.size, dtype=np.double)
        self.h_filt = np.zeros(self.mask.size, dtype=np.double)
        self.i_filt = np.zeros(self.mask.size, dtype=np.double)
        
        self.I_raw = np.zeros(self.mask.size, dtype=np.double)
        self.I_flat = np.zeros(self.mask.size, dtype=np.double)
        
    def refinement_statistics(self):
        
        self.acc_moves = []
        self.acc_temps = []
        self.rej_moves = []
        self.rej_temps = []
        
        self.chi_sq = [np.inf]
        self.energy = []
        self.temperature = [self.temp]
        self.scale = []
        
    def preprocess_supercell(self):
        
        selection = self.comboBox_type.currentIndex()    
        disorder = self.comboBox_type.itemText(selection)
        
        self.mask = np.isnan(self.I_exp)\
                  + np.isinf(self.I_exp)\
                  + np.less_equal(self.I_exp, 0, where=~np.isnan(self.I_exp))\
                  + np.isnan(self.sigma_sq_exp)\
                  + np.isinf(self.sigma_sq_exp)\
                  + np.less_equal(self.sigma_sq_exp, \
                                  0, \
                                  where=~np.isnan(self.sigma_sq_exp))
        
        self.I_expt = self.I_exp[~self.mask]
        self.inv_sigma_sq = 1/self.sigma_sq_exp[~self.mask]
        
        self.nu = np.int(self.lineEdit_nu.text())
        self.nv = np.int(self.lineEdit_nv.text())
        self.nw = np.int(self.lineEdit_nw.text())
 
        self.n_atm = np.int(self.lineEdit_n_atm.text())
        
        self.n_uvw = self.nu*self.nv*self.nw
        self.n = self.n_uvw*self.n_atm
        
        self.a = np.float(self.lineEdit_a.text())
        self.b = np.float(self.lineEdit_b.text())
        self.c = np.float(self.lineEdit_c.text())
        
        self.alpha = np.float(self.lineEdit_alpha.text())*np.pi/180
        self.beta = np.float(self.lineEdit_beta.text())*np.pi/180
        self.gamma = np.float(self.lineEdit_gamma.text())*np.pi/180
        
        self.A, self.B, self.R = crystal.matrices(self.a, 
                                                  self.b, 
                                                  self.c, 
                                                  self.alpha, 
                                                  self.beta, 
                                                  self.gamma)
            
        self.C, self.D = crystal.orthogonalized(self.a, 
                                                self.b, 
                                                self.c, 
                                                self.alpha, 
                                                self.beta, 
                                                self.gamma)
            
        self.atm, self.ion, self.occupancy = [], [], []
        self.Uiso, self.U11, self.U22, self.U33, self.g = [], [], [], [], []
        self.U23, self.U13, self.U12 = [], [], []
        self.M, self.M1, self.M2, self.M3, self.g = [], [], [], [], []
        self.u, self.v, self.w = [], [], []
        
        for i in range(self.tableWidget_CIF.rowCount()):
            if (not self.tableWidget_CIF.isRowHidden(i)):
                elm = np.str(self.tableWidget_CIF.item(i, 1).text())
                app = np.str(self.tableWidget_CIF.item(i, 2).text())
                pre = np.str(self.tableWidget_CIF.item(i, 3).text())
                self.atm.append(pre+elm)
                self.ion.append(elm+app)                 
                occ = self.tableWidget_CIF.item(i,4).text()
                self.occupancy.append(np.float(occ))
                Uiso = self.tableWidget_CIF.item(i, 5).text()
                U11 = self.tableWidget_CIF.item(i, 6).text()
                U22 = self.tableWidget_CIF.item(i, 7).text()
                U33 = self.tableWidget_CIF.item(i, 8).text()
                U23 = self.tableWidget_CIF.item(i, 9).text()
                U13 = self.tableWidget_CIF.item(i, 10).text()
                U12 = self.tableWidget_CIF.item(i, 11).text()
                self.Uiso.append(np.float(Uiso))
                self.U11.append(np.float(U11))
                self.U22.append(np.float(U22))
                self.U33.append(np.float(U33))
                self.U23.append(np.float(U23))
                self.U13.append(np.float(U13))
                self.U12.append(np.float(U12))
                M = self.tableWidget_CIF.item(i, 12).text()
                M1 = self.tableWidget_CIF.item(i, 13).text()
                M2 = self.tableWidget_CIF.item(i, 14).text()
                M3 = self.tableWidget_CIF.item(i, 15).text()
                g = self.tableWidget_CIF.item(i, 16).text()
                self.M.append(np.float(M))
                self.M1.append(np.float(M1))
                self.M2.append(np.float(M2))
                self.M3.append(np.float(M3))
                self.g.append(np.float(g))
                u = self.tableWidget_CIF.item(i, 17).text()
                v = self.tableWidget_CIF.item(i, 18).text()
                w = self.tableWidget_CIF.item(i, 19).text()
                self.u.append(np.float(u))
                self.v.append(np.float(v))
                self.w.append(np.float(w))
            
        self.atm = np.array(self.atm)
        self.ion = np.array(self.ion)
        self.occupancy = np.array(self.occupancy)
        self.Uiso = np.array(self.Uiso)
        self.U11 = np.array(self.U11)
        self.U22 = np.array(self.U22)
        self.U33 = np.array(self.U33)
        self.U23 = np.array(self.U23)
        self.U13 = np.array(self.U13)
        self.U12 = np.array(self.U12)
        self.M = np.array(self.M)
        self.M1 = np.array(self.M1)
        self.M2 = np.array(self.M2)
        self.M3 = np.array(self.M3)
        self.g = np.array(self.g)
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        self.w = np.array(self.w)
            
        self.ux, self.uy, self.uz = crystal.transform(self.u, 
                                                      self.v, 
                                                      self.w, 
                                                      self.A)
        
        self.nh = self.mask.shape[0]
        self.nk = self.mask.shape[1]
        self.nl = self.mask.shape[2]
        
        self.I_obs = np.full((self.nh, self.nk, self.nl), np.nan)
        
        t = 0
        T = np.array([[np.cos(t), -np.sin(t), 0],
                      [np.sin(t),  np.cos(t), 0],
                      [0,          0,         1]])
    
        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                
        h_range = [min_h, max_h]
        k_range = [min_k, max_k]
        l_range = [min_l, max_l]
        
        self.h, \
        self.k, \
        self.l, \
        self.H, \
        self.K, \
        self.L, \
        self.indices, \
        self.inverses,\
        self.operators = crystal.bragg(h_range, 
                                       k_range,
                                       l_range,
                                       self.nh,
                                       self.nk,
                                       self.nl,
                                       self.nu,
                                       self.nv,
                                       self.nw,
                                       T=T,
                                       folder=self.folder, 
                                       filename=self.filename,
                                       symmetry=None)
            
        self.Qh, self.Qk, self.Ql = space.nuclear(self.h, 
                                                  self.k, 
                                                  self.l, 
                                                  self.B)
        
        self.Qx, self.Qy, self.Qz = crystal.transform(self.Qh, 
                                                      self.Qk, 
                                                      self.Ql, 
                                                      self.R)
        
        self.Qx_norm, self.Qy_norm, self.Qz_norm, self.Q = space.unit(self.Qx, 
                                                                      self.Qy, 
                                                                      self.Qz)
        
        self.ix, self.iy, self.iz = space.cell(self.nu, 
                                               self.nv, 
                                               self.nw, 
                                               self.A)
        
        self.rx, self.ry, self.rz, self.atms = space.real(self.ux, 
                                                          self.uy, 
                                                          self.uz, 
                                                          self.ix, 
                                                          self.iy, 
                                                          self.iz, 
                                                          self.atm)
        
        self.i_mask, self.i_unmask = space.indices(self.inverses, self.mask)
                
        self.phase_factor = scattering.phase(self.Qx, 
                                             self.Qy, 
                                             self.Qz, 
                                             self.ux, 
                                             self.uy, 
                                             self.uz)
        
        self.space_factor = space.factor(self.nu, self.nv, self.nw)
        
        self.moment = M
        
        self.displacement = Uiso
                
        if (disorder == 'Neutron'):
            
            self.magnetic_form_factor = magnetic.form(self.Q, self.ion, self.g)
            
            self.magnetic_factors = self.magnetic_form_factor*self.phase_factor
                        
            self.scattering_length = scattering.length(self.atm, self.Q.size)
                        
            self.factors = space.prefactors(self.scattering_length, 
                                            self.phase_factor, 
                                            self.occupancy)
            
        else:
            
            self.form_factor = scattering.form(self.ion, self.Q)
            
            self.factors = space.prefactors(self.form_factor, 
                                            self.phase_factor, 
                                            self.occupancy)
            
        self.temp = np.float(self.lineEdit_prefactor.text())
        self.constant = np.float(self.lineEdit_tau.text())
        
    def stop_refinement(self):
        
        if (self.tableWidget_exp.rowCount()):

            if (self.iteration != 0):
                self.stop = True
                if (self.running):
                    self.pushButton_run.setEnabled(False)
                    self.pushButton_reset_run.setEnabled(False)
            
    def reset_refinement(self):
        
        if (self.tableWidget_exp.rowCount()):

            self.stop_refinement()
            self.stop = True
            self.progress = 0
            self.iteration = 0
            self.batch = 0
            self.started = False
            self.allocated = False
            self.progressBar_ref.setValue(self.progress)
            self.lineEdit_run.setText(str(self.batch))
            
            self.lineEdit_min_ref.setText('')
            self.lineEdit_max_ref.setText('')
            
            fig = self.canvas_ref.figure
            fig.clear()   
            with np.errstate(invalid='ignore'):
                self.canvas_ref.draw()
            
            fig = self.canvas_chi_sq.figure
            fig.clear()   
            with np.errstate(invalid='ignore'):
                self.canvas_chi_sq.draw()
                
            self.pushButton_run.setEnabled(True)

    def disorder_select(self):
                
        tab = self.tabWidget_disorder.currentIndex()
        
        self.comboBox_correlations_1d.setCurrentIndex(tab)
        self.comboBox_correlations_3d.setCurrentIndex(tab)
        
        self.tabWidget_calc.setCurrentIndex(tab)
        
        if (self.progress == 0 and self.allocated == False):
            self.started = False
            
    def run_refinement_thread(self, callback):
        
        self.running = True
                    
        runs = np.int(self.lineEdit_runs.text())
        cycles = np.int(self.lineEdit_cycles.text())
                                                        
        for _ in range(self.batch, runs):               
        
            if (self.allocated == False):
                self.preprocess_supercell()
                self.initialize_intensity()
                self.refinement_statistics()
                self.filter_sigma()
                self.allocated = True
                if (self.magnetic):
                    self.initialize_magnetic()
                if (self.occupational):
                    self.initialize_occupational()
                if (self.displacive):
                    self.initialize_displacive()
                    
            if (self.checkBox_batch.isChecked()):
                r = str(self.batch)
            else:
                r = ''
            
            n = cycles // 1
              
            for _ in range(self.iteration, n):
                self.refinement_cycle()
                self.iteration += 1
                self.progress = int(round(self.iteration/n*100))
                
                prog = self.progress
                callback.emit(prog)
                
                if (self.stop):
                    self.running = False
                    break  
            
            if (self.magnetic):
                self.save_magnetic(r)
            if (self.occupational):
                self.save_occupational(r)
            if (self.displacive):
                self.save_displacive(r)
                
            if (self.stop):
                self.stop = False
                self.pushButton_run.setEnabled(True)
                self.pushButton_reset_run.setEnabled(True)
                break     

            self.batch += 1 
            self.iteration = 0
            self.allocated = False
            
    def run_refinement_thread_complete(self):
        
        self.file_save()
        
        self.pushButton_run.setEnabled(True)
        
        self.running = False
        self.started = False
                                    
    def progress_update(self, p):
        
        self.progressBar_ref.setValue(self.progress)
        
        self.lineEdit_run.setText(str(self.batch))
        
        self.lineEdit_chi_sq.setText('%1.4e' % self.chi_sq[-1])
                
        self.replot_intensity_ref()
        
        self.plot_chi_sq()
        
    def run_refinement(self):
        
        if (self.tableWidget_exp.rowCount()):
            
            try:
                self.fname
            except:
                self.file_save_as()
                        
            try:
                self.fname
                
                self.pushButton_run.setEnabled(False)
                
                self.stop = False
                 
                if (not self.started):
                                        
                    if (self.checkBox_mag.isChecked()):
                        self.magnetic = True
                        self.occupational = False
                        self.displacive = False
                    elif (self.checkBox_occ.isChecked()):
                        self.magnetic = False
                        self.occupational = True
                        self.displacive = False
                    elif (self.checkBox_dis.isChecked()):
                        self.magnetic = False
                        self.occupational = False
                        self.displacive = True

                    self.started = True
                
                worker = Worker(self.run_refinement_thread)
                worker.signals.progress.connect(self.progress_update)
                worker.signals.finished.connect(
                                           self.run_refinement_thread_complete)
                
                self.threadpool.start(worker) 
                
            except:
                pass
        
    def refinement_cycle(self):
        
        N = self.n_uvw*self.n_atm*1
        
        if (self.magnetic):
            
            self.delta = 1
            
            self.fixed = self.checkBox_fixed_moment.isChecked()
            
            T = np.array([1.,1.,1.,0,0,0]).flatten()

            refinement.magnetic(self.Sx,
                                self.Sy,
                                self.Sz,
                                self.Qx_norm,
                                self.Qy_norm,
                                self.Qz_norm,
                                self.Sx_k,
                                self.Sy_k,
                                self.Sz_k,
                                self.Sx_k_orig,
                                self.Sy_k_orig,
                                self.Sz_k_orig,
                                self.Sx_k_cand,
                                self.Sy_k_cand,
                                self.Sz_k_cand,
                                self.Fx,
                                self.Fy,
                                self.Fz,
                                self.Fx_orig,
                                self.Fy_orig,
                                self.Fz_orig,
                                self.Fx_cand,
                                self.Fy_cand,
                                self.Fz_cand,
                                self.prod_x,
                                self.prod_y,
                                self.prod_z,
                                self.prod_x_orig,     
                                self.prod_y_orig,     
                                self.prod_z_orig,  
                                self.prod_x_cand,
                                self.prod_y_cand,
                                self.prod_z_cand,   
                                self.space_factor,
                                self.factors,
                                self.moment,
                                self.I_calc,
                                self.I_expt,
                                self.inv_sigma_sq,
                                self.I_raw, 
                                self.I_flat,
                                self.I_ref, 
                                self.v_inv, 
                                self.a_filt,
                                self.b_filt,
                                self.c_filt,
                                self.d_filt,
                                self.e_filt,
                                self.f_filt,
                                self.g_filt,
                                self.h_filt,
                                self.i_filt,
                                self.boxes,
                                self.i_dft,
                                self.inverses,
                                self.i_mask,
                                self.i_unmask,
                                self.acc_moves,
                                self.acc_temps,
                                self.rej_moves,
                                self.rej_temps,
                                self.chi_sq,
                                self.energy,
                                self.temperature,
                                self.scale,
                                self.constant,
                                self.delta,
                                self.fixed,
                                T,
                                self.nh,
                                self.nk, 
                                self.nl,
                                self.nu,
                                self.nv,
                                self.nw,
                                self.n_atm,
                                self.n,
                                N)
            
        if (self.occupational):
                        
            self.fixed = self.checkBox_fixed_composition.isChecked()
                        
            refinement.occupational(self.A_r,
                                    self.A_k,
                                    self.A_k_orig,
                                    self.A_k_cand,
                                    self.F,
                                    self.F_orig,
                                    self.F_cand,
                                    self.prod,
                                    self.prod_orig,
                                    self.prod_cand,     
                                    self.space_factor,
                                    self.factors,
                                    self.occupancy,
                                    self.I_calc,
                                    self.I_expt,
                                    self.inv_sigma_sq,
                                    self.I_raw, 
                                    self.I_flat,
                                    self.I_ref, 
                                    self.v_inv, 
                                    self.a_filt,
                                    self.b_filt,
                                    self.c_filt,
                                    self.d_filt,
                                    self.e_filt,
                                    self.f_filt,
                                    self.g_filt,
                                    self.h_filt,
                                    self.i_filt,
                                    self.boxes,
                                    self.i_dft,
                                    self.inverses,
                                    self.i_mask,
                                    self.i_unmask,
                                    self.acc_moves,
                                    self.acc_temps,
                                    self.rej_moves,
                                    self.rej_temps,
                                    self.chi_sq,
                                    self.energy,
                                    self.temperature,
                                    self.scale,
                                    self.constant,
                                    self.fixed,
                                    self.nh,
                                    self.nk, 
                                    self.nl,
                                    self.nu,
                                    self.nv,
                                    self.nw,
                                    self.n_atm,
                                    self.n,
                                    N)
            
        if (self.displacive):
            
            self.delta = 1
            
            self.fixed = self.checkBox_fixed_displacement.isChecked()
            
            T = np.array([1.,1.,1.,0,0,0]).flatten()
                                                
            refinement.displacive(self.Ux,
                                  self.Uy,
                                  self.Uz,
                                  self.U_r,
                                  self.U_r_orig,
                                  self.U_r_cand,
                                  self.U_k,
                                  self.U_k_orig,
                                  self.U_k_cand,
                                  self.V_k,
                                  self.V_k_nuc,
                                  self.V_k_orig,
                                  self.V_k_nuc_orig,
                                  self.V_k_cand,
                                  self.V_k_nuc_cand,
                                  self.F,
                                  self.F_nuc,
                                  self.F_orig,
                                  self.F_nuc_orig,
                                  self.F_cand,
                                  self.F_nuc_cand,
                                  self.prod,
                                  self.prod_nuc,
                                  self.prod_orig,     
                                  self.prod_nuc_orig,    
                                  self.prod_cand,
                                  self.prod_nuc_cand,
                                  self.space_factor,
                                  self.factors,
                                  self.coeffs,
                                  self.Q_k,
                                  self.displacement,
                                  self.I_calc,
                                  self.I_expt,
                                  self.inv_sigma_sq,
                                  self.I_raw, 
                                  self.I_flat,
                                  self.I_ref, 
                                  self.v_inv, 
                                  self.a_filt,
                                  self.b_filt,
                                  self.c_filt,
                                  self.d_filt,
                                  self.e_filt,
                                  self.f_filt,
                                  self.g_filt,
                                  self.h_filt,
                                  self.i_filt,
                                  self.bragg,
                                  self.even,
                                  self.boxes,
                                  self.i_dft,
                                  self.inverses,
                                  self.i_mask,
                                  self.i_unmask,
                                  self.acc_moves,
                                  self.acc_temps,
                                  self.rej_moves,
                                  self.rej_temps,
                                  self.chi_sq,
                                  self.energy,
                                  self.temperature,
                                  self.scale,
                                  self.constant,
                                  self.delta,
                                  self.fixed,
                                  T,
                                  self.p,
                                  self.nh,
                                  self.nk, 
                                  self.nl,
                                  self.nu,
                                  self.nv,
                                  self.nw,
                                  self.n_atm,
                                  self.n,
                                  N)
        
        self.I_obs = self.I_flat.reshape(self.nh,self.nk,self.nl)
        self.I_obs[self.mask] = np.nan
                        
    def initialize_magnetic(self):
       
        if (not self.restart):
            
            self.Sx, self.Sy, self.Sz = magnetic.spin(self.nu, 
                                                      self.nv, 
                                                      self.nw, 
                                                      self.n_atm)
        
        self.Sx_k, \
        self.Sy_k, \
        self.Sz_k, \
        self.i_dft = magnetic.transform(self.Sx, 
                                        self.Sy, 
                                        self.Sz, 
                                        self.H,
                                        self.K,
                                        self.L,
                                        self.nu, 
                                        self.nv, 
                                        self.nw, 
                                        self.n_atm)
        
        self.Fx, \
        self.Fy, \
        self.Fz, \
        self.prod_x, \
        self.prod_y, \
        self.prod_z = magnetic.structure(self.Qx_norm, 
                                         self.Qy_norm, 
                                         self.Qz_norm, 
                                         self.Sx_k, 
                                         self.Sy_k, 
                                         self.Sz_k, 
                                         self.i_dft,
                                         self.magnetic_factors)
        
        self.Fx_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.Fy_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.Fz_orig = np.zeros(self.indices.size, dtype=np.complex)
        
        self.prod_x_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.prod_y_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.prod_z_orig = np.zeros(self.indices.size, dtype=np.complex)
        
        self.Sx_k_orig = np.zeros(self.n_uvw, dtype=np.complex)
        self.Sy_k_orig = np.zeros(self.n_uvw, dtype=np.complex)
        self.Sz_k_orig = np.zeros(self.n_uvw, dtype=np.complex)
        
        self.Fx_cand = np.zeros(self.indices.size, dtype=np.complex)
        self.Fy_cand = np.zeros(self.indices.size, dtype=np.complex)
        self.Fz_cand = np.zeros(self.indices.size, dtype=np.complex)
        
        self.prod_x_cand = np.zeros(self.indices.size, dtype=np.complex)
        self.prod_y_cand = np.zeros(self.indices.size, dtype=np.complex)
        self.prod_z_cand = np.zeros(self.indices.size, dtype=np.complex)
        
        self.Sx_k_cand = np.zeros(self.n_uvw, dtype=np.complex)
        self.Sy_k_cand = np.zeros(self.n_uvw, dtype=np.complex)
        self.Sz_k_cand = np.zeros(self.n_uvw, dtype=np.complex)
        
    def initialize_occupational(self):
        
        if (not self.restart):
            
            self.A_r = occupational.composition(self.nu, 
                                                self.nv, 
                                                self.nw, 
                                                self.n_atm, 
                                                value=self.occupancy)
                            
        self.A_k, self.i_dft = occupational.transform(self.A_r, 
                                                      self.H,
                                                      self.K,
                                                      self.L,
                                                      self.nu, 
                                                      self.nv, 
                                                      self.nw, 
                                                      self.n_atm)
            
        self.F, self.prod = occupational.structure(self.A_k, 
                                                   self.i_dft,
                                                   self.factors)
                        
        self.F_orig = np.zeros(self.indices.size, dtype=np.complex)
        
        self.prod_orig = np.zeros(self.indices.size, dtype=np.complex)
        
        self.A_k_orig = np.zeros(self.n_uvw, dtype=np.complex)
        
        self.F_cand = np.zeros(self.indices.size, dtype=np.complex)
        
        self.prod_cand = np.zeros(self.indices.size, dtype=np.complex)
        
        self.A_k_cand = np.zeros(self.n_uvw, dtype=np.complex)
            
    def initialize_displacive(self):
                
        self.p = np.int(self.lineEdit_order.text())
  
        if (not self.restart):
                          
            self.Ux, self.Uy, self.Uz = displacive.expansion(self.nu, 
                                                             self.nv, 
                                                             self.nw, 
                                                             self.n_atm, 
                                                       value=self.displacement)
            

        self.coeffs = displacive.coefficients(self.p)
        
        self.U_r = displacive.products(self.Ux, self.Uy, self.Uz, self.p)
        self.Q_k = displacive.products(self.Qx, self.Qy, self.Qz, self.p)
        
        self.U_k, self.i_dft = displacive.transform(self.U_r, 
                                                    self.H,
                                                    self.K,
                                                    self.L,
                                                    self.nu, 
                                                    self.nv, 
                                                    self.nw, 
                                                    self.n_atm)
        
        centering_index = self.comboBox_centering_ref.currentIndex()    
        centering = self.comboBox_centering_ref.itemText(centering_index)
        
        lat = self.lineEdit_lat.text()
        
        if (lat == 'Rhombohedral'):
            if (centering == 'R'):
                centering = 'P'
                
        self.centering = centering
        
        self.H_nuc, \
        self.K_nuc, \
        self.L_nuc, \
        self.cond = crystal.nuclear(self.H, 
                                    self.K, 
                                    self.L, 
                                    self.h, 
                                    self.k, 
                                    self.l, 
                                    self.nu, 
                                    self.nv, 
                                    self.nw, 
                                    self.centering)    
        
        self.F, \
        self.F_nuc, \
        self.prod, \
        self.prod_nuc, \
        self.V_k, \
        self.V_k_nuc, \
        self.even, \
        self.bragg = displacive.structure(self.U_k, 
                                          self.Q_k, 
                                          self.coeffs, 
                                          self.cond,
                                          self.p,
                                          self.i_dft,
                                          self.factors)
        
        self.F_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.F_nuc_orig = np.zeros(self.bragg.size, dtype=np.complex)
        
        self.prod_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.prod_nuc_orig = np.zeros(self.bragg.size, dtype=np.complex)
        
        self.V_k_orig = np.zeros(self.indices.size, dtype=np.complex)
        self.V_k_nuc_orig = np.zeros(self.bragg.size, dtype=np.complex)
        
        self.U_k_orig = np.zeros(self.n_uvw*self.coeffs.size, dtype=np.complex)
        
        self.F_cand = np.zeros(self.indices.shape, dtype=np.complex)
        self.F_nuc_cand = np.zeros(self.bragg.shape, dtype=np.complex)
        
        self.prod_cand = np.zeros(self.indices.shape, dtype=np.complex)
        self.prod_nuc_cand = np.zeros(self.bragg.shape, dtype=np.complex)
        
        self.V_k_cand = np.zeros(self.indices.size, dtype=np.complex)
        self.V_k_nuc_cand = np.zeros(self.bragg.size, dtype=np.complex)
        
        self.U_k_cand = np.zeros(self.n_uvw*self.coeffs.size, dtype=np.complex)
        
        self.U_r_orig = np.zeros(self.coeffs.size, dtype=np.double)
        
        self.U_r_cand = np.zeros(self.coeffs.size, dtype=np.double)
        
    def save_magnetic(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''

        np.save(self.fname+'-calculated-spin-x'+r+'.npy', self.Sx)
        np.save(self.fname+'-calculated-spin-y'+r+'.npy', self.Sy)
        np.save(self.fname+'-calculated-spin-z'+r+'.npy', self.Sz)
        
        np.save(self.fname+\
                '-calculated-intensity-magnetic'+r+'.npy', self.I_obs)
        
        np.save(self.fname+'-goodness-of-fit-magnetic'+r+'.npy', self.chi_sq)
        np.save(self.fname+'-energy-magnetic'+r+'.npy', self.energy)        
        np.save(self.fname+'-temperature-magnetic'+r+'.npy', self.temperature)
        np.save(self.fname+'-scale-factor-magnetic'+r+'.npy', self.scale)

        np.save(self.fname+'-accepted-moves-magnetic'+r+'.npy', self.acc_moves)
        np.save(self.fname+'-rejected-moves-magnetic'+r+'.npy', self.rej_moves)     
        
        np.save(self.fname+\
                '-accepted-temperature-magnetic'+r+'.npy', self.acc_temps)
        np.save(self.fname+\
                '-rejected-temperature-magnetic'+r+'.npy', self.rej_temps)
        
    def save_occupational(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''
            
        np.save(self.fname+'-calculated-composition'+r+'.npy', self.A_r)
    
        np.save(self.fname+\
                '-calculated-intensity-chemical'+r+'.npy', self.I_obs)
                
        np.save(self.fname+'-goodness-of-fit-chemical'+r+'.npy', self.chi_sq)
        np.save(self.fname+'-energy-chemical'+r+'.npy', self.energy)
        np.save(self.fname+'-temperature-chemical'+r+'.npy', self.temperature)
        np.save(self.fname+'-scale-factor-chemical'+r+'.npy', self.scale)
        
        np.save(self.fname+'-accepted-moves-chemical'+r+'.npy', self.acc_moves)
        np.save(self.fname+'-rejected-moves-chemical'+r+'.npy', self.rej_moves)
        
        np.save(self.fname+\
                '-accepted-temperature-chemical'+r+'.npy', self.acc_temps)
        np.save(self.fname+\
                '-rejected-temperature-chemical'+r+'.npy', self.rej_temps)
        
    def save_displacive(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''
            
        np.save(self.fname+'-calculated-displacement-x'+r+'.npy', self.Ux)
        np.save(self.fname+'-calculated-displacement-y'+r+'.npy', self.Uy)
        np.save(self.fname+'-calculated-displacement-z'+r+'.npy', self.Uz)
        
        np.save(self.fname+\
                '-calculated-intensity-displacement'+r+'.npy', self.I_obs)
        
        np.save(self.fname+\
                '-goodness-of-fit-displacement'+r+'.npy', self.chi_sq)
        np.save(self.fname+'-energy-displacement'+r+'.npy', self.energy)
        np.save(self.fname+\
                '-temperature-displacement'+r+'.npy', self.temperature)
        np.save(self.fname+'-scale-factor-displacement'+r+'.npy', self.scale)
       
        np.save(self.fname+\
                '-accepted-moves-displacement'+r+'.npy', self.acc_moves)
        np.save(self.fname+\
                '-rejected-moves-displacement'+r+'.npy', self.rej_moves)    
        np.save(self.fname+\
                '-accepted-temperature-displacement'+r+'.npy', self.acc_temps)
        np.save(self.fname+\
                '-rejected-temperature-displacement'+r+'.npy', self.rej_temps)
        
    def load_magnetic(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''

        self.Sx = np.load(self.fname+'-calculated-spin-x'+r+'.npy')
        self.Sy = np.load(self.fname+'-calculated-spin-y'+r+'.npy')
        self.Sz = np.load(self.fname+'-calculated-spin-z'+r+'.npy')
        
        self.I_obs = np.load(self.fname+\
                             '-calculated-intensity-magnetic'+r+'.npy')
        
        self.chi_sq = np.load(self.fname+\
                              '-goodness-of-fit-magnetic'+r+'.npy').tolist()
        self.energy = np.load(self.fname+\
                              '-energy-magnetic'+r+'.npy').tolist()
        self.temperature = np.load(self.fname+\
                                   '-temperature-magnetic'+r+'.npy').tolist()
        self.scale = np.load(self.fname+\
                             '-scale-factor-magnetic'+r+'.npy').tolist()

        self.acc_moves = np.load(self.fname+\
                                 '-accepted-moves-magnetic'+r+'.npy').tolist()
        self.rej_moves = np.load(self.fname+\
                                 '-rejected-moves-magnetic'+r+'.npy') .tolist()     
        self.acc_temps = np.load(self.fname+\
                            '-accepted-temperature-magnetic'+r+'.npy').tolist()
        self.rej_temps = np.load(self.fname+\
                            '-rejected-temperature-magnetic'+r+'.npy').tolist()

    def load_occupational(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''
            
        self.A_r = np.load(self.fname+'-calculated-composition'+r+'.npy')
    
        self.I_obs = np.load(self.fname+\
                             '-calculated-intensity-chemical'+r+'.npy')

        self.chi_sq = np.load(self.fname+\
                              '-goodness-of-fit-chemical'+r+'.npy').tolist()
        self.energy = np.load(self.fname+\
                              '-energy-chemical'+r+'.npy').tolist()  
        self.temperature = np.load(self.fname+\
                                   '-temperature-chemical'+r+'.npy').tolist()        
        self.scale = np.load(self.fname+\
                             '-scale-factor-chemical'+r+'.npy').tolist()
        
        self.acc_moves = np.load(self.fname+
                                 '-accepted-moves-chemical'+r+'.npy').tolist()
        self.rej_moves = np.load(self.fname+\
                                 '-rejected-moves-chemical'+r+'.npy').tolist()       
        self.acc_temps = np.load(self.fname+\
                            '-accepted-temperature-chemical'+r+'.npy').tolist()
        self.rej_temps = np.load(self.fname+\
                            '-rejected-temperature-chemical'+r+'.npy').tolist()
        
    def load_displacive(self, run):
        
        if (run):
            r = '-'+run
        else:
            r = ''
            
        self.Ux = np.load(self.fname+'-calculated-displacement-x'+r+'.npy')
        self.Uy = np.load(self.fname+'-calculated-displacement-y'+r+'.npy')
        self.Uz = np.load(self.fname+'-calculated-displacement-z'+r+'.npy')
        
        self.I_obs = np.load(self.fname+\
                             '-calculated-intensity-displacement'+r+'.npy')
        
        self.chi_sq = np.load(self.fname+\
                             '-goodness-of-fit-displacement'+r+'.npy').tolist()
        self.energy = np.load(self.fname+\
                              '-energy-displacement'+r+'.npy').tolist()
        self.temperature = np.load(self.fname+\
                                 '-temperature-displacement'+r+'.npy').tolist()
        self.scale = np.load(self.fname+\
                                '-scale-factor-displacement'+r+'.npy').tolist()
       
        self.acc_moves = np.load(self.fname+\
                              '-accepted-moves-displacement'+r+'.npy').tolist()
        self.rej_moves = np.load(self.fname+\
                              '-rejected-moves-displacement'+r+'.npy').tolist()
        self.acc_temps = np.load(self.fname+\
                        '-accepted-temperature-displacement'+r+'.npy').tolist()
        self.rej_temps = np.load(self.fname+\
                        '-rejected-temperature-displacement'+r+'.npy').tolist()
                   
    def reset_punch(self):

        if (self.tableWidget_exp.rowCount()):
                        
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')     
            
            update_min_h = np.float(self.lineEdit_min_h.text())
            update_max_h = np.float(self.lineEdit_max_h.text())
    
            update_min_k = np.float(self.lineEdit_min_k.text())
            update_max_k = np.float(self.lineEdit_max_k.text())
            
            update_min_l = np.float(self.lineEdit_min_l.text())
            update_max_l = np.float(self.lineEdit_max_l.text())
        
            ind_min_h = np.round((update_min_h-self.fixed_min_h)
                                 /self.fixed_step_h, 4).astype(int)
            ind_max_h = np.round((update_max_h-self.fixed_min_h)
                                 /self.fixed_step_h, 4).astype(int)
            
            ind_min_k = np.round((update_min_k-self.fixed_min_k)
                                 /self.fixed_step_k, 4).astype(int)
            ind_max_k = np.round((update_max_k-self.fixed_min_k)
                                 /self.fixed_step_k, 4).astype(int)
            
            ind_min_l = np.round((update_min_l-self.fixed_min_l)
                                 /self.fixed_step_l, 4).astype(int)
            ind_max_l = np.round((update_max_l-self.fixed_min_l)
                                 /self.fixed_step_l, 4).astype(int)
            
            h_slice = [ind_min_h, ind_max_h+1]
            k_slice = [ind_min_k, ind_max_k+1]
            l_slice = [ind_min_l, ind_max_l+1]
            
            self.I_exp = experimental.crop(self.I_exp, 
                                           h_slice, k_slice, l_slice)
            self.sigma_sq_exp = experimental.crop(self.sigma_sq_exp, 
                                                  h_slice, k_slice, l_slice)
            
            size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))
    
            size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))
    
            size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))
            
            binsize = [size_h, size_k, size_l]
                            
            self.I_exp = experimental.rebin(self.I_exp, binsize)
            self.sigma_sq_exp = experimental.rebin(self.sigma_sq_exp, binsize)
     
            self.replot_intensity_exp()
        
    def punch(self):
        
        if (self.tableWidget_exp.rowCount()):
                           
            self.bragg_punch(self.I_exp)
            
            self.mask = np.isnan(self.I_exp)\
                      + np.isinf(self.I_exp)\
                      + np.isnan(self.sigma_sq_exp)\
                      + np.isinf(self.sigma_sq_exp)
                      
            self.I_exp[self.mask] = np.nan
            self.sigma_sq_exp[self.mask] = np.nan
                  
            self.replot_intensity_exp()
        
    def bragg_punch(self, data):
        
        step_h = np.float(self.tableWidget_exp.item(0, 0).text())
        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        step_k = np.float(self.tableWidget_exp.item(1, 0).text())
        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        step_l = np.float(self.tableWidget_exp.item(2, 0).text())
        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        max_l = np.float(self.tableWidget_exp.item(2, 3).text())
        
        centering_index = self.comboBox_centering.currentIndex()    
        centering = self.comboBox_centering.itemText(centering_index)
        
        lat = self.lineEdit_lat.text()
        
        if (lat == 'Rhombohedral'):
            if (centering == 'R'):
                centering = 'P'
            else:
                centering = 'R (hexagonal axes, triple obverse cell)'
        
        punch_index = self.comboBox_punch.currentIndex()    
        punch = self.comboBox_punch.itemText(punch_index)
        
        radius_h = np.float(self.lineEdit_radius_h.text())
        radius_k = np.float(self.lineEdit_radius_k.text())
        radius_l = np.float(self.lineEdit_radius_l.text())
 
        outlier = np.float(self.lineEdit_outlier.text())
       
        box = [np.round(radius_h).astype(int), 
               np.round(radius_k).astype(int),
               np.round(radius_l).astype(int)]
        
        h_range = [np.round(min_h).astype(int), np.round(max_h).astype(int)]
        k_range = [np.round(min_k).astype(int), np.round(max_k).astype(int)]
        l_range = [np.round(min_l).astype(int), np.round(max_l).astype(int)]

        with warnings.catch_warnings():                    

            warnings.filterwarnings('ignore')
            
            for h in np.arange(h_range[0], h_range[1]+1):
                for k in np.arange(k_range[0], k_range[1]+1):
                    for l in np.arange(l_range[0], l_range[1]+1):
                        
                        allow = experimental.reflections(h,
                                                         k, 
                                                         l, 
                                                         centering=centering)
            
                        if (allow == 1):
                            
                            i_hkl = [int(np.round((h-h_range[0])/step_h,4)), \
                                     int(np.round((k-k_range[0])/step_k,4)), \
                                     int(np.round((l-l_range[0])/step_l,4))]
                                    
                            h0, h1 = i_hkl[0]-box[0], i_hkl[0]+box[0]+1  
                            k0, k1 = i_hkl[1]-box[1], i_hkl[1]+box[1]+1      
                            l0, l1 = i_hkl[2]-box[2], i_hkl[2]+box[2]+1
                            
                            if (h0 < 0):
                                h0 = 0
                            if (h1 >= data.shape[0]):
                                h1 = data.shape[0]
                                
                            if (k0 < 0):
                                k0 = 0
                            if (k1 >= data.shape[1]):
                                k1 = data.shape[1]
                                
                            if (l0 < 0):
                                l0 = 0
                            if (l1 >= data.shape[2]):
                                l1 = data.shape[2]
                                                
                            values = data[h0:h1,k0:k1,l0:l1].copy()
                            
                            if (punch == 'Ellipsoid'):
                                values_outside = values.copy()
                                x, \
                                y, \
                                z = np.meshgrid(np.arange(h0,h1)-i_hkl[0], 
                                                np.arange(k0,k1)-i_hkl[1], 
                                                np.arange(l0,l1)-i_hkl[2], 
                                                indexing='ij')
                                mask = x**2/np.float(box[0])**2\
                                     + y**2/np.float(box[1])**2\
                                     + z**2/np.float(box[2])**2 > 1
                                values[mask] = np.nan
                                
                            Q3 = np.nanpercentile(values,75)
                            Q1 = np.nanpercentile(values,25)
                        
                            interquartile = Q3-Q1                
                        
                            reject = (values >= Q3+outlier*interquartile) | \
                                     (values <  Q1-outlier*interquartile)
                            
                            values[reject] = np.nan
                            
                            if (punch == 'Ellipsoid'):
                                values[mask] = values_outside[mask].copy()
                               
                            data[h0:h1,k0:k1,l0:l1] = values.copy()
                            
    def change_radius_h(self):
        
        try:
            radius_h = np.int(self.lineEdit_radius_h.text())
            size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))

            if (radius_h < 0):
                h = 0
            elif (radius_h > size_h):
                h = size_h    
            else:
                h = radius_h
            self.lineEdit_radius_h.setText(str(h))  
        except:
            self.lineEdit_radius_h.setText('0')  
            
    def change_radius_k(self):
        
        try:
            radius_k = np.int(self.lineEdit_radius_k.text())
            size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))

            if (radius_k < 0):
                k = 0
            elif (radius_k > size_k):
                k = size_k     
            else:
                k = radius_k
            self.lineEdit_radius_k.setText(str(k))  
        except:
            self.lineEdit_radius_k.setText('0')  
            
    def change_radius_l(self):
        
        try:
            radius_l = np.int(self.lineEdit_radius_l.text())
            size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))

            if (radius_l < 0):
                l = 0
            elif (radius_l > size_l):
                l = size_l  
            else:
                l = radius_l
            self.lineEdit_radius_l.setText(str(l))  
        except:
            self.lineEdit_radius_l.setText('0')  
            
    def change_outlier(self):
        
        try:
            outlier = np.float(self.lineEdit_outlier.text())
            if (outlier < -0.5):
                outlier = -0.5
            self.lineEdit_outlier.setText(str(outlier))  
        except:
            self.lineEdit_outlier.setText('0')         
            
    def change_min_exp(self):
 
        if (self.tableWidget_exp.rowCount()):
           
            try:
                vmin = np.float(self.lineEdit_min_exp.text())
                
                if (vmin >= np.float(self.lineEdit_max_exp.text())):
                    raise
                    
                self.lineEdit_min_exp.setText('%1.4e' % vmin)
                
                index = self.comboBox_plot_exp.currentIndex()    
                plot = self.comboBox_plot_exp.itemText(index)
                
                if (plot == 'Intensity'):
                    self.plot_intensity_exp(self.I_exp)
                else:
                    self.plot_intensity_exp(self.sigma_sq_exp)
            except:
                self.replot_intensity_exp()
                
    def change_max_exp(self):

        if (self.tableWidget_exp.rowCount()):
            
            try:
                vmax = np.float(self.lineEdit_max_exp.text())
            
                if (vmax <= np.float(self.lineEdit_min_exp.text())):
                    raise    
                
                self.lineEdit_max_exp.setText('%1.4e' % vmax)
                
                index = self.comboBox_plot_exp.currentIndex()    
                plot = self.comboBox_plot_exp.itemText(index)
                
                if (plot == 'Intensity'):
                    self.plot_intensity_exp(self.I_exp)
                else:
                    self.plot_intensity_exp(self.sigma_sq_exp)
            except:
                self.replot_intensity_exp()

    def change_slice_h(self):
 
        if (self.tableWidget_exp.rowCount()):

            step_h = np.float(self.tableWidget_exp.item(0, 0).text())
            min_h = np.float(self.tableWidget_exp.item(0, 2).text())
            max_h = np.float(self.tableWidget_exp.item(0, 3).text())
                
            slice_h = np.float(self.lineEdit_slice_h.text())
                    
            if (slice_h < min_h):
                h = min_h
            elif (slice_h > max_h):
                h = max_h
            else:
                ih = np.int(np.round((slice_h-min_h)/step_h))
                h = np.round(min_h+step_h*ih, 4)
                
            self.lineEdit_slice_h.setText(str(h))
            
            self.replot_intensity_exp()
        
    def change_slice_k(self):

        if (self.tableWidget_exp.rowCount()):
        
            step_k = np.float(self.tableWidget_exp.item(1, 0).text())
            min_k = np.float(self.tableWidget_exp.item(1, 2).text())
            max_k = np.float(self.tableWidget_exp.item(1, 3).text())
                
            slice_k = np.float(self.lineEdit_slice_k.text())
                    
            if (slice_k < min_k):
                k = min_k
            elif (slice_k > max_k):
                k = max_k
            else:
                ik = np.int(np.round((slice_k-min_k)/step_k))
                k = np.round(min_k+step_k*ik, 4)
                
            self.lineEdit_slice_k.setText(str(k))
            
            self.replot_intensity_exp()
        
    def change_slice_l(self):

        if (self.tableWidget_exp.rowCount()):
            
            step_l = np.float(self.tableWidget_exp.item(2, 0).text())
            min_l = np.float(self.tableWidget_exp.item(2, 2).text())
            max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                
            slice_l = np.float(self.lineEdit_slice_l.text())
                    
            if (slice_l < min_l):
                l = min_l
            elif (slice_l > max_l):
                l = max_l
            else:
                il = np.int(np.round((slice_l-min_l)/step_l))
                l = np.round(min_l+step_l*il, 4)
                
            self.lineEdit_slice_l.setText(str(l))
            
            self.replot_intensity_exp()
          
    def replot_intensity_exp(self):
    
        if (self.tableWidget_exp.rowCount()):

            index = self.comboBox_plot_exp.currentIndex()    
            plot = self.comboBox_plot_exp.itemText(index)
            
            if (plot == 'Intensity'):
                self.I_exp_m = np.ma.masked_less_equal(
                                              np.ma.masked_invalid(self.I_exp), 
                                              0.0, 
                                              copy=False)
                self.lineEdit_min_exp.setText('%1.4e' % self.I_exp_m.min())
                self.lineEdit_max_exp.setText('%1.4e' % self.I_exp_m.max())
                self.plot_intensity_exp(self.I_exp)
            else:
                self.sigma_sq_exp_m = np.ma.masked_less_equal(
                                       np.ma.masked_invalid(self.sigma_sq_exp), 
                                       0.0, 
                                       copy=False)
                self.lineEdit_min_exp.setText('%1.4e' % 
                                              self.sigma_sq_exp_m.min())
                self.lineEdit_max_exp.setText('%1.4e' % 
                                              self.sigma_sq_exp_m.max())
                self.plot_intensity_exp(self.sigma_sq_exp)
                
    def rebin(self):
        
        size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))

        size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))

        size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))
        
        binsize = [size_h, size_k, size_l]
                        
        self.I_exp = experimental.rebin(self.I_exp, binsize)
        self.sigma_sq_exp = experimental.rebin(self.sigma_sq_exp, binsize)
        
        self.replot_intensity_exp()
        
    def crop(self):
                
        update_min_h = np.float(self.lineEdit_min_h.text())
        update_max_h = np.float(self.lineEdit_max_h.text())

        update_min_k = np.float(self.lineEdit_min_k.text())
        update_max_k = np.float(self.lineEdit_max_k.text())
        
        update_min_l = np.float(self.lineEdit_min_l.text())
        update_max_l = np.float(self.lineEdit_max_l.text())

        step_h = np.float(self.tableWidget_exp.item(0, 0).text())
        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        #max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        step_k = np.float(self.tableWidget_exp.item(1, 0).text())
        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        #max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        step_l = np.float(self.tableWidget_exp.item(2, 0).text())
        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        #max_l = np.float(self.tableWidget_exp.item(2, 3).text())
    
        ind_min_h = np.round((update_min_h-min_h)/step_h, 4).astype(int)
        ind_max_h = np.round((update_max_h-min_h)/step_h, 4).astype(int)
        
        ind_min_k = np.round((update_min_k-min_k)/step_k, 4).astype(int)
        ind_max_k = np.round((update_max_k-min_k)/step_k, 4).astype(int)
        
        ind_min_l = np.round((update_min_l-min_l)/step_l, 4).astype(int)
        ind_max_l = np.round((update_max_l-min_l)/step_l, 4).astype(int)
        
        h_slice = [ind_min_h, ind_max_h+1]
        k_slice = [ind_min_k, ind_max_k+1]
        l_slice = [ind_min_l, ind_max_l+1]
        
        self.I_exp = experimental.crop(self.I_exp, h_slice, k_slice, l_slice)
        self.sigma_sq_exp = experimental.crop(self.sigma_sq_exp, 
                                              h_slice, 
                                              k_slice, 
                                              l_slice)
                
    def plot_intensity_exp(self, data):
        
        index = self.comboBox_norm_exp.currentIndex()    
        norm = self.comboBox_norm_exp.itemText(index)
        
        slice_h = np.float(self.lineEdit_slice_h.text())
        slice_k = np.float(self.lineEdit_slice_k.text())
        slice_l = np.float(self.lineEdit_slice_l.text())
        
        step_h = np.float(self.tableWidget_exp.item(0, 0).text())
        size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))

        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        max_h = np.float(self.tableWidget_exp.item(0, 3).text())
        
        step_k = np.float(self.tableWidget_exp.item(1, 0).text())
        size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))

        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        max_k = np.float(self.tableWidget_exp.item(1, 3).text())
        
        step_l = np.float(self.tableWidget_exp.item(2, 0).text())
        size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))

        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        max_l = np.float(self.tableWidget_exp.item(2, 3).text())
        
        vmin = np.float(self.lineEdit_min_exp.text())
        vmax = np.float(self.lineEdit_max_exp.text())
        
        if (norm == 'Logarithmic'):
            normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            normalize = colors.Normalize(vmin=vmin, vmax=vmax)
        
        if (size_h == 1):
            ih = 0
            h = min_h
        else:
            ih = np.int(np.round((slice_h-min_h)/step_h))
            h = np.round(min_h+step_h*ih,4)

        if (size_k == 1):
            ik = 0
            k = min_k
        else:
            ik = np.int(np.round((slice_k-min_k)/step_k))
            k = np.round(min_k+step_k*ik,4)  
            
        if (size_l == 1):
            il = 0
            l = min_l
        else:
            il = np.int(np.round((slice_l-min_l)/step_l))
            l = np.round(min_l+step_l*il,4) 
            
        dh = np.float(max_h-min_h)/size_h
        dk = np.float(max_k-min_k)/size_k
        dl = np.float(max_l-min_l)/size_l
        
        extents_h = [min_k-dk/2, max_k+dk/2, min_l-dl/2, max_l+dl/2]
        extents_k = [min_h-dh/2, max_h+dh/2, min_l-dl/2, max_l+dl/2]
        extents_l = [min_h-dh/2, max_h+dh/2, min_k-dk/2, max_k+dk/2]
            
        fig = self.canvas_exp.figure
        fig.clear()   
        
        ax_h = fig.add_subplot(131)
        ax_k = fig.add_subplot(132)
        ax_l = fig.add_subplot(133)
        
        im_h = ax_h.imshow(data[ih,:,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_h,
                           zorder=100)
        
        im_k = ax_k.imshow(data[:,ik,:].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_k,
                           zorder=100)
        
        im_l = ax_l.imshow(data[:,:,il].T,
                           norm=normalize,
                           interpolation='nearest', 
                           origin='lower',
                           extent=extents_l,
                           zorder=100)
        
        trans_h = mtransforms.Affine2D()
        trans_k = mtransforms.Affine2D()
        trans_l = mtransforms.Affine2D()
        
        M_h = np.array([[self.B[1,1]/self.B[2,2],self.B[1,2]/self.B[2,2],0],
                        [self.B[2,1]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                        [0,0,1]])
        
        M_k = np.array([[self.B[0,0]/self.B[2,2],self.B[0,2]/self.B[2,2],0],
                        [self.B[2,0]/self.B[2,2],self.B[2,2]/self.B[2,2],0],
                        [0,0,1]])
        
        M_l = np.array([[self.B[0,0]/self.B[1,1],self.B[0,1]/self.B[1,1],0],
                        [self.B[1,0]/self.B[1,1],self.B[1,1]/self.B[1,1],0],
                        [0,0,1]])
                                
        scale_h = M_h[0,0]
        scale_k = M_h[0,0]
        scale_l = M_l[0,0]

        M_h[0,1] /= scale_h
        M_h[0,0] /= scale_h
        
        M_k[0,1] /= scale_k
        M_k[0,0] /= scale_k
        
        M_l[0,1] /= scale_l
        M_l[0,0] /= scale_l
        
        trans_h.set_matrix(M_h)
        trans_k.set_matrix(M_k)
        trans_l.set_matrix(M_l)
        
        offset_h = -np.dot(M_h,[0,min_l,0])[0]
        offset_k = -np.dot(M_k,[0,min_l,0])[0]
        offset_l = -np.dot(M_l,[0,min_k,0])[0]
        
        shift_h = mtransforms.Affine2D().translate(offset_h,0)
        shift_k = mtransforms.Affine2D().translate(offset_k,0)
        shift_l = mtransforms.Affine2D().translate(offset_l,0)
        
        ax_h.set_aspect(1/scale_h)
        ax_k.set_aspect(1/scale_k)
        ax_l.set_aspect(1/scale_l)
        
        trans_data_h = trans_h+shift_h+ax_h.transData
        trans_data_k = trans_k+shift_k+ax_k.transData
        trans_data_l = trans_l+shift_l+ax_l.transData
        
        im_h.set_transform(trans_data_h)
        im_k.set_transform(trans_data_k)
        im_l.set_transform(trans_data_l)
        
        ext_min_h = np.dot(M_h[0:2,0:2],extents_h[0::2])
        ext_max_h = np.dot(M_h[0:2,0:2],extents_h[1::2])
        
        ext_min_k = np.dot(M_k[0:2,0:2],extents_k[0::2])
        ext_max_k = np.dot(M_k[0:2,0:2],extents_k[1::2])
        
        ext_min_l = np.dot(M_l[0:2,0:2],extents_l[0::2])
        ext_max_l = np.dot(M_l[0:2,0:2],extents_l[1::2])
        
        ax_h.set_xlim(ext_min_h[0]+offset_h,ext_max_h[0]+offset_h)
        ax_h.set_ylim(ext_min_h[1],ext_max_h[1])
        
        ax_k.set_xlim(ext_min_k[0]+offset_k,ext_max_k[0]+offset_k)
        ax_k.set_ylim(ext_min_k[1],ext_max_k[1])
        
        ax_l.set_xlim(ext_min_l[0]+offset_l,ext_max_l[0]+offset_l)
        ax_l.set_ylim(ext_min_l[1],ext_max_l[1])
                
        ax_h.xaxis.tick_bottom()
        ax_k.xaxis.tick_bottom()
        ax_l.xaxis.tick_bottom()
        
        ax_h.axes.tick_params(labelsize='small')
        ax_k.axes.tick_params(labelsize='small')
        ax_l.axes.tick_params(labelsize='small')
        
        ax_h.set_title(r'$h='+str(h)+'$', fontsize='small') 
        ax_k.set_title(r'$k='+str(k)+'$', fontsize='small') 
        ax_l.set_title(r'$l='+str(l)+'$', fontsize='small') 
        
        ax_h.set_xlabel(r'$k$', fontsize='small')
        ax_h.set_ylabel(r'$l$', fontsize='small')
        
        ax_k.set_xlabel(r'$h$', fontsize='small')
        ax_k.set_ylabel(r'$l$', fontsize='small')
        
        ax_l.set_xlabel(r'$h$', fontsize='small') 
        ax_l.set_ylabel(r'$k$', fontsize='small') 
        
        ax_h.minorticks_on()
        ax_k.minorticks_on()
        ax_l.minorticks_on()
        
        fig.tight_layout(pad=3.24)
 
        cb = fig.colorbar(im_l, ax=[ax_h,ax_k,ax_l])
        cb.ax.minorticks_on()           
        
        if (norm == 'Linear'):
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
        else:
            cb.ax.xaxis.set_major_locator(plt.NullLocator())
            cb.ax.xaxis.set_minor_locator(plt.NullLocator())

        cb.ax.tick_params(labelsize='small') 

        with np.errstate(invalid='ignore'):
            self.canvas_exp.draw()

    def save_intensity_exp(self):

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        name, filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                 'Save file', 
                                                 '.', 
                                                 'pdf files *.pdf;;'+
                                                 'png files *.png',
                                                 options=options)
                      
        if (name):
            
            fig = self.canvas_exp.figure
            fig.savefig(name)

    def load_NXS(self):
         
        name, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file', 
                                                        '.', 
                                                        'NeXus files *.nxs',
                                                        options=options) 
                
        if (name):

            folder, filename = name.rsplit('/', 1)
            folder += '/'
            
            self.folder_exp = folder
            self.filename_exp = filename
    
            data = nxload(self.folder_exp+self.filename_exp)
            
            self.sigma_sq_nxs = \
                               data.MDHistoWorkspace.data.errors_squared.nxdata
            self.I_nxs = data.MDHistoWorkspace.data.signal.nxdata
            
            self.sigma_sq_nxs = np.swapaxes(self.sigma_sq_nxs, 0, 2)
            self.I_nxs = np.swapaxes(self.I_nxs, 0, 2)
                    
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')
            
            try:     
                Qh = data.MDHistoWorkspace.data.Q3
                Qk = data.MDHistoWorkspace.data.Q1
                Ql = data.MDHistoWorkspace.data.Q2
            except:
                Qh = data.MDHistoWorkspace.data['[H,0,0]']
                Qk = data.MDHistoWorkspace.data['[0,K,0]']
                Ql = data.MDHistoWorkspace.data['[0,0,L]']       
                
            Qh_min, Qk_min, Ql_min = Qh.min(), Qk.min(), Ql.min()
            Qh_max, Qk_max, Ql_max = Qh.max(), Qk.max(), Ql.max()
            
            mh, mk, ml = Qh.size, Qk.size, Ql.size
                    
            #del Qh, Qk, Ql
            
            size_h = np.int(mh-1)
            size_k = np.int(mk-1)
            size_l = np.int(ml-1)
            
            step_h = np.round((Qh_max-Qh_min)/size_h, 4)
            step_k = np.round((Qk_max-Qk_min)/size_k, 4)
            step_l = np.round((Ql_max-Ql_min)/size_l, 4)
            
            min_h = np.round(Qh_min+step_h/2, 4)
            min_k = np.round(Qk_min+step_k/2, 4)
            min_l = np.round(Ql_min+step_l/2, 4)
            
            max_h = np.round(Qh_max-step_h/2, 4)
            max_k = np.round(Qk_max-step_k/2, 4)
            max_l = np.round(Ql_max-step_l/2, 4)
            
            self.fixed_step_h = step_h
            self.fixed_size_h = size_h
            self.fixed_min_h = min_h
            self.fixed_max_h = max_h
            
            self.fixed_step_k = step_k
            self.fixed_size_k = size_k
            self.fixed_min_k = min_k
            self.fixed_max_k = max_k
            
            self.fixed_step_l = step_l
            self.fixed_size_l = size_l
            self.fixed_min_l = min_l
            self.fixed_max_l = max_l
            
            ih = size_h // 2
            h = np.round(min_h+step_h*ih,4)
     
            ik = size_k // 2
            k = np.round(min_k+step_k*ik,4)
    
            il = size_l // 2
            l = np.round(min_l+step_l*il,4)
    
            self.lineEdit_slice_h.setText(str(h))            
            self.lineEdit_slice_l.setText(str(l))         
            self.lineEdit_slice_k.setText(str(k))
            
            self.tableWidget_exp.setRowCount(3)
            self.tableWidget_exp.setColumnCount(4)
            
            lbl = 'step,size,min,max'    
            lbl = lbl.split(',')
            self.tableWidget_exp.setHorizontalHeaderLabels(lbl)
            
            lbl = 'h,k,l'
            lbl = lbl.split(',')
            self.tableWidget_exp.setVerticalHeaderLabels(lbl)
     
            self.reset_hkl()
            
            for i in range(3):
                 for j in range(4):
                     self.tableWidget_exp.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                     
            # ---
            
            self.tableWidget_calc.setRowCount(3)
            self.tableWidget_calc.setColumnCount(5)
            
            lbl = 'step,size,min,max,filter'    
            lbl = lbl.split(',')
            self.tableWidget_calc.setHorizontalHeaderLabels(lbl)
            
            lbl = 'h,k,l'
            lbl = lbl.split(',')
            self.tableWidget_calc.setVerticalHeaderLabels(lbl)
                     
            self.tableWidget_calc.setItem(0, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_h)))
            self.tableWidget_calc.setItem(0, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_h)))
            self.tableWidget_calc.setItem(0, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_h)))
            self.tableWidget_calc.setItem(0, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_h)))
            self.tableWidget_calc.setItem(0, 4, 
                                       QtWidgets.QTableWidgetItem('0.0'))

            self.tableWidget_calc.setItem(1, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_k)))
            self.tableWidget_calc.setItem(1, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_k)))
            self.tableWidget_calc.setItem(1, 0,
                                       QtWidgets.QTableWidgetItem(str(step_k)))
            self.tableWidget_calc.setItem(1, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_k)))
            self.tableWidget_calc.setItem(1, 4, 
                                       QtWidgets.QTableWidgetItem('0.0'))     

            self.tableWidget_calc.setItem(2, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_l)))
            self.tableWidget_calc.setItem(2, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_l)))
            self.tableWidget_calc.setItem(2, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_l)))
            self.tableWidget_calc.setItem(2, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_l)))
            self.tableWidget_calc.setItem(2, 4, 
                                       QtWidgets.QTableWidgetItem('0.0'))                            

            for i in range(3):
                for j in range(5):
                    self.tableWidget_calc.item(i, 
                                               j).setTextAlignment(alignment)
     
            self.tableWidget_calc.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
            for i in range(3):
                self.tableWidget_calc.item(i, 0).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

            self.tableWidget_calc.itemChanged.connect(self.update_calc)

    def reset_h(self):
        
        if (self.tableWidget_exp.rowCount()):
                        
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')
            
            step_h = self.fixed_step_h
            size_h = self.fixed_size_h
            min_h = self.fixed_min_h
            max_h = self.fixed_max_h
            
            self.tableWidget_exp.setItem(0, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_h)))
            self.tableWidget_exp.setItem(0, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_h)))
            self.tableWidget_exp.setItem(0, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_h)))
            self.tableWidget_exp.setItem(0, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_h)))
                            
            self.rebin_parameters_h()
            self.rebin_parameters_k()
            self.rebin_parameters_l()
                
            self.lineEdit_min_h.setText(str(min_h))
            self.lineEdit_max_h.setText(str(max_h))
              
            for j in range(4):
                self.tableWidget_exp.item(0, j).setTextAlignment(alignment)
     
            self.tableWidget_exp.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
            self.replot_intensity_exp()
        
    def reset_k(self):

        if (self.tableWidget_exp.rowCount()):
                        
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')
            
            step_k = self.fixed_step_k
            size_k = self.fixed_size_k
            min_k = self.fixed_min_k
            max_k = self.fixed_max_k
                    
            self.tableWidget_exp.setItem(1, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_k)))
            self.tableWidget_exp.setItem(1, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_k)))
            self.tableWidget_exp.setItem(1, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_k)))
            self.tableWidget_exp.setItem(1, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_k)))
            
            self.rebin_parameters_h()
            self.rebin_parameters_k()
            self.rebin_parameters_l()
                
            self.lineEdit_min_k.setText(str(min_k))
            self.lineEdit_max_k.setText(str(max_k))
              
            for j in range(4):
                self.tableWidget_exp.item(1, j).setTextAlignment(alignment)
     
            self.tableWidget_exp.horizontalHeader().setSectionResizeMode(
                                                QtWidgets.QHeaderView.Stretch)
            
            self.replot_intensity_exp()
       
    def reset_l(self):
 
        if (self.tableWidget_exp.rowCount()):
                       
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')
            
            step_l = self.fixed_step_l
            size_l = self.fixed_size_l
            min_l = self.fixed_min_l
            max_l = self.fixed_max_l
    
            self.tableWidget_exp.setItem(2, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_l)))
            self.tableWidget_exp.setItem(2, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_l)))
            self.tableWidget_exp.setItem(2, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_l)))
            self.tableWidget_exp.setItem(2, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_l)))
        
            self.rebin_parameters_h()
            self.rebin_parameters_k()
            self.rebin_parameters_l()
            
            self.lineEdit_min_l.setText(str(min_l))
            self.lineEdit_max_l.setText(str(max_l))
            
            for j in range(4):
                self.tableWidget_exp.item(2, j).setTextAlignment(alignment)
    
            self.tableWidget_exp.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
            self.replot_intensity_exp()
        
    def reset_hkl(self):
              
        if (self.tableWidget_exp.rowCount()):
          
            self.I_exp = self.I_nxs.copy(order='C')
            self.sigma_sq_exp = self.sigma_sq_nxs.copy(order='C')
            
            step_h = self.fixed_step_h
            size_h = self.fixed_size_h
            min_h = self.fixed_min_h
            max_h = self.fixed_max_h
            
            self.tableWidget_exp.setItem(0, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_h)))
            self.tableWidget_exp.setItem(0, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_h)))
            self.tableWidget_exp.setItem(0, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_h)))
            self.tableWidget_exp.setItem(0, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_h)))
            
            step_k = self.fixed_step_k
            size_k = self.fixed_size_k
            min_k = self.fixed_min_k
            max_k = self.fixed_max_k
                    
            self.tableWidget_exp.setItem(1, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_k)))
            self.tableWidget_exp.setItem(1, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_k)))
            self.tableWidget_exp.setItem(1, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_k)))
            self.tableWidget_exp.setItem(1, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_k)))
            
            step_l = self.fixed_step_l
            size_l = self.fixed_size_l
            min_l = self.fixed_min_l
            max_l = self.fixed_max_l
    
            self.tableWidget_exp.setItem(2, 2, 
                                        QtWidgets.QTableWidgetItem(str(min_l)))
            self.tableWidget_exp.setItem(2, 3, 
                                        QtWidgets.QTableWidgetItem(str(max_l)))
            self.tableWidget_exp.setItem(2, 0, 
                                       QtWidgets.QTableWidgetItem(str(step_l)))
            self.tableWidget_exp.setItem(2, 1, 
                                       QtWidgets.QTableWidgetItem(str(size_l)))
            
            self.rebin_parameters_h()
            self.rebin_parameters_k()
            self.rebin_parameters_l()
            
            self.lineEdit_min_h.setText(str(min_h))
            self.lineEdit_max_h.setText(str(max_h))
            
            self.lineEdit_min_k.setText(str(min_k))
            self.lineEdit_max_k.setText(str(max_k))
            
            self.lineEdit_min_l.setText(str(min_l))
            self.lineEdit_max_l.setText(str(max_l))
            
            for i in range(3):
                for j in range(4):
                    self.tableWidget_exp.item(i, j).setTextAlignment(alignment)
     
            self.tableWidget_exp.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            
            self.replot_intensity_exp()
       
    def rebin_parameters_h(self):
        
        self.comboBox_rebin_h.blockSignals(True)
        self.comboBox_rebin_h.clear()
        
        step_h = np.float(self.tableWidget_exp.item(0, 0).text())
        size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))

        min_h = np.float(self.tableWidget_exp.item(0, 2).text())
        max_h = np.float(self.tableWidget_exp.item(0, 3).text())   
   
        if (self.checkBox_centered_h.isChecked()):         
            round_min_h = np.round(min_h)
            round_max_h = np.round(max_h)  
            offset_min_h = np.round((round_min_h-min_h)/step_h, 4).astype(int)
            offset_max_h = np.round((round_max_h-min_h)/step_h, 4).astype(int)
            scale_h = experimental.factors(offset_max_h-offset_min_h)
        else:
            scale_h = experimental.factors(size_h-1)            
        
        if (self.checkBox_centered_h.isChecked()):
            mask_h = np.mod(1/(step_h*scale_h),1) == 0.
            scale_h = scale_h[mask_h]

        mask_h = step_h*scale_h < 1
        scale_h = scale_h[mask_h]
              
        for i in range(scale_h.size):
            step = np.round(step_h*scale_h[i], 4)
            size = (size_h-1) // scale_h[i]+1               
            self.comboBox_rebin_h.addItem('h-step : '+str(step)+', '+
                                          'h-size : '+str(size))
            
        self.comboBox_rebin_h.blockSignals(False)
                    
    def rebin_parameters_k(self):
        
        self.comboBox_rebin_k.blockSignals(True)
        self.comboBox_rebin_k.clear()
        
        step_k = np.float(self.tableWidget_exp.item(1, 0).text())
        size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))

        min_k = np.float(self.tableWidget_exp.item(1, 2).text())
        max_k = np.float(self.tableWidget_exp.item(1, 3).text())

        if (self.checkBox_centered_k.isChecked()):         
            round_min_k = np.round(min_k)
            round_max_k = np.round(max_k)  
            offset_min_k = np.round((round_min_k-min_k)/step_k, 4).astype(int)
            offset_max_k = np.round((round_max_k-min_k)/step_k, 4).astype(int)
            scale_k = experimental.factors(offset_max_k-offset_min_k)
        else:
            scale_k = experimental.factors(size_k-1)            
        
        if (self.checkBox_centered_k.isChecked()):
            mask_k = np.mod(1/(step_k*scale_k),1) == 0.
            scale_k = scale_k[mask_k]

        mask_k = step_k*scale_k < 1
        scale_k = scale_k[mask_k]
              
        for i in range(scale_k.size):
            step = np.round(step_k*scale_k[i], 4)
            size = (size_k-1) // scale_k[i]+1               
            self.comboBox_rebin_k.addItem('k-step : '+str(step)+', '+
                                          'k-size : '+str(size))
            
        self.comboBox_rebin_k.blockSignals(False)
        
    def rebin_parameters_l(self):
        
        self.comboBox_rebin_l.blockSignals(True)
        self.comboBox_rebin_l.clear()
        
        step_l = np.float(self.tableWidget_exp.item(2, 0).text())
        size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))

        min_l = np.float(self.tableWidget_exp.item(2, 2).text())
        max_l = np.float(self.tableWidget_exp.item(2, 3).text())
            
        if (self.checkBox_centered_l.isChecked()):         
            round_min_l = np.round(min_l)
            round_max_l = np.round(max_l)  
            offset_min_l = np.round((round_min_l-min_l)/step_l, 4).astype(int)
            offset_max_l = np.round((round_max_l-min_l)/step_l, 4).astype(int)
            scale_l = experimental.factors(offset_max_l-offset_min_l)
        else:
            scale_l = experimental.factors(size_l-1)            
        
        if (self.checkBox_centered_l.isChecked()):
            mask_l = np.mod(1/(step_l*scale_l),1) == 0.
            scale_l = scale_l[mask_l]

        mask_l = step_l*scale_l < 1
        scale_l = scale_l[mask_l]
              
        for i in range(scale_l.size):
            step = np.round(step_l*scale_l[i], 4)
            size = (size_l-1) // scale_l[i]+1               
            self.comboBox_rebin_l.addItem('l-step : '+str(step)+', '+
                                          'l-size : '+str(size))
            
        self.comboBox_rebin_l.blockSignals(False)
                
    def change_crop_min_h(self):

        if (self.tableWidget_exp.rowCount()):
        
            step_h = np.float(self.tableWidget_exp.item(0, 0).text())
            size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))
    
            min_h = np.float(self.tableWidget_exp.item(0, 2).text())
            max_h = np.float(self.tableWidget_exp.item(0, 3).text())
                    
            try:
                update_min_h = np.float(self.lineEdit_min_h.text())
                arg_min_h = np.round((update_min_h-min_h)/step_h, 
                                     4).astype(int)
                arg_max_h = np.round((max_h-min_h)/step_h, 4).astype(int)
                new_min_h = step_h*arg_min_h+min_h
                new_size_h = arg_max_h-arg_min_h+1
                if (new_min_h >= max_h or new_min_h < min_h):
                    new_min_h = min_h
                    new_size_h = size_h     
            except:   
                new_min_h = min_h
                new_size_h = size_h
                            
            self.lineEdit_min_h.setText(str(new_min_h))
            
            self.crop()
    
            self.tableWidget_exp.setItem(0, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_h))) 
            self.tableWidget_exp.setItem(0, 2, 
                                    QtWidgets.QTableWidgetItem(str(new_min_h))) 
            self.tableWidget_exp.item(0, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(0, 2).setTextAlignment(alignment)
                    
            self.rebin_parameters_h()
            
            slice_h = np.float(self.lineEdit_slice_h.text())
                    
            if (slice_h < min_h):
                h = min_h
            elif (slice_h > max_h):
                h = max_h
            else:
                ih = np.int(np.round((slice_h-min_h)/step_h))
                h = np.round(min_h+step_h*ih, 4)
                
            self.lineEdit_slice_h.setText(str(h))
    
            self.replot_intensity_exp()
        
    def change_crop_max_h(self):

        if (self.tableWidget_exp.rowCount()):
        
            step_h = np.float(self.tableWidget_exp.item(0, 0).text())
            size_h = np.int(np.float(self.tableWidget_exp.item(0, 1).text()))
    
            min_h = np.float(self.tableWidget_exp.item(0, 2).text())
            max_h = np.float(self.tableWidget_exp.item(0, 3).text())
                    
            try:
                update_max_h = np.float(self.lineEdit_max_h.text())
                arg_min_h = np.round((min_h-min_h)/step_h, 4).astype(int)
                arg_max_h = np.round((update_max_h-min_h)/step_h, 
                                     4).astype(int)
                new_max_h = step_h*arg_max_h+min_h
                new_size_h = arg_max_h-arg_min_h+1
                if (new_max_h <= min_h or new_max_h > max_h):
                    new_max_h = max_h
                    new_size_h = size_h
            except:   
                new_max_h = max_h
                new_size_h = size_h
                
            self.lineEdit_max_h.setText(str(new_max_h))
            
            self.crop()
    
            self.tableWidget_exp.setItem(0, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_h)))      
            self.tableWidget_exp.setItem(0, 3, 
                                    QtWidgets.QTableWidgetItem(str(new_max_h)))  
            self.tableWidget_exp.item(0, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(0, 3).setTextAlignment(alignment)
            
            self.rebin_parameters_h()
            
            slice_h = np.float(self.lineEdit_slice_h.text())
                    
            if (slice_h < min_h):
                h = min_h
            elif (slice_h > max_h):
                h = max_h
            else:
                ih = np.int(np.round((slice_h-min_h)/step_h))
                h = np.round(min_h+step_h*ih, 4)
                
            self.lineEdit_slice_h.setText(str(h))
    
            self.replot_intensity_exp()

    def change_crop_min_k(self):

        if (self.tableWidget_exp.rowCount()):
        
            step_k = np.float(self.tableWidget_exp.item(1, 0).text())
            size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))
    
            min_k = np.float(self.tableWidget_exp.item(1, 2).text())
            max_k = np.float(self.tableWidget_exp.item(1, 3).text())
                    
            try:
                update_min_k = np.float(self.lineEdit_min_k.text())
                arg_min_k = np.round((update_min_k-min_k)/step_k, 
                                     4).astype(int)
                arg_max_k = np.round((max_k-min_k)/step_k, 4).astype(int)
                new_min_k = step_k*arg_min_k+min_k
                new_size_k = arg_max_k-arg_min_k+1
                if (new_min_k >= max_k or new_min_k < min_k):
                    new_min_k = min_k
                    new_size_k = size_k     
            except:   
                new_min_k = min_k
                new_size_k = size_k
                
            self.lineEdit_min_k.setText(str(new_min_k))
            
            self.crop()
    
            self.tableWidget_exp.setItem(1, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_k))) 
            self.tableWidget_exp.setItem(1, 2, 
                                    QtWidgets.QTableWidgetItem(str(new_min_k))) 
            self.tableWidget_exp.item(1, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(1, 2).setTextAlignment(alignment)
            
            self.rebin_parameters_k()
            
            slice_k = np.float(self.lineEdit_slice_k.text())
                    
            if (slice_k < min_k):
                k = min_k
            elif (slice_k > max_k):
                k = max_k
            else:
                ik = np.int(np.round((slice_k-min_k)/step_k))
                k = np.round(min_k+step_k*ik, 4)
                
            self.lineEdit_slice_k.setText(str(k))
    
            self.replot_intensity_exp()
        
    def change_crop_max_k(self):
 
        if (self.tableWidget_exp.rowCount()):

            step_k = np.float(self.tableWidget_exp.item(1, 0).text())
            size_k = np.int(np.float(self.tableWidget_exp.item(1, 1).text()))
    
            min_k = np.float(self.tableWidget_exp.item(1, 2).text())
            max_k = np.float(self.tableWidget_exp.item(1, 3).text())
                    
            try:
                update_max_k = np.float(self.lineEdit_max_k.text())
                arg_min_k = np.round((min_k-min_k)/step_k, 4).astype(int)
                arg_max_k = np.round((update_max_k-min_k)/step_k, 
                                     4).astype(int)
                new_max_k = step_k*arg_max_k+min_k
                new_size_k = arg_max_k-arg_min_k+1
                if (new_max_k <= min_k or new_max_k > max_k):
                    new_max_k = max_k
                    new_size_k = size_k
            except:   
                new_max_k = max_k
                new_size_k = size_k
                
            self.lineEdit_max_k.setText(str(new_max_k))
    
            self.crop()
    
            self.tableWidget_exp.setItem(1, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_k)))      
            self.tableWidget_exp.setItem(1, 3, 
                                    QtWidgets.QTableWidgetItem(str(new_max_k)))  
            self.tableWidget_exp.item(1, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(1, 3).setTextAlignment(alignment)
            
            self.rebin_parameters_k()
            
            slice_k = np.float(self.lineEdit_slice_k.text())
                    
            if (slice_k < min_k):
                k = min_k
            elif (slice_k > max_k):
                k = max_k
            else:
                ik = np.int(np.round((slice_k-min_k)/step_k))
                k = np.round(min_k+step_k*ik, 4)
                
            self.lineEdit_slice_k.setText(str(k))
    
            self.replot_intensity_exp()

    def change_crop_min_l(self):

        if (self.tableWidget_exp.rowCount()):
        
            step_l = np.float(self.tableWidget_exp.item(2, 0).text())
            size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))
    
            min_l = np.float(self.tableWidget_exp.item(2, 2).text())
            max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                    
            try:
                update_min_l = np.float(self.lineEdit_min_l.text())
                arg_min_l = np.round((update_min_l-min_l)/step_l, 
                                     4).astype(int)
                arg_max_l = np.round((max_l-min_l)/step_l, 4).astype(int)
                new_min_l = step_l*arg_min_l+min_l
                new_size_l = arg_max_l-arg_min_l+1
                if (new_min_l >= max_l or new_min_l < min_l):
                    new_min_l = min_l
                    new_size_l = size_l     
            except:   
                new_min_l = min_l
                new_size_l = size_l
                
            self.lineEdit_min_l.setText(str(new_min_l))
    
            self.crop()
    
            self.tableWidget_exp.setItem(2, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_l))) 
            self.tableWidget_exp.setItem(2, 2, 
                                    QtWidgets.QTableWidgetItem(str(new_min_l))) 
            self.tableWidget_exp.item(2, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(2, 2).setTextAlignment(alignment)
            
            self.rebin_parameters_l()
            
            slice_l = np.float(self.lineEdit_slice_l.text())
                    
            if (slice_l < min_l):
                l = min_l
            elif (slice_l > max_l):
                l = max_l
            else:
                il = np.int(np.round((slice_l-min_l)/step_l))
                l = np.round(min_l+step_l*il, 4)
                
            self.lineEdit_slice_l.setText(str(l))
    
            self.replot_intensity_exp()
        
    def change_crop_max_l(self):
 
        if (self.tableWidget_exp.rowCount()):
            
            step_l = np.float(self.tableWidget_exp.item(2, 0).text())
            size_l = np.int(np.float(self.tableWidget_exp.item(2, 1).text()))
    
            min_l = np.float(self.tableWidget_exp.item(2, 2).text())
            max_l = np.float(self.tableWidget_exp.item(2, 3).text())
                    
            try:
                update_max_l = np.float(self.lineEdit_max_l.text())
                arg_min_l = np.round((min_l-min_l)/step_l, 4).astype(int)
                arg_max_l = np.round((update_max_l-min_l)/step_l, 
                                     4).astype(int)
                new_max_l = step_l*arg_max_l+min_l
                new_size_l = arg_max_l-arg_min_l+1
                if (new_max_l <= min_l or new_max_l > max_l):
                    new_max_l = max_l
                    new_size_l = size_l
            except:   
                new_max_l = max_l
                new_size_l = size_l
                
            self.lineEdit_max_l.setText(str(new_max_l))
            
            self.crop()
    
            self.tableWidget_exp.setItem(2, 1, 
                                   QtWidgets.QTableWidgetItem(str(new_size_l)))      
            self.tableWidget_exp.setItem(2, 3, 
                                    QtWidgets.QTableWidgetItem(str(new_max_l)))  
            self.tableWidget_exp.item(2, 1).setTextAlignment(alignment)
            self.tableWidget_exp.item(2, 3).setTextAlignment(alignment)
            
            self.rebin_parameters_l()
            
            slice_l = np.float(self.lineEdit_slice_l.text())
                    
            if (slice_l < min_l):
                l = min_l
            elif (slice_l > max_l):
                l = max_l
            else:
                il = np.int(np.round((slice_l-min_l)/step_l))
                l = np.round(min_l+step_l*il, 4)
                
            self.lineEdit_slice_l.setText(str(l))
    
            self.replot_intensity_exp()

    def change_rebin_h(self):
        
        index = self.comboBox_rebin_h.currentIndex()    
        data = self.comboBox_rebin_h.itemText(index)
        
        step, size = data.split(',')
        step, size = step.split(': ')[1], size.split(': ')[1]
        
        self.tableWidget_exp.setItem(0, 0, 
                                     QtWidgets.QTableWidgetItem(str(step)))       
        self.tableWidget_exp.setItem(0, 1, 
                                     QtWidgets.QTableWidgetItem(str(size)))
        self.tableWidget_exp.item(0, 0).setTextAlignment(alignment)
        self.tableWidget_exp.item(0, 1).setTextAlignment(alignment)
        
        self.rebin_parameters_h()

        self.rebin()
        
    def change_rebin_k(self):
        
        index = self.comboBox_rebin_k.currentIndex()    
        data = self.comboBox_rebin_k.itemText(index)
        
        step, size = data.split(',')
        step, size = step.split(': ')[1], size.split(': ')[1]
        
        self.tableWidget_exp.setItem(1, 0, 
                                     QtWidgets.QTableWidgetItem(str(step)))       
        self.tableWidget_exp.setItem(1, 1, 
                                     QtWidgets.QTableWidgetItem(str(size)))
        self.tableWidget_exp.item(1, 0).setTextAlignment(alignment)
        self.tableWidget_exp.item(1, 1).setTextAlignment(alignment)
        
        self.rebin_parameters_k()

        self.rebin()
        
    def change_rebin_l(self):
        
        index = self.comboBox_rebin_l.currentIndex()    
        data = self.comboBox_rebin_l.itemText(index)
        
        step, size = data.split(',')
        step, size = step.split(': ')[1], size.split(': ')[1]
        
        self.tableWidget_exp.setItem(2, 0, 
                                     QtWidgets.QTableWidgetItem(str(step)))       
        self.tableWidget_exp.setItem(2, 1, 
                                     QtWidgets.QTableWidgetItem(str(size)))
        self.tableWidget_exp.item(2, 0).setTextAlignment(alignment)
        self.tableWidget_exp.item(2, 1).setTextAlignment(alignment)
        
        self.rebin_parameters_l()

        self.rebin()
        
    def centered_integer(self):
        
        if (self.tableWidget_exp.rowCount()):
        
            self.comboBox_rebin_h.blockSignals(True)
            self.comboBox_rebin_k.blockSignals(True)
            self.comboBox_rebin_l.blockSignals(True)
            
            self.comboBox_rebin_h.clear()
            self.comboBox_rebin_k.clear()
            self.comboBox_rebin_l.clear()
            
            self.rebin_parameters_h()
            self.rebin_parameters_k()
            self.rebin_parameters_l()
            
            self.comboBox_rebin_h.blockSignals(False)
            self.comboBox_rebin_k.blockSignals(False)
            self.comboBox_rebin_l.blockSignals(False)
            
    def load_CIF(self):
  
        name, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file', 
                                                        '.', 
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options) 
        
        if (name):
                        
            folder, filename = name.rsplit('/', 1)
            folder += '/'      
                            
            self.folder = folder
            self.filename = filename
            
            u, \
            v, \
            w, \
            occupancy, \
            displacement, \
            moment, \
            site, \
            atms, \
            n_atm = crystal.unitcell(folder=self.folder, 
                                     filename=self.filename,
                                     occupancy=True,
                                     displacement=True,
                                     moment=True,
                                     site=True)
            
            if displacement.shape[1] == 1:
                displacement = np.column_stack((displacement,
                                                displacement,
                                                displacement,
                                                displacement*0,
                                                displacement*0,
                                                displacement*0))
                        
            gfactor = np.full(moment.size, 2.0)
            
            uni, ind, inv = np.unique(site, 
                                      return_index=True, 
                                      return_inverse=True)
                    
            group, hm = crystal.group(folder=self.folder, 
                                      filename=self.filename)
            
            a, \
            b, \
            c, \
            alpha, \
            beta, \
            gamma = crystal.parameters(folder=self.folder, 
                                       filename=self.filename)
            
            self.lineEdit_a.setText(str(a))
            self.lineEdit_b.setText(str(b))
            self.lineEdit_c.setText(str(c))
            
            self.lineEdit_alpha.setText(str(np.round(alpha*180/np.pi,8)))
            self.lineEdit_beta.setText(str(np.round(beta*180/np.pi,8)))
            self.lineEdit_gamma.setText(str(np.round(gamma*180/np.pi,8)))
            
            lat = crystal.lattice(a, b, c, alpha, beta, gamma)
            self.lineEdit_lat.setText(str(lat))
            
            self.a = np.float(self.lineEdit_a.text())
            self.b = np.float(self.lineEdit_b.text())
            self.c = np.float(self.lineEdit_c.text())
            
            self.alpha = np.deg2rad(float(self.lineEdit_alpha.text()))
            self.beta = np.deg2rad(float(self.lineEdit_beta.text()))
            self.gamma = np.deg2rad(float(self.lineEdit_gamma.text()))
            
            self.A, self.B, self.R = crystal.matrices(self.a, 
                                                      self.b, 
                                                      self.c, 
                                                      self.alpha, 
                                                      self.beta, 
                                                      self.gamma)
            
            self.C, self.D = crystal.orthogonalized(self.a, 
                                                    self.b, 
                                                    self.c, 
                                                    self.alpha, 
                                                    self.beta, 
                                                    self.gamma)
                
            atm = np.array([s.rstrip(numbers+pm) for s in atms.tolist()])
            
            nuc = np.array([s.lstrip(letters) for s in atms.tolist()])
            
            # ion = np.array([s[::(1-2*np.sum(np.array(
            #                [s.find(i) for i in ('+', '-')]) == 0))] 
            #                           for s in ion.tolist()])
            
            self.tableWidget_atm.setRowCount(ind.size)
            self.tableWidget_atm.setColumnCount(19)
            
            lbl = 'atom,ion,occupancy,'\
            'Uiso,U11,U22,U33,U23,U13,U12,mu,M1,M2,M3,g,u,v,w, '
            
            lbl = lbl.split(',')
            self.tableWidget_atm.setHorizontalHeaderLabels(lbl)
            
            lbl = ['%d' % (s+1) for s in range(ind.size)]
            self.tableWidget_atm.setVerticalHeaderLabels(lbl)
            
            hidden = [1,3,4,5,6,7,8,9,10,11,12,13,14]
            for i in range(19):
                if (i in hidden):
                    self.tableWidget_atm.setColumnHidden(i, True)
                else:
                    self.tableWidget_atm.setColumnHidden(i, False)    
            
            index = self.comboBox_type.currentIndex()
            data = self.comboBox_type.itemData(index)
            
            ion = []               

            for i in range(ind.size):
                combo = QtWidgets.QComboBox()
                combo.setObjectName('comboBox_site'+str(i))
                for t in data:
                    combo.addItem(t)
                self.tableWidget_atm.setCellWidget(i, 0, combo)
                index = combo.findText(atm[ind[i]], QtCore.Qt.MatchFixedString)
                if (index >= 0):
                     combo.setCurrentIndex(index)
                else:
                    index = combo.findText(atm[ind[i]][:2], 
                                           QtCore.Qt.MatchStartsWith)
                    if (index >= 0):
                        combo.setCurrentIndex(index)
                    else:
                        index = combo.findText(atm[ind[i]][0], 
                                               QtCore.Qt.MatchStartsWith)
                        if (index >= 0):
                            combo.setCurrentIndex(index)                    
                combo.currentIndexChanged.connect(self.combo_change_site)
                atm[ind[i]] = combo.currentText()
                
                mag_atm = j0_atm[j0_atm == atm[ind[i]]]
                mag_ion = j0_ion[j0_atm == atm[ind[i]]]

                combo = QtWidgets.QComboBox()
                combo.setObjectName('comboBox_ion'+str(i))
                for j in range(mag_ion.size):
                    combo.addItem(mag_atm[j]+mag_ion[j])
                if (mag_ion.size == 0):
                    combo.addItem('None')
                combo.currentIndexChanged.connect(self.combo_change_ion)
                                    
                self.tableWidget_atm.setCellWidget(i, 1, combo)
                ion.append(combo.currentText().lstrip(letters))

                occ = str(occupancy[ind[i]])
                self.tableWidget_atm.setItem(i, 2, 
                                             QtWidgets.QTableWidgetItem(occ))
                U11 = displacement[ind[i],0]
                U22 = displacement[ind[i],1]
                U33 = displacement[ind[i],2]
                U23 = displacement[ind[i],3]
                U13 = displacement[ind[i],4]
                U12 = displacement[ind[i],5]
                U = np.array([[U11,U12,U13], [U12,U22,U13], [U13,U23,U33]])   
                Up, _ = np.linalg.eig(np.dot(np.dot(self.D, U), 
                                             np.linalg.inv(self.D)))
                Uiso = np.mean(Up).real
                U11, U22, U33 = str(U11), str(U22), str(U33)
                U23, U13, U12, Uiso = str(U23), str(U13), str(U12), str(Uiso)
                self.tableWidget_atm.setItem(i, 3, 
                                             QtWidgets.QTableWidgetItem(Uiso))
                self.tableWidget_atm.setItem(i, 4, 
                                             QtWidgets.QTableWidgetItem(U11))
                self.tableWidget_atm.setItem(i, 5, 
                                             QtWidgets.QTableWidgetItem(U22))
                self.tableWidget_atm.setItem(i, 6, 
                                             QtWidgets.QTableWidgetItem(U33))
                self.tableWidget_atm.setItem(i, 7, 
                                             QtWidgets.QTableWidgetItem(U23))
                self.tableWidget_atm.setItem(i, 8, 
                                             QtWidgets.QTableWidgetItem(U13))
                self.tableWidget_atm.setItem(i, 9, 
                                             QtWidgets.QTableWidgetItem(U12))
                
                M1 = moment[ind[i],0]
                M2 = moment[ind[i],1]
                M3 = moment[ind[i],2]
                M = np.array([M1,M2,M3])
                mu = np.sum(np.dot(self.D, M)**2)
                M1, M2, M3, mu = str(M1), str(M2), str(M3), str(mu)
                self.tableWidget_atm.setItem(i, 10, 
                                             QtWidgets.QTableWidgetItem(mu))
                self.tableWidget_atm.setItem(i, 11, 
                                             QtWidgets.QTableWidgetItem(M1))
                self.tableWidget_atm.setItem(i, 12, 
                                             QtWidgets.QTableWidgetItem(M2))
                self.tableWidget_atm.setItem(i, 13, 
                                             QtWidgets.QTableWidgetItem(M3))
                
                g = str(gfactor[ind[i]])
                self.tableWidget_atm.setItem(i, 14, 
                                             QtWidgets.QTableWidgetItem(g))
                
                U, V, W = str(u[ind[i]]), str(v[ind[i]]), str(w[ind[i]])
                self.tableWidget_atm.setItem(i, 15, 
                                             QtWidgets.QTableWidgetItem(U))
                self.tableWidget_atm.setItem(i, 16, 
                                             QtWidgets.QTableWidgetItem(V))
                self.tableWidget_atm.setItem(i, 17, 
                                             QtWidgets.QTableWidgetItem(W))
                
                check = QtWidgets.QCheckBox()
                check.setObjectName('checkBox_site'+str(i))
                check.setCheckState(QtCore.Qt.Checked) 
                check.clicked.connect(self.check_change)
                self.tableWidget_atm.setCellWidget(i, 18, check)
                
            self.tableWidget_atm.itemChanged.connect(self.update_params)   
                        
            for i in range(ind.size):
                for j in range(2, 18):
                    self.tableWidget_atm.item(i, j).setTextAlignment(alignment)
                     
            self.tableWidget_atm.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            self.tableWidget_atm.horizontalHeader().setSectionResizeMode(2, 
                                        QtWidgets.QHeaderView.ResizeToContents)
            
            ion = np.array(ion)[inv]
            atm = atm[ind][inv]
            
            self.tableWidget_CIF.setRowCount(n_atm)
            self.tableWidget_CIF.setColumnCount(20)
            
            # selection = self.comboBox_type.currentIndex()
            
            lbl = 'site,atom,isotope,ion,occupancy,'\
                  'Uiso,U11,U22,U33,U23,U13,U12,mu,M1,M2,M3,g,u,v,w, '
                
            lbl = lbl.split(',')
            self.tableWidget_CIF.setHorizontalHeaderLabels(lbl)
            
            hidden = [3,5,6,7,8,9,10,11,12,13,14,15,16]
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)                 
            
            lbl = ['%d' % (s+1) for s in range(n_atm)]
            self.tableWidget_CIF.setVerticalHeaderLabels(lbl)
        
            for i in range(n_atm):
                s = str(1+site[i])
                self.tableWidget_CIF.setItem(i, 0, 
                                             QtWidgets.QTableWidgetItem(s))
                elm = atm[i]
                self.tableWidget_CIF.setItem(i, 1, 
                                             QtWidgets.QTableWidgetItem(elm))
                typ = ion[i]
                self.tableWidget_CIF.setItem(i, 2, 
                                              QtWidgets.QTableWidgetItem(typ))
                typ = nuc[i]
                self.tableWidget_CIF.setItem(i, 3, 
                                              QtWidgets.QTableWidgetItem(typ))
                occ = str(occupancy[i])
                self.tableWidget_CIF.setItem(i, 4, 
                                             QtWidgets.QTableWidgetItem(occ))
                U11 = displacement[i,0]
                U22 = displacement[i,1]
                U33 = displacement[i,2]
                U23 = displacement[i,3]
                U13 = displacement[i,4]
                U12 = displacement[i,5]
                U = np.array([[U11,U12,U13], [U12,U22,U13], [U13,U23,U33]])
                Up, _ = np.linalg.eig(np.dot(np.dot(self.D, U), 
                                             np.linalg.inv(self.D)))
                Uiso = np.mean(Up).real
                U11, U22, U33 = str(U11), str(U22), str(U33)
                U23, U13, U12, Uiso = str(U23), str(U13), str(U12), str(Uiso)
                self.tableWidget_CIF.setItem(i, 5, 
                                             QtWidgets.QTableWidgetItem(Uiso))
                self.tableWidget_CIF.setItem(i, 6, 
                                             QtWidgets.QTableWidgetItem(U11))
                self.tableWidget_CIF.setItem(i, 7, 
                                             QtWidgets.QTableWidgetItem(U22))
                self.tableWidget_CIF.setItem(i, 8, 
                                             QtWidgets.QTableWidgetItem(U33))
                self.tableWidget_CIF.setItem(i, 9, 
                                             QtWidgets.QTableWidgetItem(U23))
                self.tableWidget_CIF.setItem(i, 10, 
                                             QtWidgets.QTableWidgetItem(U13))
                self.tableWidget_CIF.setItem(i, 11, 
                                             QtWidgets.QTableWidgetItem(U12))
                M1 = moment[i,0]
                M2 = moment[i,1]
                M3 = moment[i,2]
                M = np.array([M1,M2,M3])
                mu = np.sum(np.dot(self.D, M)**2)
                M1, M2, M3, mu = str(M1), str(M2), str(M3), str(mu)
                self.tableWidget_CIF.setItem(i, 12, 
                                             QtWidgets.QTableWidgetItem(mu))
                self.tableWidget_CIF.setItem(i, 13, 
                                             QtWidgets.QTableWidgetItem(M1))
                self.tableWidget_CIF.setItem(i, 14, 
                                             QtWidgets.QTableWidgetItem(M2))
                self.tableWidget_CIF.setItem(i, 15, 
                                             QtWidgets.QTableWidgetItem(M3))
                g = str(gfactor[i])
                self.tableWidget_CIF.setItem(i, 16, 
                                             QtWidgets.QTableWidgetItem(g))
                U, V, W = str(u[i]), str(v[i]), str(w[i])
                self.tableWidget_CIF.setItem(i, 17, 
                                             QtWidgets.QTableWidgetItem(U))
                self.tableWidget_CIF.setItem(i, 18, 
                                             QtWidgets.QTableWidgetItem(V))
                self.tableWidget_CIF.setItem(i, 19, 
                                             QtWidgets.QTableWidgetItem(W))
                        
            for i in range(n_atm):
                for j in range(20):
                    self.tableWidget_CIF.item(i, j).setTextAlignment(alignment)
                    
            for i in range(n_atm):
                 for j in range(0, 19):
                     self.tableWidget_CIF.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
    
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(4, 
                                        QtWidgets.QHeaderView.ResizeToContents)
            
            self.comboBox_parameters.setCurrentIndex(0)
                                    
            self.lineEdit_n_atm.setText(str(n_atm))
           
            self.lineEdit_nu.setText(str(1))
            self.lineEdit_nv.setText(str(1))
            self.lineEdit_nw.setText(str(1))
    
            self.lineEdit_n.setText(str(n_atm))
             
            self.lineEdit_space_group.setText(str(group))
            self.lineEdit_space_group_hm.setText(hm)
            
            index = self.comboBox_centering.findText(hm[0], 
                                                    QtCore.Qt.MatchFixedString)
            if (index >= 0):
                 self.comboBox_centering.setCurrentIndex(index)
            index = self.comboBox_centering_ref.findText(hm[0], 
                                                    QtCore.Qt.MatchFixedString)
            if (index >= 0):
                 self.comboBox_centering_ref.setCurrentIndex(index)
        
    def save_CIF(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file', 
                                                        '.', 
                                                        'CIF files *.cif;;'\
                                                        'mCIF files *.mcif',
                                                        options=options)
                
        if (name):
        
            nu = np.int(self.lineEdit_nu.text())
            nv = np.int(self.lineEdit_nv.text())
            nw = np.int(self.lineEdit_nw.text())
            
            atm, occ = [], []
            mu, mv, mw = [], [], []
            U11, U22, U33, U13, U23, U12 = [], [], [], [], [], []
            u, v, w = [], [], []
            
            for i in range(self.tableWidget_CIF.rowCount()):
                if (not self.tableWidget_CIF.isRowHidden(i)):
                    atm.append(str(self.tableWidget_CIF.item(i, 1).text()))
                    occ.append(float(self.tableWidget_CIF.item(i, 4).text()))
                    U11.append(float(self.tableWidget_CIF.item(i, 6).text()))
                    U22.append(float(self.tableWidget_CIF.item(i, 7).text()))
                    U33.append(float(self.tableWidget_CIF.item(i, 8).text()))
                    U23.append(float(self.tableWidget_CIF.item(i, 9).text()))
                    U13.append(float(self.tableWidget_CIF.item(i, 10).text()))
                    U12.append(float(self.tableWidget_CIF.item(i, 11).text()))
                    mu.append(float(self.tableWidget_CIF.item(i, 13).text()))
                    mv.append(float(self.tableWidget_CIF.item(i, 14).text()))
                    mw.append(float(self.tableWidget_CIF.item(i, 15).text()))
                    u.append(float(self.tableWidget_CIF.item(i, 17).text()))
                    v.append(float(self.tableWidget_CIF.item(i, 18).text()))
                    w.append(float(self.tableWidget_CIF.item(i, 19).text()))
                
            atm, occ = np.array(atm), np.array(occ)
            disp = np.column_stack((U11,U22,U33,U23,U13,U12))
            mom = np.column_stack((mu,mv,mw))
            u, v, w = np.array(u), np.array(v), np.array(w)
            
            crystal.supercell(atm, 
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
                              folder=self.folder, 
                              filename=self.filename)

    def change_type(self):
                
        selection = self.comboBox_type.currentIndex()    
        data = self.comboBox_type.itemData(selection)

        disorder = self.comboBox_type.itemText(selection)

        if (self.tableWidget_atm.rowCount() > 0):
            
            for i in range(self.tableWidget_atm.rowCount()):
                widget = self.tableWidget_atm.cellWidget(i, 0)
                item = widget.currentText()
                combo = QtWidgets.QComboBox()
                combo.setObjectName('comboBox_site'+str(i))
                for t in data:
                    combo.addItem(t)
                self.tableWidget_atm.setCellWidget(i, 0, combo)
                index = combo.findText(item, QtCore.Qt.MatchFixedString)
                if (index >= 0):
                     combo.setCurrentIndex(index)
                else:
                    atm = item.strip(pm).strip(numbers)       
                    index = combo.findText(atm, QtCore.Qt.MatchStartsWith)
                    if (index >= 0):
                        combo.setCurrentIndex(index)
                    elif (len(atm) >= 2):
                        index = combo.findText(atm[:2], 
                                               QtCore.Qt.MatchStartsWith)
                        if (index >= 0):
                            combo.setCurrentIndex(index)
                        else:
                            index = combo.findText(atm[0], 
                                                   QtCore.Qt.MatchStartsWith)
                            if (index >= 0):
                                combo.setCurrentIndex(index)
                combo.currentIndexChanged.connect(self.combo_change_site)
                
                for j in range(self.tableWidget_CIF.rowCount()):
                    s = np.int(self.tableWidget_CIF.item(j, 0).text())-1
                    if (i == s):
                        atom = str(combo.currentText())
                        atm = atom.lstrip(numbers).rstrip(numbers+pm)
                        nuc = atom.rstrip(numbers+pm).rstrip(letters)
                        ion = atom.lstrip(numbers).lstrip(letters)
                        self.tableWidget_CIF.setItem(j, 1, 
                                               QtWidgets.QTableWidgetItem(atm))
                        self.tableWidget_CIF.setItem(j, 2, 
                                               QtWidgets.QTableWidgetItem(nuc))
                        self.tableWidget_CIF.setItem(j, 3, 
                                               QtWidgets.QTableWidgetItem(ion))
                        
                        self.tableWidget_CIF.item(j, 
                                                 1).setTextAlignment(alignment)
                        self.tableWidget_CIF.item(j, 
                                                 2).setTextAlignment(alignment)
                        self.tableWidget_CIF.item(j, 
                                                 3).setTextAlignment(alignment)
                
                mag_atm = j0_atm[j0_atm == atm]
                mag_ion = j0_ion[j0_atm == atm]
    
                combo_ion = QtWidgets.QComboBox()
                combo_ion.setObjectName('comboBox_ion'+str(i))
                for j in range(mag_ion.size):
                    combo_ion.addItem(mag_atm[j]+mag_ion[j])
                if (mag_ion.size == 0):
                    combo_ion.addItem('None')
                mag_ion = str(combo.currentText()
                                              ).lstrip(numbers).lstrip(letters)
                self.tableWidget_atm.setCellWidget(i, 1, combo_ion)
                
                index = combo_ion.findText(atm+ion, QtCore.Qt.MatchFixedString)
                if (index >= 0):
                     combo_ion.setCurrentIndex(index)

                if (disorder == 'Neutron'):                
                    for j in range(self.tableWidget_CIF.rowCount()):
                        s = np.int(self.tableWidget_CIF.item(j, 0).text())-1
                        if (j == s):
                            self.tableWidget_CIF.setItem(j, 3, 
                                           QtWidgets.QTableWidgetItem(mag_ion))
        
                        self.tableWidget_CIF.item(j, 
                                                 1).setTextAlignment(alignment)
                        self.tableWidget_CIF.item(j, 
                                                 2).setTextAlignment(alignment)
                        self.tableWidget_CIF.item(j, 
                                                 3).setTextAlignment(alignment)
        
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(4, 
                                        QtWidgets.QHeaderView.ResizeToContents)
                
        hidden = [1,3,4,5,6,7,8,9,10,11,12,13,14]
        for i in range(19):
            if i in hidden:
                self.tableWidget_atm.setColumnHidden(i, True)
            else:
                self.tableWidget_atm.setColumnHidden(i, False)
                    
        if (disorder == 'Neutron'):
            hidden = [3,5,6,7,8,9,10,11,12,13,14,15,16]
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)    
            self.tabWidget_disorder.setTabEnabled(0,True)
            self.tabWidget_disorder.setTabEnabled(1,True)
            self.tabWidget_disorder.setTabEnabled(2,True)
            self.tabWidget_calc.setTabEnabled(0,True)
            self.tabWidget_calc.setTabEnabled(1,True)
            self.tabWidget_calc.setTabEnabled(2,True)
            self.checkBox_mag.setEnabled(True)
            self.checkBox_occ.setEnabled(True)
            self.checkBox_dis.setEnabled(True)
            self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
            self.checkBox_occ.setCheckState(QtCore.Qt.Checked)
            self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)     
            self.comboBox_parameters.blockSignals(True)        
            self.comboBox_parameters.clear()
            self.comboBox_parameters.addItem('Site parameters')
            self.comboBox_parameters.addItem('Structural parameters')
            self.comboBox_parameters.addItem('Magnetic parameters')         
            self.comboBox_parameters.blockSignals(False)      
        else:
            hidden = [2,5,6,7,8,9,10,11,12,13,14,15,16]
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)
            self.tabWidget_disorder.setTabEnabled(0,False)
            self.tabWidget_disorder.setTabEnabled(1,True)
            self.tabWidget_disorder.setTabEnabled(2,True)
            self.tabWidget_calc.setTabEnabled(0,False)
            self.tabWidget_calc.setTabEnabled(1,True)
            self.tabWidget_calc.setTabEnabled(2,True)
            self.checkBox_mag.setEnabled(False)
            self.checkBox_occ.setEnabled(True)
            self.checkBox_dis.setEnabled(True)
            self.checkBox_mag.setCheckState(QtCore.Qt.Unchecked)
            self.checkBox_occ.setCheckState(QtCore.Qt.Checked)
            self.checkBox_dis.setCheckState(QtCore.Qt.Unchecked)
            self.comboBox_parameters.blockSignals(True)        
            self.comboBox_parameters.clear()
            self.comboBox_parameters.addItem('Site parameters')
            self.comboBox_parameters.addItem('Structural parameters')
            self.comboBox_parameters.blockSignals(False) 
            
    def change_parameters(self):

        selection = self.comboBox_parameters.currentIndex()    
        parameters = self.comboBox_parameters.itemText(selection)
        
        selection = self.comboBox_type.currentIndex()    
        disorder = self.comboBox_type.itemText(selection)
        
        if (parameters == 'Site parameters'):
            hidden = [1,3,4,5,6,7,8,9,10,11,12,13,14]
            for i in range(19):
                if i in hidden:
                    self.tableWidget_atm.setColumnHidden(i, True)
                else:
                    self.tableWidget_atm.setColumnHidden(i, False) 
            if (disorder == 'Neutron'):
                hidden = [3,5,6,7,8,9,10,11,12,13,14,15,16]
            else:
                hidden = [2,5,6,7,8,9,10,11,12,13,14,15,16]               
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)
        elif (parameters == 'Structural parameters'):
            hidden = [0,1,2,3,10,11,12,13,14,15,16,17,18]
            for i in range(19):
                if i in hidden:
                    self.tableWidget_atm.setColumnHidden(i, True)
                else:
                    self.tableWidget_atm.setColumnHidden(i, False)
            if (disorder == 'Neutron'):
                hidden = [3,4,6,7,8,9,10,11,12,13,14,15,16]
            else:
                hidden = [2,4,6,7,8,9,10,11,12,13,14,15,16]                
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)
        else:
            hidden = [0,2,3,4,5,6,7,8,9,10,15,16,17]
            for i in range(19):
                if i in hidden:
                    self.tableWidget_atm.setColumnHidden(i, True)
                else:
                    self.tableWidget_atm.setColumnHidden(i, False) 
            hidden = [2,4,5,6,7,8,9,10,11,13,14,15,16]
            for i in range(20):
                if i in hidden:
                    self.tableWidget_CIF.setColumnHidden(i, True)
                else:
                    self.tableWidget_CIF.setColumnHidden(i, False)
                    
    def update_params(self, item):
        
        site, j = item.row(), item.column()

        self.tableWidget_atm.blockSignals(True)
        
        if (j == 2):
            
            try:
                occ = np.float(self.tableWidget_atm.item(site, j).text())
                if (occ > 1):
                    occ = 1.0
                elif (occ < 0):
                    occ = 0.0
            except:
                occ = 1.0            
            
            occ = np.str(np.round(occ,4))
            
            self.tableWidget_atm.setItem(site, j, 
                                         QtWidgets.QTableWidgetItem(occ))
            self.tableWidget_atm.item(site, j).setTextAlignment(alignment)
                                
            for i in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
                if (site == s):
                    self.tableWidget_CIF.setItem(i, 4, 
                                               QtWidgets.QTableWidgetItem(occ))
                    self.tableWidget_CIF.item(i, 4).setTextAlignment(alignment)
       
        elif (j >= 4 and j <= 9):
                          
            try:
                Uij = np.float(self.tableWidget_atm.item(site, j).text())
            except:
                Uij = 0.0       
            
            Uij = np.str(np.round(Uij,4))

            self.tableWidget_atm.setItem(site, j, 
                                         QtWidgets.QTableWidgetItem(Uij))
            self.tableWidget_atm.item(site, j).setTextAlignment(alignment)
                   
            U11 = float(self.tableWidget_atm.item(site, 4).text())
            U22 = float(self.tableWidget_atm.item(site, 5).text())
            U33 = float(self.tableWidget_atm.item(site, 6).text())
            U23 = float(self.tableWidget_atm.item(site, 7).text())
            U13 = float(self.tableWidget_atm.item(site, 8).text())
            U12 = float(self.tableWidget_atm.item(site, 9).text())
            
            U = np.array([[U11,U12,U13], [U12,U22,U13], [U13,U23,U33]])
            Up, _ = np.linalg.eig(np.dot(np.dot(self.D, U), 
                                         np.linalg.inv(self.D)))
            Uiso = np.str(np.round(np.mean(Up).real,4))
    
            self.tableWidget_atm.setItem(site, 3, 
                                         QtWidgets.QTableWidgetItem(Uiso))
            self.tableWidget_atm.item(site, 3).setTextAlignment(alignment)
    
            for i in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
                if (site == s):
                    self.tableWidget_CIF.setItem(i, 5, 
                                              QtWidgets.QTableWidgetItem(Uiso))
                    self.tableWidget_CIF.setItem(i, 2+j, 
                                               QtWidgets.QTableWidgetItem(Uij))
                    self.tableWidget_CIF.item(i, 
                                              5).setTextAlignment(alignment)
                    self.tableWidget_CIF.item(i, 
                                              2+j).setTextAlignment(alignment)

        elif (j >= 11 and j <= 13):

            try:
                Mi = np.float(self.tableWidget_atm.item(site, j).text())
            except:
                Mi = 0.0
            
            Mi = np.str(np.round(Mi,4))
            
            self.tableWidget_atm.setItem(site, j, 
                                         QtWidgets.QTableWidgetItem(Mi))
            self.tableWidget_atm.item(site, j).setTextAlignment(alignment)
                   
            M1 = float(self.tableWidget_atm.item(site, 11).text())
            M2 = float(self.tableWidget_atm.item(site, 12).text())
            M3 = float(self.tableWidget_atm.item(site, 13).text())

            M = np.array([M1,M2,M3])
            mu = np.str(np.round(np.sum(np.dot(self.D, M)**2),4))
                
            self.tableWidget_atm.setItem(site, 10, 
                                         QtWidgets.QTableWidgetItem(mu))
            self.tableWidget_atm.item(site, 10).setTextAlignment(alignment)
    
            for i in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
                if (site == s):
                    self.tableWidget_CIF.setItem(i, 12, 
                                              QtWidgets.QTableWidgetItem(mu))
                    self.tableWidget_CIF.setItem(i, 2+j, 
                                               QtWidgets.QTableWidgetItem(Mi))
                    self.tableWidget_CIF.item(i, 
                                              12).setTextAlignment(alignment)
                    self.tableWidget_CIF.item(i, 
                                              2+j).setTextAlignment(alignment)

        self.tableWidget_atm.blockSignals(False)

    def supercell_n(self):
        
        try:
            n_atm = np.int(self.lineEdit_n_atm.text())
            try:
                nu = np.int(self.lineEdit_nu.text())
            except:
                nu = 1
                self.lineEdit_nu.setText('1')   
                
            try:
                nv = np.int(self.lineEdit_nv.text())
            except:
                nv = 1
                self.lineEdit_nv.setText('1')   
                
            try:
                nw = np.int(self.lineEdit_nw.text())
            except:
                nw = 1
                self.lineEdit_nw.setText('1')               
            self.lineEdit_n.setText(str(n_atm*nu*nv*nw))  
        except:
            self.lineEdit_n.setText('')            
     
    def check_change(self):
        
        check = self.sender()
        index = self.tableWidget_atm.indexAt(check.pos())

        site = index.row()
        
        n_atm = np.int(self.lineEdit_n_atm.text())
        
        for i in range(self.tableWidget_CIF.rowCount()):
            s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
            if (site == s):
                if check.isChecked():
                    self.tableWidget_CIF.setRowHidden(i, False)
                    n_atm += 1
                else:
                    self.tableWidget_CIF.setRowHidden(i, True)
                    n_atm -= 1
        
        self.lineEdit_n_atm.setText(str(n_atm))  
        
        nu = np.int(self.lineEdit_nu.text())
        nv = np.int(self.lineEdit_nv.text())
        nw = np.int(self.lineEdit_nw.text())
        
        self.lineEdit_n.setText(str(n_atm*nu*nv*nw))
        
    def combo_change_ion(self):
        
        combo = self.sender()
        index = self.tableWidget_atm.indexAt(combo.pos())
        
        site = index.row()
 
        atom = str(combo.currentText())
        ion = atom.lstrip(numbers).lstrip(letters)
        
        for i in range(self.tableWidget_CIF.rowCount()):
            s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
            if (site == s):
                self.tableWidget_CIF.setItem(i, 3, 
                                             QtWidgets.QTableWidgetItem(ion))
                self.tableWidget_CIF.item(i, 3).setTextAlignment(alignment)

        self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
        self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(4, 
                                        QtWidgets.QHeaderView.ResizeToContents)
        
    def combo_change_site(self):
        
        combo = self.sender()
        index = self.tableWidget_atm.indexAt(combo.pos())
        
        site = index.row()
 
        atom = str(combo.currentText())
        atm = atom.lstrip(numbers).rstrip(numbers+pm)
        nuc = atom.rstrip(numbers+pm).rstrip(letters)
        ion = atom.lstrip(numbers).lstrip(letters)
               
        for i in range(self.tableWidget_CIF.rowCount()):
            s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
            if (site == s):
                self.tableWidget_CIF.setItem(i, 1, 
                                             QtWidgets.QTableWidgetItem(atm))
                self.tableWidget_CIF.setItem(i, 2, 
                                             QtWidgets.QTableWidgetItem(nuc))
                self.tableWidget_CIF.setItem(i, 3, 
                                             QtWidgets.QTableWidgetItem(ion))
                self.tableWidget_CIF.item(i, 1).setTextAlignment(alignment)
                self.tableWidget_CIF.item(i, 2).setTextAlignment(alignment)
                self.tableWidget_CIF.item(i, 3).setTextAlignment(alignment)

        mag_atm = j0_atm[j0_atm == atm]
        mag_ion = j0_ion[j0_atm == atm]

        combo_ion = QtWidgets.QComboBox()
        combo_ion.setObjectName('comboBox_ion'+str(i))
        for j in range(mag_ion.size):
            combo_ion.addItem(mag_atm[j]+mag_ion[j])
        if (mag_ion.size == 0):
            combo_ion.addItem('None')
        mag_ion = str(combo.currentText()).lstrip(numbers).lstrip(letters)
        self.tableWidget_atm.setCellWidget(i, 1, combo_ion)
       
        selection = self.comboBox_type.currentIndex()    
        disorder = self.comboBox_type.itemText(selection)
        
        index = combo_ion.findText(atm+ion, QtCore.Qt.MatchFixedString)
        if (index >= 0):
             combo_ion.setCurrentIndex(index)
        
        if (disorder == 'Neutron'):
            for i in range(self.tableWidget_CIF.rowCount()):
                s = np.int(self.tableWidget_CIF.item(i, 0).text())-1
                if (site == s):
                    self.tableWidget_CIF.setItem(i, 3, 
                                           QtWidgets.QTableWidgetItem(mag_ion))
                    self.tableWidget_CIF.item(i, 3).setTextAlignment(alignment)
                        
        self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
        self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(4, 
                                        QtWidgets.QHeaderView.ResizeToContents)
            
    def file_new(self):
        
        self.lineEdit_n_atm.setText('')
        
        self.lineEdit_nu.setText('')
        self.lineEdit_nv.setText('')
        self.lineEdit_nw.setText('')
        
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
        
        # ---
        
        self.comboBox_rebin_h.blockSignals(True)
        self.comboBox_rebin_k.blockSignals(True)
        self.comboBox_rebin_l.blockSignals(True)
        
        self.comboBox_rebin_h.clear()
        self.comboBox_rebin_k.clear()
        self.comboBox_rebin_l.clear()
        
        self.comboBox_rebin_h.blockSignals(False)
        self.comboBox_rebin_k.blockSignals(False)
        self.comboBox_rebin_l.blockSignals(False)
        
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
        
        self.canvas_exp.figure.clear()  
        with np.errstate(invalid='ignore'):
            self.canvas_exp.draw()
        
        # ---
        
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

        self.allocated = False
        self.stop = False        
        self.batch = 0
        self.iteration = 0
        self.progress = 0
        self.progressBar_ref.setValue(self.progress)

        self.canvas_ref.figure.clear()         
        with np.errstate(invalid='ignore'):
            self.canvas_ref.draw()
            
        self.canvas_chi_sq.figure.clear()         
        with np.errstate(invalid='ignore'):
            self.canvas_chi_sq.draw()
            
        # ---
            
        self.tableWidget_pairs_1d.setRowCount(0)
        self.tableWidget_pairs_1d.setColumnCount(0)
        
        self.tableWidget_pairs_3d.setRowCount(0)
        self.tableWidget_pairs_3d.setColumnCount(0)
        
        self.canvas_1d.figure.clear()         
        with np.errstate(invalid='ignore'):
            self.canvas_1d.draw()
            
        self.canvas_3d.figure.clear()         
        with np.errstate(invalid='ignore'):
            self.canvas_3d.draw()
            
        # ---
            
        self.tableWidget_calc.setRowCount(0)
        self.tableWidget_calc.setColumnCount(0)
        
        self.lineEdit_order_calc.setText('2')
       
        self.lineEdit_slice_calc.setText('0.0')
        
        self.lineEdit_min_calc.setText('')       
        self.lineEdit_max_calc.setText('')    
        
        self.canvas_calc.figure.clear()         
        with np.errstate(invalid='ignore'):
            self.canvas_calc.draw()
                    
    def file_save_as(self):
        
        name, \
        filters = QtWidgets.QFileDialog.getSaveFileName(self, 
                                                        'Save file',
                                                        '.', 
                                                        'ini files *.ini',
                                                        options=options)        
        if (name):
            
            try:
                fname, ext = name.rsplit('.', 1)
                if (ext != 'ini'):
                    name += '.ini'
                    fname, ext = name.rsplit('.', 1)
            except:
                name += '.ini'
                fname, ext = name.rsplit('.', 1)
                    
            settings = QtCore.QSettings(name, QtCore.QSettings.IniFormat)
            save_gui(self, settings)
            
            try:
                folder = self.folder
                filename = self.filename
                
                copyfile(folder+filename, self.fname+'.cif')
                            
                folder, filename = self.fname.rsplit('/', 1)
                
                folder += '/'   
                filename = filename+'.cif'
                
                self.folder = folder
                self.filename = filename
            except:
                pass
            
            try:
                folder = self.folder_exp
                filename = self.filename_exp
                
                copyfile(folder+filename, self.fname+'.nxs')
                            
                folder, filename = self.fname.rsplit('/', 1)
                
                folder += '/'   
                filename = filename+'.nxs'
                
                self.folder_exp = folder
                self.filename_exp = filename
            except:
                pass
            
            if (self.tableWidget_exp.rowCount() > 0):
                np.save(fname+'-intensity-nxs.npy', self.I_nxs)
                np.save(fname+'-error-nxs.npy', self.sigma_sq_nxs)
                np.save(fname+'-intensity-exp.npy', self.I_exp)
                np.save(fname+'-error-exp.npy', self.sigma_sq_exp)
                
            if (self.progress > 0):
                np.save(self.fname+'-intensity-obs.npy', self.I_obs)    
                
            try:
                self.recalculated
                np.save(self.fname+'-intensity-recalc.npy', self.I_recalc)    
            except:
                pass
                
            self.fname = fname
                
    def file_save(self):
        
        try:
            
            name = self.fname+'.ini'
            
            settings = QtCore.QSettings(name, QtCore.QSettings.IniFormat)
            save_gui(self, settings)
            
            try:
                folder = self.folder
                filename = self.filename
                
                copyfile(folder+filename, self.fname+'.cif')
                            
                folder, filename = self.fname.rsplit('/', 1)
                
                folder += '/'   
                filename = filename+'.cif'
                
                self.folder = folder
                self.filename = filename
            except:
                pass
            
            try:
                folder = self.folder_exp
                filename = self.filename_exp
                
                copyfile(folder+filename, self.fname+'.nxs')
                            
                folder, filename = self.fname.rsplit('/', 1)
                
                folder += '/'   
                filename = filename+'.nxs'
                
                self.folder_exp = folder
                self.filename_exp = filename
            except:
                pass
            
            if (self.tableWidget_exp.rowCount() > 0):
                np.save(self.fname+'-intensity-nxs.npy', self.I_nxs)
                np.save(self.fname+'-error-nxs.npy', self.sigma_sq_nxs)
                np.save(self.fname+'-intensity-exp.npy', self.I_exp)
                np.save(self.fname+'-error-exp.npy', self.sigma_sq_exp)
                
            if (self.progress > 0):
                np.save(self.fname+'-intensity-obs.npy', self.I_obs)    
                
            try:
                self.recalculated
                np.save(self.fname+'-intensity-recalc.npy', self.I_recalc)    
            except:
                pass
               
        except:
            
            self.file_save_as()
        
    def file_open(self):
              
        name, \
        filters = QtWidgets.QFileDialog.getOpenFileName(self, 
                                                        'Open file',
                                                        '.', 
                                                        'ini files *.ini',
                                                        options=options)        

        if (name):
                        
            fname, ext = name.rsplit('.', 1)
            
            folder, filename = fname.rsplit('/', 1)
            folder += '/'
            
            self.file_new()
            
            self.comboBox_rebin_h.blockSignals(True)
            self.comboBox_rebin_k.blockSignals(True)
            self.comboBox_rebin_l.blockSignals(True)
    
            settings = QtCore.QSettings(name, QtCore.QSettings.IniFormat)
            load_gui(self, settings)
            
            try:                
                self.a = np.float(self.lineEdit_a.text())
                self.b = np.float(self.lineEdit_b.text())
                self.c = np.float(self.lineEdit_c.text())
                
                self.alpha = np.float(self.lineEdit_alpha.text())*np.pi/180
                self.beta = np.float(self.lineEdit_beta.text())*np.pi/180
                self.gamma = np.float(self.lineEdit_gamma.text())*np.pi/180
                
                self.A, self.B, self.R = crystal.matrices(self.a, 
                                                          self.b, 
                                                          self.c, 
                                                          self.alpha, 
                                                          self.beta, 
                                                          self.gamma)
            
                self.C, self.D = crystal.orthogonalized(self.a, 
                                                        self.b, 
                                                        self.c, 
                                                        self.alpha, 
                                                        self.beta, 
                                                        self.gamma)
                                
                self.folder = folder
                self.filename = filename+'.cif'
            except:
                pass
                                              
            self.change_parameters()      

            self.tableWidget_atm.itemChanged.connect(self.update_params)

            for i in range(self.tableWidget_CIF.rowCount()):
                for j in range(0, 19):
                     self.tableWidget_CIF.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                for j in range(self.tableWidget_CIF.columnCount()):
                     self.tableWidget_CIF.item(i, 
                                               j).setTextAlignment(alignment)
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            self.tableWidget_CIF.horizontalHeader().setSectionResizeMode(4, 
                                        QtWidgets.QHeaderView.ResizeToContents)
            
            for i in range(self.tableWidget_atm.rowCount()):
                for j in range(2, 18):
                     self.tableWidget_atm.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                for j in range(2, 18):
                     self.tableWidget_atm.item(i, 
                                               j).setTextAlignment(alignment)
            self.tableWidget_atm.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
            self.tableWidget_atm.horizontalHeader().setSectionResizeMode(2, 
                                        QtWidgets.QHeaderView.ResizeToContents)
                                   
            if (self.tableWidget_exp.rowCount() > 0):
                                                                            
                self.folder_exp = folder
                self.filename_exp = filename+'.nxs'
                
                data = nxload(self.folder_exp+self.filename_exp)
                
                try:     
                    Qh = data.MDHistoWorkspace.data.Q3
                    Qk = data.MDHistoWorkspace.data.Q1
                    Ql = data.MDHistoWorkspace.data.Q2
                except:
                    Qh = data.MDHistoWorkspace.data['[H,0,0]']
                    Qk = data.MDHistoWorkspace.data['[0,K,0]']
                    Ql = data.MDHistoWorkspace.data['[0,0,L]']       
                     
                Qh_min, Qk_min, Ql_min = Qh.min(), Qk.min(), Ql.min()
                Qh_max, Qk_max, Ql_max = Qh.max(), Qk.max(), Ql.max()
                
                mh, mk, ml = Qh.size, Qk.size, Ql.size
                                        
                size_h = np.int(mh-1)
                size_k = np.int(mk-1)
                size_l = np.int(ml-1)
                
                step_h = np.round((Qh_max-Qh_min)/size_h, 4)
                step_k = np.round((Qk_max-Qk_min)/size_k, 4)
                step_l = np.round((Ql_max-Ql_min)/size_l, 4)
                
                min_h = np.round(Qh_min+step_h/2, 4)
                min_k = np.round(Qk_min+step_k/2, 4)
                min_l = np.round(Ql_min+step_l/2, 4)
                
                max_h = np.round(Qh_max-step_h/2, 4)
                max_k = np.round(Qk_max-step_k/2, 4)
                max_l = np.round(Ql_max-step_l/2, 4)
                
                self.fixed_step_h = step_h
                self.fixed_size_h = size_h
                self.fixed_min_h = min_h
                self.fixed_max_h = max_h
                
                self.fixed_step_k = step_k
                self.fixed_size_k = size_k
                self.fixed_min_k = min_k
                self.fixed_max_k = max_k
                
                self.fixed_step_l = step_l
                self.fixed_size_l = size_l
                self.fixed_min_l = min_l
                self.fixed_max_l = max_l
                
                for i in range(3):
                    for j in range(4):
                        self.tableWidget_exp.item(i, 
                                                 j).setTextAlignment(alignment)
         
                self.tableWidget_exp.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
                
                for i in range(3):
                     for j in range(4):
                         self.tableWidget_exp.item(i, j).setFlags(
                          QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                
                index_h = self.comboBox_rebin_h.currentIndex()   
                index_k = self.comboBox_rebin_k.currentIndex()   
                index_l = self.comboBox_rebin_l.currentIndex()   
                
                data_h = self.comboBox_rebin_h.itemText(index_h)
                data_k = self.comboBox_rebin_k.itemText(index_k)
                data_l = self.comboBox_rebin_l.itemText(index_l)
                
                self.rebin_parameters_h()
                self.rebin_parameters_k()
                self.rebin_parameters_l()
                
                index_h = self.comboBox_rebin_h.findText(data_h, 
                                                    QtCore.Qt.MatchFixedString)
                if (index_h >= 0):
                     self.comboBox_rebin_h.setCurrentIndex(index_h)    
                     
                index_k = self.comboBox_rebin_k.findText(data_k, 
                                                    QtCore.Qt.MatchFixedString)
                if (index_k >= 0):
                     self.comboBox_rebin_k.setCurrentIndex(index_k)
                     
                index_l = self.comboBox_rebin_l.findText(data_l, 
                                                    QtCore.Qt.MatchFixedString)
                if (index_l >= 0):
                     self.comboBox_rebin_l.setCurrentIndex(index_l)  
                
                self.comboBox_rebin_h.blockSignals(False)
                self.comboBox_rebin_k.blockSignals(False)
                self.comboBox_rebin_l.blockSignals(False)
                
                try:
                    self.I_nxs = np.load(fname+'-intensity-nxs.npy')
                except:
                    self.I_nxs = None
        
                try:
                    self.sigma_sq_nxs = np.load(fname+'-error-nxs.npy')
                except:
                    self.sigma_sq_nxs = None
                    
                try:
                    self.I_exp = np.load(fname+'-intensity-exp.npy')
                except:
                    self.I_exp = None
        
                try:
                    self.sigma_sq_exp = np.load(fname+'-error-exp.npy')
                except:
                    self.sigma_sq_exp = None
                    
                try:
                    self.I_obs = np.load(fname+'-intensity-obs.npy')
                except:
                    self.I_obs = None
                    
                try:
                    self.I_recalc = np.load(fname+'-intensity-recalc.npy')
                    self.recalculated = True
                except:
                    self.I_recalc = None
                                        
                self.replot_intensity_exp()
                
                self.fname = fname
                
                self.batch = np.int(self.lineEdit_run.text())
                self.progress = np.int(self.progressBar_ref.value())
                
                if (self.batch > 0 or self.progress > 0):
                    
                    self.restart = True
                    
                    self.preprocess_supercell()
                    self.initialize_intensity()
                    self.filter_sigma()
                
                    if (self.checkBox_batch.isChecked()):
                        if (self.batch == np.int(self.lineEdit_runs.text())):
                            run = str(self.batch-1)
                        else:
                            run = str(self.batch)
                    else:
                        run = ''
                    
                    try:
                        self.load_magnetic(run)
                        self.initialize_magnetic()
                        self.magnetic = True
                    except:
                        self.magnetic = False
                    try:
                        self.load_occupational(run)
                        self.initialize_occupational()
                        self.occupational = True
                    except:
                        self.occupational = False
                    try:
                        self.load_displacive(run)
                        self.initialize_displacive()
                        self.displacive = True
                    except:
                        self.displacive = False
                    
                    n = self.nu*self.nv*self.nw*self.n_atm
                    
                    self.iteration = len(self.scale) // n
                    self.restart = False                
                    self.allocated = True
                    
                    self.replot_intensity_ref()
                    self.plot_chi_sq()
                    
                    self.replot_intensity_calc()
                 
            try: 
                self.recalculate_correlations_1d()
            except:
                pass
            
            try: 
                self.recalculate_correlations_3d()
            except:
                pass

            if (self.tableWidget_calc.rowCount() > 0):
                
                for i in range(3):
                    for j in range(5):
                        self.tableWidget_calc.item(i, 
                                                 j).setTextAlignment(alignment)
         
                self.tableWidget_calc.horizontalHeader().setSectionResizeMode(
                                                 QtWidgets.QHeaderView.Stretch)
                
                for i in range(3):
                     self.tableWidget_calc.item(i, 0).setFlags(
                      QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
    
                self.tableWidget_calc.itemChanged.connect(self.update_calc)
                                
                batch = self.checkBox_batch_calc.isChecked()
                
                runs = np.int(self.lineEdit_runs_calc.text())
                                
                nh = np.int(self.tableWidget_calc.item(0, 1).text())
                nk = np.int(self.tableWidget_calc.item(1, 1).text())
                nl = np.int(self.tableWidget_calc.item(2, 1).text())
                            
                index = self.comboBox_axes.currentIndex()
                
                min_h = np.float(self.tableWidget_calc.item(0, 2).text())
                min_k = np.float(self.tableWidget_calc.item(1, 2).text())
                min_l = np.float(self.tableWidget_calc.item(2, 2).text())
                
                max_h = np.float(self.tableWidget_calc.item(0, 3).text())
                max_k = np.float(self.tableWidget_calc.item(1, 3).text())
                max_l = np.float(self.tableWidget_calc.item(2, 3).text())
            
                laue_index = self.comboBox_laue.currentIndex()    
                                
                self.recalc_params = [nh,nk,nl,
                                      min_h,min_k,min_l,
                                      max_h,max_k,max_l,
                                      laue_index,index,batch,runs]
                
                self.changed_params = True
                
            self.fname = fname
                                 
    def close_application(self):
        
        choice = QtWidgets.QMessageBox.question(self, 
                                            'Quit?', 
                                            'Are you sure?',
                                            QtWidgets.QMessageBox.Yes |
                                            QtWidgets.QMessageBox.No,
                                            QtWidgets.QMessageBox.Yes)
                
        if (choice == QtWidgets.QMessageBox.Yes):
            sys.exit()
        else:
            pass

def run():
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle()
    window = Window()
    window.show()    
    sys.exit(app.exec_())