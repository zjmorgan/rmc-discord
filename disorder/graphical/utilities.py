#!/ur/bin/env/python3

import re
import os
import sys
import traceback
import logging
import inspect

from PyQt5 import QtWidgets, QtGui, QtCore

from distutils.util import strtobool

from IPython.core.ultratb import ColorTB

_root = os.path.abspath(os.path.dirname(__file__))

class FractionalDelegate(QtWidgets.QItemDelegate):
    
    def createEditor(self, parent, option, index):
        lineEdit = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator(0, 1, 4, lineEdit)
        lineEdit.setValidator(validator)
        
        return lineEdit
    
class StandardDoubleDelegate(QtWidgets.QItemDelegate):
    
    def createEditor(self, parent, option, index):
        lineEdit = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator(-999999, 999999, 4, lineEdit)
        lineEdit.setValidator(validator)
        
        return lineEdit
    
class PositiveDoubleDelegate(QtWidgets.QItemDelegate):
    
    def createEditor(self, parent, option, index):
        lineEdit = QtWidgets.QLineEdit(parent)
        validator = QtGui.QDoubleValidator(0, 999999, 4, lineEdit)
        lineEdit.setValidator(validator)
        
        return lineEdit
    
class SizeIntDelegate(QtWidgets.QItemDelegate):
    
    def createEditor(self, parent, option, index):
        lineEdit = QtWidgets.QLineEdit(parent)
        validator = QtGui.QIntValidator(1, 999, lineEdit)
        lineEdit.setValidator(validator)
        
        return lineEdit
    
class WorkerSignals(QtCore.QObject):

    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(list)

class Worker(QtCore.QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        self.kwargs['callback'] = self.signals.progress   
        
        self.stop = False

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
            
    def abort(self):
        self.stop = True
        
    def proceed(self):
        return not self.stop
    
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
                        cell_obj = obj.cellWidget(row, col)
                        if isinstance(cell_obj, QtWidgets.QComboBox):
                            cell_name = cell_obj.objectName()
                            cell_index = cell_obj.currentIndex()
                            cell_text = cell_obj.itemText(cell_index)
                            settings.setValue(cell_name, cell_text)
                            stream.writeQString(cell_name)
                        if isinstance(cell_obj, QtWidgets.QCheckBox):
                            cell_name = cell_obj.objectName()
                            cell_state = cell_obj.isChecked()
                            settings.setValue(cell_name, cell_state)
                            stream.writeQString(cell_name)
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
            value = str(settings.value(name))
            obj.setText(value)
            
        if isinstance(obj, QtWidgets.QProgressBar):
            name = obj.objectName()
            value = int(settings.value(name))
            obj.setValue(value)
            
        if isinstance(obj, QtWidgets.QCheckBox):
            name = obj.objectName()
            value = bool(strtobool(str(settings.value(name))))
            obj.setChecked(value)
                
        if isinstance(obj, QtWidgets.QTabWidget):
            name = obj.objectName()          
            index = int(settings.value(name))
            obj.setCurrentIndex(index) 
 
        if isinstance(obj, QtWidgets.QTableWidget):
            name = obj.objectName()
            data = settings.value('{}/data'.format(name))
            if not data:
                continue
            stream = QtCore.QDataStream(data, QtCore.QIODevice.ReadOnly)
            rowCount = stream.readInt()
            columnCount = stream.readInt()
            obj.setRowCount(rowCount)
            obj.setColumnCount(columnCount)
            for row in range(rowCount):
                cellText = str(stream.readQString())
                if cellText:
                    item = QtWidgets.QTableWidgetItem(cellText)
                    obj.setVerticalHeaderItem(row, item)
                    
            for col in range(columnCount):
                cellText = str(stream.readQString())
                if cellText:
                    item = QtWidgets.QTableWidgetItem(cellText)
                    obj.setHorizontalHeaderItem(col, item)
                    
            for row in range(rowCount):
                for col in range(columnCount):
                    cellText = str(stream.readQString())
                    if ('comboBox' in cellText):
                        combo = QtWidgets.QComboBox()
                        combo.setObjectName(cellText)         
                        obj.setCellWidget(row, col, combo)
                        combo.addItem(str(settings.value(cellText)))
                    elif ('checkBox' in cellText):
                        check = QtWidgets.QCheckBox()
                        check.setObjectName(cellText)
                        obj.setCellWidget(row, col, check)  
                        value = bool(strtobool(str(settings.value(cellText))))
                        check.setChecked(value)
                    else:
                        item = QtWidgets.QTableWidgetItem(cellText)
                        obj.setItem(row, col, item)

def report_exception(*args):
    ansi = re.compile('\x1b' + r'\[([\dA-Fa-f;]*?)m')
    icon = os.path.join(_root, 'logo.png')
    if (len(args) == 3):
        error_type, error, trace = args[:3]
    elif (len(args) == 1):
        exc = args[0]
        error_type, error, trace = exc.__class__, exc, exc.__traceback__
    message = ''.join(traceback.format_exception_only(error_type, error))
    information = ColorTB(mode="Context").text(error_type, error, trace)
    logging.error('Exception in GUI event loop\n'+information+'\n')
    message_box = QtWidgets.QMessageBox()
    message_box.setWindowIcon(QtGui.QIcon(icon))
    message_box.setWindowTitle('rmc-discord')
    message_box.setText(message)
    message_box.setInformativeText(ansi.sub('', information))
    message_box.setIcon(QtWidgets.QMessageBox.Warning)
    layout = message_box.layout()
    layout.setColumnMinimumWidth(layout.columnCount()-1, 600)
    return message_box.exec_()