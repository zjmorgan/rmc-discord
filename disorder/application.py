#/usr/bin/env/python3

from PyQt5 import QtWidgets

import sys

from disorder.graphical.model import Model
from disorder.graphical.view import View
from disorder.graphical.presenter import Presenter
from disorder.graphical.utilities import report_exception

from disorder.diffuse.scattering import parallelism

class Window:

  def __init__(self, view, adder):
      self.view = view
      self.adder = adder

  def show(self):
      self.view.show()
     
def run(): 
    app = QtWidgets.QApplication(sys.argv)
    parallelism()
    view = View()
    adder = Presenter(Model(), view)
    window = Window(view, adder)
    window.show()    
    sys.excepthook = report_exception
    sys.exit(app.exec_())