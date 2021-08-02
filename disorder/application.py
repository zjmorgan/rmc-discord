#/usr/bin/env/python3

from PyQt5 import QtWidgets

import sys

from disorder.graphical.model import Model
from disorder.graphical.view import View
from disorder.graphical.presenter import Presenter

class Window:

  def __init__(self, view, adder):
      self.view = view
      self.adder = adder

  def show(self):
      self.view.show()
     
def run():
    app = QtWidgets.QApplication(sys.argv)
    view = View()
    adder = Presenter(Model(), view)
    window = Window(view, adder)
    window.show()    
    sys.exit(app.exec_())