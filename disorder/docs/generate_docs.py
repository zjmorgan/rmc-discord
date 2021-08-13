#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

import numpy as np

import sys

from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)

from disorder.graphical.presenter import Presenter
from disorder.graphical.view import View
from disorder.graphical.model import Model

from disorder.graphical.utilities import save_screenshot

import os
directory = os.path.dirname(os.path.abspath(__file__))

class Window:

  def __init__(self, view, adder):
      self.view = view
      self.adder = adder

  def show(self):
      self.view.show()

class test_docs(unittest.TestCase):      

    def setUp(self):
        self.view = View()
        self.presenter = Presenter(Model(), self.view)
            
    def test_screenshot(self):
        adder = Presenter(Model(), self.view)
        window = Window(self.view, self.presenter)
        window.show()    
        save_screenshot(app, directory+'main')
        
if __name__ == '__main__':
    unittest.main()