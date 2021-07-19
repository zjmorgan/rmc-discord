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

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_presenter(unittest.TestCase):      

    def setUp(self):
        self.view = View()
        self.presenter = Presenter(Model(), self.view)
            
    def test_supercell_n(self):
        
        self.view.get_nu = MagicMock(return_value=2)
        self.view.get_nv = MagicMock(return_value=3)
        self.view.get_nw = MagicMock(return_value=5)
        self.view.get_n_atm = MagicMock(return_value=7)   
        
        self.presenter.supercell_n()
        
        self.assertEqual(self.view.get_n(), 2*3*5*7)
        
    def test_change_type(self):

        self.view.create_atom_site_table(2)
        self.view.create_unit_cell_table(4)
                
        every_site = np.array(['1','2','2','2']).astype(int)
        
        self.view.get_every_site = MagicMock(return_value=every_site)

        atom_combo = np.array(['Au','Cu'])
        ion_combo = np.array(['Fe','Mn'])
        
        self.view.set_atom_combo(atom_combo)
        self.view.set_ion_combo(ion_combo)
                
        self.view.get_type = MagicMock(return_value='Neutron')

        self.presenter.change_type()
    
        every_atom = ['Fe','Mn','Mn','Mn']
        every_ion = ['-','-','-','-']
        np.testing.assert_array_equal(self.view.get_atom(), every_atom)
        np.testing.assert_array_equal(self.view.get_ion(), every_ion)
        
        self.view.set_atom_combo(atom_combo)
        self.view.set_ion_combo(ion_combo)
                
        self.view.get_type = MagicMock(return_value='X-ray')

        self.presenter.change_type()
        
        every_atom = ['Au','Cu','Cu','Cu']
        every_ion = ['-','-','-','-']
        np.testing.assert_array_equal(self.view.get_atom(), every_atom)
        np.testing.assert_array_equal(self.view.get_ion(), every_ion)
        
    def test_populate_atoms(self):
        
        self.view.create_atom_site_table(2)
        self.view.create_unit_cell_table(4)
                
        every_site = np.array(['1','2','2','2']).astype(int)
        
        self.view.get_every_site = MagicMock(return_value=every_site)

        atom_combo = np.array(['144Sm','Am'])
        
        self.view.set_atom_combo(atom_combo)
                
        self.presenter.populate_atoms()
        
        ions = ['Sm2+','Am2+']
        np.testing.assert_array_equal(self.view.get_ion_combo(), ions)
        
        every_atom = ['Sm','Am','Am','Am']
        every_ion = ['2+','2+','2+','2+']
        every_isotope = ['144','-','-','-']
        np.testing.assert_array_equal(self.view.get_atom(), every_atom)
        np.testing.assert_array_equal(self.view.get_ion(), every_ion)
        np.testing.assert_array_equal(self.view.get_isotope(), every_isotope)
        
    def test_populate_ions(self):
        
        self.view.create_atom_site_table(2)
        self.view.create_unit_cell_table(4)
                
        every_site = np.array(['1','2','2','2']).astype(int)
        
        self.view.get_every_site = MagicMock(return_value=every_site)

        ion_combo = np.array(['Sm2+','Am2+'])
        
        self.view.set_ion_combo(ion_combo)
                
        self.presenter.populate_ions()
        
        every_atom = ['Sm','Am','Am','Am']
        every_ion = ['2+','2+','2+','2+']
        np.testing.assert_array_equal(self.view.get_atom(), every_atom)
        np.testing.assert_array_equal(self.view.get_ion(), every_ion)
        
if __name__ == '__main__':
    unittest.main()