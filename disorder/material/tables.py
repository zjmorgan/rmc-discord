#!/usr/bin/env python3

import os

import numpy as np

directory = os.path.abspath(os.path.dirname(__file__))

def magnetic_form_factor_coefficients_j0():
    """
    Table of magnetic form factors :math:`j_0` coefficients.

    Returns
    -------
    j0 : dict
        Dictionary of magnetic form factors :math:`j_0` coefficients

    """
    
    filename = directory+'/j0.csv'
    names = ('Ion', 'A', 'a', 'B', 'b', 'C', 'c', 'D')
    formats = ('U15', float, float, float, float, float, float, float)
    columns = (0, 1, 2, 3, 4, 5, 6, 7)
    
    ion, A, a, B, b, C, c, D = np.loadtxt(filename,
                                          delimiter=',',
                                          dtype={'names': names,
                                                 'formats': formats},           
                                          usecols=columns,
                                          skiprows=1, 
                                          unpack=True)
    
    vals = [A, a, B, b, C, c, D]
    
    return dict(zip(ion, zip(*vals)))

def magnetic_form_factor_coefficients_j2():
    """
    Table of magnetic form factors :math:`j_2` coefficients.

    Returns
    -------
    j2 : dict
        Dictionary of magnetic form factors :math:`j_2` coefficients

    """

    filename = directory+'/j2.csv'
    names = ('Ion', 'A', 'a', 'B', 'b', 'C', 'c', 'D')
    formats = ('U15', float, float, float, float, float, float, float)
    columns = (0, 1, 2, 3, 4, 5, 6, 7)
    
    ion, A, a, B, b, C, c, D = np.loadtxt(filename,
                                          delimiter=',',
                                          dtype={'names': names,
                                                 'formats': formats},           
                                          usecols=columns,
                                          skiprows=1, 
                                          unpack=True)
    
    vals = [A, a, B, b, C, c, D]
    
    return dict(zip(ion, zip(*vals)))

def neutron_scattering_length_b():
    """
    Table of neutron scattering lengths.

    Returns
    -------
    b : dict
        Dictionary of neutron scattering lengths isotope keys

    """
    
    filename = directory+'/b.csv'
    names = ('Isotope', 'b')
    formats = ('U15', complex)
    columns = (0, 1)
    
    isotope, b = np.loadtxt(filename,
                            delimiter=',',
                            dtype={'names': names,'formats': formats},
                            usecols=columns,
                            skiprows=1, 
                            unpack=True)
                            
    return dict(zip(isotope, b))

def xray_form_factor_coefficients():
    
    filename = directory+'/x.csv'
    names = ('Ion', 'a1', 'b1', 'a2,', 'b2', 'a3', 'b3', 'a4', 'b4', 'c')
    formats = ('U15', float, float, float, float, 
               float, float, float, float, float)
    columns = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    
    data = np.loadtxt(filename,
                      delimiter=',',
                      dtype={'names': names, 'formats': formats},           
                      usecols=columns,
                      skiprows=1, 
                      unpack=True)
    
    ion, a1, b1, a2, b2, a3, b3, a4, b4, c = data
                                                               
    vals = [a1, b1, a2, b2, a3, b3, a4, b4, c]
    
    return dict(zip(ion, zip(*vals)))

def electron_form_factor_coefficients():
    
    filename = directory+'/e.csv'
    names = ('Ion', 'a1', 'b1', 'a2,', 'b2', 
             'a3', 'b3', 'a4', 'b4', 'a5', 'b5')
    formats = ('U15', float, float, float, float, float, 
               float, float, float, float, float) 
    columns = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    data = np.loadtxt(filename,
                      delimiter=',',
                      dtype={'names': names, 'formats': formats},           
                      usecols=columns,
                      skiprows=1, 
                      unpack=True)
        
    ion, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = data
                                                               
    vals = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    
    return dict(zip(ion, zip(*vals)))

def atomic_numbers():
     
    filename = directory+'/z.csv'
    names = ('Ion', 'Z')
    formats = ('U15', int)
    columns = (0, 1)
               
    ion, z = np.loadtxt(filename,
                        delimiter=',',
                        dtype={'names': names, 'formats': formats},           
                        usecols=columns,
                        skiprows=1, 
                        unpack=True)
                                                               
    vals = [z]
    
    return dict(zip(ion, zip(*vals)))

def space_groups():
    
    filename = directory+'/groups.csv'
    names = ('Number', 'Name')
    formats = (int, 'U15')  
    columns = (0, 1)
    
    sg_number, sg_name = np.loadtxt(filename,
                                    delimiter=',',
                                    dtype={'names': names, 'formats': formats},           
                                    usecols=columns,
                                    skiprows=0, 
                                    unpack=True)
                                                                                     
    sg_name = [sg.replace('\"', '').replace(' ', '') for sg in sg_name]
    
    return dict(zip(sg_name, sg_number))

def element_radii():
    
    filename = directory+'/radii.csv'
    names = ('Element', 'Atomic', 'Ionic', 'van der Waals')
    formats = ('U15', float, float, float)
    columns = (0, 1, 2, 3)
       
    element, atm, ion, vdw = np.loadtxt(filename,
                                        delimiter=',',
                                        dtype={'names': names, 
                                               'formats': formats},           
                                        usecols=columns,
                                        skiprows=1, 
                                        unpack=True)
                                                               
    vals = [atm, ion, vdw]
    
    return dict(zip(element, zip(*vals)))

def element_colors():
    
    filename = directory+'/colors.csv'
    names = ('Element', 'Red', 'Green', 'Blue')
    formats = ('U15', float, float, float)
    columns = (0, 1, 2, 3)
       
    element, r, g, b = np.loadtxt(filename,
                                  delimiter=',',
                                  dtype={'names': names, 'formats': formats},           
                                  usecols=columns,
                                  skiprows=1, 
                                  unpack=True)
                                                               
    vals = [r, g, b]
    
    return dict(zip(element, zip(*vals)))

j0 = magnetic_form_factor_coefficients_j0()
j2 = magnetic_form_factor_coefficients_j2()
bc = neutron_scattering_length_b()

X = xray_form_factor_coefficients()
E = electron_form_factor_coefficients()
Z = atomic_numbers()

sg = space_groups()
r = element_radii()
rgb = element_colors()