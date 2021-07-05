#!/usr/bin/env python3

import numpy as np

import os
directory = os.path.abspath(os.path.dirname(__file__))

def magnetic_form_factor_coefficients_j0():
    
    ion, A, a, B, b, C, c, D = np.loadtxt(directory+'/j0.csv',
                                          delimiter=',',
                                          dtype={'names': ('Ion', 
                                                           'A', 
                                                           'a', 
                                                           'B', 
                                                           'b', 
                                                           'C', 
                                                           'c', 
                                                           'D'),
                                                 'formats': ('U15', 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float)},           
                                          usecols=(0,1,2,3,4,5,6,7),
                                          skiprows=1, 
                                          unpack=True)
    
    vals = [A, a, B, b, C, c, D]
    
    return dict(zip(ion, zip(*vals)))

def magnetic_form_factor_coefficients_j2():
    
    ion, A, a, B, b, C, c, D = np.loadtxt(directory+'/j2.csv',
                                          delimiter=',',
                                          dtype={'names': ('Ion', 
                                                           'A', 
                                                           'a', 
                                                           'B', 
                                                           'b', 
                                                           'C', 
                                                           'c', 
                                                           'D'),
                                                 'formats': ('U15', 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float, 
                                                             float)},           
                                          usecols=(0,1,2,3,4,5,6,7),
                                          skiprows=1, 
                                          unpack=True)
    
    vals = [A, a, B, b, C, c, D]
    
    return dict(zip(ion, zip(*vals)))

def neutron_scattering_length_b():
    
    isotope, b = np.loadtxt(directory+'/b.csv',
                            delimiter=',',
                            dtype={'names': ('Isotope', 
                                             'b'),
                                   'formats': ('U15', 
                                               complex)},
                            usecols=(0,1),
                            skiprows=1, 
                            unpack=True)
                            
    return dict(zip(isotope, b))

def xray_form_factor_coefficients():
    
    ion, \
    a1, \
    b1, \
    a2, \
    b2, \
    a3, \
    b3, \
    a4, \
    b4, \
    c = np.loadtxt(directory+'/x.csv',
                   delimiter=',',
                   dtype={'names': ('Ion', 
                                    'a1', 
                                    'b1', 
                                    'a2,', 
                                    'b2', 
                                    'a3', 
                                    'b3', 
                                    'a4', 
                                    'b4', 
                                    'c'),
                          'formats': ('U15', 
                                      float, 
                                      float, 
                                      float, 
                                      float, 
                                      float, 
                                      float, 
                                      float, 
                                      float, 
                                      float)},           
                          usecols=(0,1,2,3,4,5,6,7,8,9),
                          skiprows=1, 
                          unpack=True)
                                                               
    vals = [a1, b1, a2, b2, a3, b3, a4, b4, c]
    
    return dict(zip(ion, zip(*vals)))

def electron_form_factor_coefficients():
    
    ion, \
    a1, \
    b1, \
    a2, \
    b2, \
    a3, \
    b3, \
    a4, \
    b4, \
    a5, \
    b5 = np.loadtxt(directory+'/e.csv',
                    delimiter=',',
                    dtype={'names': ('Ion', 
                                     'a1', 
                                     'b1', 
                                     'a2,', 
                                     'b2', 
                                     'a3', 
                                     'b3', 
                                     'a4', 
                                     'b4', 
                                     'a5', 
                                     'b5'),
                           'formats': ('U15', 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float, 
                                       float)},           
                           usecols=(0,1,2,3,4,5,6,7,8,9,10),
                           skiprows=1, 
                           unpack=True)
                                                               
    vals = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    
    return dict(zip(ion, zip(*vals)))

def atomic_numbers():
                        
    ion, z = np.loadtxt(directory+'/z.csv',
                        delimiter=',',
                        dtype={'names': ('Ion', 
                                         'Z'),
                               'formats': ('U15', 
                                           int)},           
                                usecols=(0,1),
                                skiprows=1, 
                                unpack=True)
                                                               
    vals = [z]
    
    return dict(zip(ion, zip(*vals)))

def space_groups():
                        
    sg_number, sg_name = np.loadtxt(directory+'/groups.csv',
                                    delimiter=',',
                                    dtype={'names': ('Number', 
                                                     'Name'),
                                           'formats': (int, 
                                                       'U15')},           
                                    usecols=(0,1),
                                    skiprows=0, 
                                    unpack=True)
                                                                                     
    sg_name = [sg.replace('\"', '').replace(' ', '') for sg in sg_name]
    
    return dict(zip(sg_name, sg_number))

j0 = magnetic_form_factor_coefficients_j0()
j2 = magnetic_form_factor_coefficients_j2()
bc = neutron_scattering_length_b()

X = xray_form_factor_coefficients()
E = electron_form_factor_coefficients()
Z = atomic_numbers()

sg = space_groups()