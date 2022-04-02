#!/usr/bin/env python3

import re

import numpy as np

from disorder.material import tables

def length(atms, n_hkl):

    n_atm = len(atms)

    b = np.zeros(n_hkl*n_atm, dtype=complex)

    for i, atm in enumerate(atms):

        bc = tables.bc.get(atm)

        for i_hkl in range(n_hkl):

            b[i::n_atm] = bc

    return b

def form(ions, Q, source='x-ray'):

    n_hkl = Q.shape[0]
    n_atm = len(ions)

    factor = np.zeros(n_hkl*n_atm)

    s = Q/(4*np.pi)

    for i, ion in enumerate(ions):

        if (source == 'x-ray'):

            a1, b1, a2, b2, a3, b3, a4, b4, c = tables.X.get(ion)

            factor[i::n_atm] = a1*np.exp(-b1*s**2)\
                             + a2*np.exp(-b2*s**2)\
                             + a3*np.exp(-b3*s**2)\
                             + a4*np.exp(-b4*s**2)\
                             + c

        else:

            a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = tables.E.get(ion)

            factor[i::n_atm] = a1*np.exp(-b1*s**2)\
                             + a2*np.exp(-b2*s**2)\
                             + a3*np.exp(-b3*s**2)\
                             + a4*np.exp(-b4*s**2)\
                             + a5*np.exp(-b5*s**2)

    factor[factor < 0] = 0

    if (source == 'electron'):

        for i, ion in enumerate(ions):

            delta_Z = re.sub(r'[a-zA-Z]', '', ion)[::-1]
            delta_Z = int(delta_Z) if delta_Z != '' else 0
            factor[i::n_atm] += 0.023934*delta_Z/s**2

    return factor

def phase(Qx, Qy, Qz, rx, ry, rz):

    Q_dot_r = Qx[:,np.newaxis]*rx+Qy[:,np.newaxis]*ry+Qz[:,np.newaxis]*rz

    factor = np.exp(1j*Q_dot_r)

    return factor.flatten()