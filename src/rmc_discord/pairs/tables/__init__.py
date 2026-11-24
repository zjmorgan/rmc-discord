import os

import numpy as np

directory = os.path.abspath(os.path.dirname(__file__))


def neutron_scattering_lengths():
    """
    Table of bound coherent neutron scattering lengths.

    Values are taken from the ``discord`` package parameter tables
    (BSD-3-Clause, Z. Morgan). Missing entries are ``nan``.

    Returns
    -------
    b : dict
        Scattering lengths in fm keyed by element or isotope symbol.
        Absorbing isotopes have complex values.

    """

    filename = os.path.join(directory, "scattering_lengths.csv")
    table = np.loadtxt(filename, delimiter=",", dtype=str, skiprows=1)
    return {key: complex(value) for key, value in table}


def scattering_length(species):
    """
    Bound coherent neutron scattering length of one species.

    Parameters
    ----------
    species : str
        Element or isotope symbol, e.g. ``"Tb"`` or ``"7Li"``.

    Returns
    -------
    b : complex
        Scattering length in fm.

    """

    table = neutron_scattering_lengths()
    if species not in table or np.isnan(table[species]):
        raise KeyError(f"No scattering length for {species!r}")
    return table[species]
