"""Scattering lengths, form factors, and display properties of elements.

Values are read once from the CSV tables in ``crystal/data``:

- ``b.csv``: coherent neutron scattering lengths (fm), Sears (1992).
- ``j0.csv``, ``j2.csv``: magnetic form factor coefficients, International
  Tables C, 4.4.5.
- ``x.csv``: X-ray form factor coefficients, International Tables C, 6.1.1.
- ``radii.csv``, ``colors.csv``: atomic radii (A) and RGB colors.
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

#: Magnetic scattering amplitude per Bohr magneton, gamma r_0 / 2 (fm).
MAGNETIC_LENGTH = 2.69544


@lru_cache(maxsize=None)
def _table(name):
    with open(DATA / name, newline="") as f:
        rows = list(csv.reader(f))
    return {row[0]: row[1:] for row in rows[1:]}


def element(symbol):
    """
    Element symbol of an ion or isotope label.

    Parameters
    ----------
    symbol : str
        Label such as ``"Fe3+"``, ``"O2-"``, ``"Fe"``, or ``"57Fe"``.

    Returns
    -------
    element : str
        Element symbol, e.g. ``"Fe"``.

    """

    match = re.match(r"^\d*([A-Z][a-z]?)", symbol)
    if match is None:
        raise ValueError(f"cannot parse element from {symbol!r}")
    return match.group(1)


def scattering_length(isotope):
    """
    Coherent neutron scattering length.

    Parameters
    ----------
    isotope : str
        Element (natural abundance) or isotope, e.g. ``"Fe"``, ``"57Fe"``.
        Ion charges are ignored.

    Returns
    -------
    b : complex
        Scattering length (fm); the imaginary part is absorption.

    """

    table = _table("b.csv")
    key = re.match(r"^(\d*[A-Z][a-z]?)", isotope).group(1)
    if key not in table:
        key = element(isotope)
    return complex(table[key][0])


def _coefficients(name, ion):
    table = _table(name)
    if ion not in table:
        raise KeyError(f"{ion!r} not in {name}")
    return np.array([float(v) for v in table[ion]])


def magnetic_form_factor(ion, Q, g=2.0):
    r"""
    Magnetic form factor in the dipole approximation.

    .. math::

        f(Q) = \langle j_0 \rangle + (2/g - 1) \langle j_2 \rangle

    with :math:`\langle j_l \rangle` evaluated at :math:`s = Q/4\pi`.

    Parameters
    ----------
    ion : str
        Magnetic ion, e.g. ``"Fe3+"``.
    Q : array_like
        Momentum transfer magnitude (1/A).
    g : float, optional
        Lande g-factor. The default ``2`` gives :math:`\langle j_0 \rangle`.

    Returns
    -------
    f : ndarray
        Form factor, normalized to 1 at Q = 0.

    """

    s2 = (np.asarray(Q, dtype=float) / (4 * np.pi)) ** 2

    A, a, B, b, C, c, D = _coefficients("j0.csv", ion)
    j0 = A * np.exp(-a * s2) + B * np.exp(-b * s2) + C * np.exp(-c * s2) + D

    if g == 2.0:
        return j0

    A, a, B, b, C, c, D = _coefficients("j2.csv", ion)
    j2 = (
        A * np.exp(-a * s2) + B * np.exp(-b * s2) + C * np.exp(-c * s2) + D
    ) * s2

    return j0 + (2 / g - 1) * j2


def xray_form_factor(ion, Q):
    r"""
    X-ray atomic form factor.

    .. math::

        f(Q) = \sum_{i=1}^4 a_i \exp(-b_i s^2) + c, \quad s = Q/4\pi

    Parameters
    ----------
    ion : str
        Element or ion, e.g. ``"O2-"``.
    Q : array_like
        Momentum transfer magnitude (1/A).

    Returns
    -------
    f : ndarray
        Form factor (electrons).

    """

    s2 = (np.asarray(Q, dtype=float) / (4 * np.pi)) ** 2

    table = _table("x.csv")
    key = ion if ion in table else element(ion)
    a1, b1, a2, b2, a3, b3, a4, b4, c = _coefficients("x.csv", key)

    return (
        a1 * np.exp(-b1 * s2)
        + a2 * np.exp(-b2 * s2)
        + a3 * np.exp(-b3 * s2)
        + a4 * np.exp(-b4 * s2)
        + c
    )


def radius(symbol, kind="Empirical"):
    """
    Atomic radius for display.

    Parameters
    ----------
    symbol : str
        Element, ion, or isotope label.
    kind : str, optional
        Column of ``radii.csv``.

    Returns
    -------
    r : float
        Radius (A).

    """

    with open(DATA / "radii.csv", newline="") as f:
        header = next(csv.reader(f))
    column = header.index(kind) - 1
    return float(_table("radii.csv")[element(symbol)][column])


def color(symbol):
    """
    Display color of an element.

    Parameters
    ----------
    symbol : str
        Element, ion, or isotope label.

    Returns
    -------
    rgb : tuple of float
        Red, green, blue in [0, 1].

    """

    return tuple(float(v) for v in _table("colors.csv")[element(symbol)])
