"""Reading the average structure from a CIF file."""

import re
from dataclasses import dataclass

import numpy as np


def number(text):
    """
    Value of a CIF number, dropping any standard uncertainty.

    Parameters
    ----------
    text : str
        CIF value such as ``"0.335(1)"``, ``"-.004"``, or ``"?"``.

    Returns
    -------
    value : float
        Parsed value, NaN for ``?`` or ``.``.

    """

    text = str(text).strip()
    if text in ("?", "."):
        return np.nan
    return float(re.sub(r"\(\d+\)$", "", text))


@dataclass
class AsymmetricSite:
    """
    Atom site of the asymmetric unit as listed in the CIF.

    Attributes
    ----------
    label : str
        Site label, e.g. ``"Fe1"``.
    symbol : str
        Type symbol, e.g. ``"Fe"`` or ``"Ba2+"``.
    xyz : ndarray, (3,)
        Fractional coordinates.
    occupancy : float
        Site occupancy.
    U : ndarray, (3, 3) or None
        Anisotropic displacement tensor in the CIF convention :math:`U^{ij}`
        (A^2, referred to the reciprocal-normalized basis), None if only an
        isotropic value is given.
    U_iso : float
        Isotropic displacement parameter (A^2).

    """

    label: str
    symbol: str
    xyz: np.ndarray
    occupancy: float
    U: np.ndarray = None
    U_iso: float = 0.0


@dataclass
class CifStructure:
    """
    Contents of a CIF needed to build the average structure.

    Attributes
    ----------
    cell : tuple of float
        a, b, c (A), alpha, beta, gamma (degrees).
    space_group : str
        Hermann-Mauguin symbol.
    number : int or None
        International Tables number.
    operators : list of str
        Symmetry operators in ``x,y,z`` form.
    sites : list of AsymmetricSite
        Asymmetric unit.

    """

    cell: tuple
    space_group: str
    number: int
    operators: list
    sites: list


def _first(block, *keys):
    for key in keys:
        if key in block:
            return block[key]
    return None


def read_cif(filename, block=None):
    """
    Read cell, symmetry, and atom sites from a CIF file.

    Parameters
    ----------
    filename : str or path
        CIF file.
    block : str, optional
        Data block name. The first block with atom sites is used if omitted.

    Returns
    -------
    structure : CifStructure
        Parsed average structure.

    """

    import CifFile

    cf = CifFile.ReadCif(str(filename))

    if block is None:
        names = [k for k in cf.keys() if "_atom_site_fract_x" in cf[k]]
        if not names:
            raise ValueError(f"no atom sites in {filename}")
        block = names[0]
    b = cf[block]

    cell = tuple(
        number(b[f"_cell_{key}"])
        for key in (
            "length_a",
            "length_b",
            "length_c",
            "angle_alpha",
            "angle_beta",
            "angle_gamma",
        )
    )

    symbol = _first(
        b,
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
        "_space_group_name_h-m_alt",
    )
    symbol = None if symbol is None else str(symbol).strip("'\" ")

    num = _first(b, "_space_group_IT_number", "_symmetry_Int_Tables_number")
    num = None if num is None or str(num) in ("?", ".") else int(num)

    operators = _first(
        b, "_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"
    )
    operators = [] if operators is None else [str(op) for op in operators]

    labels = list(b["_atom_site_label"])
    n = len(labels)

    def column(key, default):
        return list(b[key]) if key in b else [default] * n

    symbols = column("_atom_site_type_symbol", None)
    occupancies = column("_atom_site_occupancy", "1")
    U_iso = column("_atom_site_U_iso_or_equiv", "?")
    B_iso = column("_atom_site_B_iso_or_equiv", "?")

    aniso = {}
    if "_atom_site_aniso_label" in b:
        keys = ["11", "22", "33", "12", "13", "23"]
        for i, label in enumerate(b["_atom_site_aniso_label"]):
            if f"_atom_site_aniso_U_{keys[0]}" in b:
                U = [number(b[f"_atom_site_aniso_U_{k}"][i]) for k in keys]
            else:
                U = [
                    number(b[f"_atom_site_aniso_B_{k}"][i]) / (8 * np.pi**2)
                    for k in keys
                ]
            U11, U22, U33, U12, U13, U23 = U
            aniso[label] = np.array(
                [[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]]
            )

    sites = []
    for i, label in enumerate(labels):
        xyz = np.array(
            [number(b[f"_atom_site_fract_{c}"][i]) for c in "xyz"]
        )
        symbol_i = symbols[i] if symbols[i] is not None else label
        symbol_i = re.sub(r"[^A-Za-z0-9+-]", "", str(symbol_i))
        u = number(U_iso[i])
        if np.isnan(u):
            u = number(B_iso[i]) / (8 * np.pi**2)
        u = 0.0 if np.isnan(u) else u
        U = aniso.get(label)
        if U is not None and u == 0.0:
            u = np.trace(U) / 3
        occupancy = number(occupancies[i])
        sites.append(
            AsymmetricSite(
                label=str(label),
                symbol=symbol_i,
                xyz=xyz,
                occupancy=1.0 if np.isnan(occupancy) else occupancy,
                U=U,
                U_iso=u,
            )
        )

    return CifStructure(
        cell=cell,
        space_group=symbol,
        number=num,
        operators=operators,
        sites=sites,
    )
