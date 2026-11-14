r"""Scattering prefactors of each kind at the sample points.

The structure factor of a configuration is

.. math::

    F(\mathbf{Q}) = \sum_i P_{k_i}(\mathbf{Q})
        \exp[i\mathbf{Q}\cdot(\mathbf{r}_i + \mathbf{u}_i)]

    \mathbf{F}_M(\mathbf{Q}) = \sum_i M_{k_i}(\mathbf{Q}) \mathbf{S}_i
        \exp[i\mathbf{Q}\cdot(\mathbf{r}_i + \mathbf{u}_i)]

with prefactors :math:`P_k` (nuclear or X-ray) and :math:`M_k = p \mu_k
f_k(Q)` (magnetic, :math:`p = \gamma r_0/2`) for kind :math:`k`, so both are
in fm. A rigid group is a kind whose prefactor is the phase sum over its
members, :math:`P_k = \sum_a b_a e^{i\mathbf{Q}\cdot\mathbf{d}_a}`, with
offsets :math:`\mathbf{d}_a` from the group center.

Prefactors are stored factored, :math:`P_k = c_k T_{t_k}(\mathbf{Q})`, with
a per-kind constant and a table shared by all kinds with the same
Q dependence; :math:`t_k = -1` means no Q dependence.
"""

from dataclasses import dataclass

import numpy as np

from rmc_discord.crystal import tables


@dataclass
class Factors:
    """
    Factored prefactors :math:`P_k(\\mathbf{Q}) = c_k T_{t_k}(\\mathbf{Q})`.

    Attributes
    ----------
    c : ndarray, (n_kinds,)
        Constant of each kind.
    t : ndarray, (n_kinds,), int
        Table of each kind, -1 for none.
    T : ndarray, (n_tables, n_points)
        Q-dependent tables (at least one row so the array is never empty).

    """

    c: np.ndarray
    t: np.ndarray
    T: np.ndarray

    def dense(self):
        """Prefactors of every kind at every point, (n_kinds, n_points)."""
        P = np.repeat(self.c[:, None], self.T.shape[1], axis=1)
        has = self.t >= 0
        P[has] *= self.T[self.t[has]]
        return P

    def select(self, indices):
        """Factors restricted to a subset of points."""
        return Factors(self.c, self.t, np.ascontiguousarray(self.T[:, indices]))


class _Tables:
    def __init__(self, n_points, dtype):
        self.rows, self.keys = [], {}
        self.n_points, self.dtype = n_points, dtype

    def index(self, key, make):
        if key not in self.keys:
            self.keys[key] = len(self.rows)
            self.rows.append(make())
        return self.keys[key]

    def array(self):
        if not self.rows:
            return np.ones((1, self.n_points), dtype=self.dtype)
        return np.ascontiguousarray(np.array(self.rows, dtype=self.dtype))


def _debye_waller(Q, U):
    return np.exp(-0.5 * np.einsum("ni,ij,nj->n", Q, U, Q))


def _group(kind, sample, form, dw):
    r"""Group phase sum :math:`\sum_a f_a(Q) e^{i\mathbf{Q}\cdot\mathbf{d}_a}`."""
    T = np.zeros(len(sample), dtype=complex)
    for m in kind.members:
        f = form(m) * np.exp(1j * sample.Q @ m.d)
        T += f * _debye_waller(sample.Q, m.U) if dw else f
    return T


def nuclear_factors(supercell, sample, radiation="neutron", debye_waller=()):
    """
    Factored nuclear (or X-ray) prefactors.

    Parameters
    ----------
    supercell : Supercell
        Kinds and sites.
    sample : QSample
        Sample points.
    radiation : {"neutron", "xray"}, optional
        Scattering length or atomic form factor.
    debye_waller : iterable of int, optional
        Active sites whose prefactor includes the Debye-Waller factor
        :math:`\\exp(-\\mathbf{Q}^T U \\mathbf{Q}/2)` of each occupant's
        ADP (each member's, for rigid groups). Use for sites whose
        displacements are not refined.

    Returns
    -------
    factors : Factors
        Complex constants and tables.

    """

    debye_waller = set(debye_waller)
    n_kinds = len(supercell.kinds)
    c = np.zeros(n_kinds, dtype=complex)
    t = np.full(n_kinds, -1, dtype=np.int64)
    T = _Tables(len(sample), complex)

    if radiation == "neutron":
        def form(m):
            return m.b
    elif radiation == "xray":
        def form(m):
            return tables.xray_form_factor(m.symbol, sample.Q_norm)
    else:
        raise ValueError(f"unknown radiation {radiation!r}")

    for k, kind in enumerate(supercell.kinds):
        if kind.vacancy:
            continue
        dw = kind.site in debye_waller
        U = kind.U.tobytes() if dw else None
        if kind.group:
            c[k] = 1.0
            t[k] = T.index(
                ("group", k, dw), lambda kind=kind, dw=dw: _group(kind, sample, form, dw)
            )
        elif radiation == "neutron":
            c[k] = kind.b
            if dw:
                t[k] = T.index(("dw", U), lambda kind=kind: _debye_waller(sample.Q, kind.U))
        else:
            c[k] = 1.0

            def make(kind=kind, dw=dw):
                f = form(kind)
                return f * _debye_waller(sample.Q, kind.U) if dw else f

            t[k] = T.index(("xray", kind.symbol, U), make)

    return Factors(c, t, T.array())


def magnetic_factors(supercell, sample, debye_waller=()):
    """
    Factored magnetic prefactors :math:`p \\mu_k f_k(Q)`.

    Parameters
    ----------
    supercell : Supercell
        Kinds with ions and moment sizes.
    sample : QSample
        Sample points.
    debye_waller : iterable of int, optional
        Active sites whose prefactor includes the Debye-Waller factor.

    Returns
    -------
    factors : Factors
        Real constants and tables.

    """

    debye_waller = set(debye_waller)
    n_kinds = len(supercell.kinds)
    c = np.zeros(n_kinds)
    t = np.full(n_kinds, -1, dtype=np.int64)
    T = _Tables(len(sample), float)

    for k, kind in enumerate(supercell.kinds):
        if not kind.magnetic:
            continue
        c[k] = tables.MAGNETIC_LENGTH * kind.moment
        dw = kind.site in debye_waller

        def make(kind=kind, dw=dw):
            f = tables.magnetic_form_factor(kind.ion, sample.Q_norm, kind.g)
            return f * _debye_waller(sample.Q, kind.U) if dw else f

        t[k] = T.index((kind.ion, kind.g, kind.U.tobytes() if dw else None), make)

    return Factors(c, t, T.array())


def nuclear_prefactor(supercell, sample, radiation="neutron", debye_waller=()):
    """Dense nuclear prefactors, (n_kinds, n_points) complex."""
    return nuclear_factors(supercell, sample, radiation, debye_waller).dense()


def magnetic_prefactor(supercell, sample, debye_waller=()):
    """Dense magnetic prefactors, (n_kinds, n_points)."""
    return magnetic_factors(supercell, sample, debye_waller).dense()


def paramagnetic_intensity(supercell, M):
    r"""
    Intensity of uncorrelated spins with the current kinds.

    .. math::

        I_\mathrm{para}(\mathbf{Q}) = \frac{2}{3n} \sum_i M_{k_i}(Q)^2

    Parameters
    ----------
    supercell : Supercell
        Current kinds.
    M : ndarray, (n_kinds, n_points)
        Dense magnetic prefactors.

    Returns
    -------
    I : ndarray, (n_points,)
        Intensity per atom (fm^2), normalized like the refined intensity.

    """

    counts = np.bincount(supercell.kind, minlength=len(supercell.kinds))
    return (2.0 / 3.0) * (counts @ M**2) / len(supercell)
