r"""
Effective moments of magnetic ions from their electron configuration.

* :math:`d^n` ions (3d, 4d, 5d): low- and high-spin states in an
  octahedral field (low spin listed first), spin only, :math:`\mu_\mathrm{eff} = g\sqrt{S(S+1)}`
  with :math:`g = 2` (orbital moments of d ions are largely quenched).
* :math:`f^n` ions (4f, 5f): the free-ion Hund's-rule ground multiplet,
  :math:`\mu_\mathrm{eff} = g_J\sqrt{J(J+1)}` with the Landé factor
  :math:`g_J`, which also enters the dipole form factor.

Only ions with tabulated magnetic form factors are offered.
"""

import re
from dataclasses import dataclass

import numpy as np

from rmc_discord.crystal.tables import _table

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf"
).split()
Z = {s: i + 1 for i, s in enumerate(SYMBOLS)}

# (first, last) atomic numbers of each open shell, and the core electrons before it
_SHELLS = (("3d", 21, 30, 18), ("4d", 39, 48, 36), ("5d", 72, 80, 68), ("4f", 57, 71, 54), ("5f", 89, 98, 86))


@dataclass(frozen=True)
class SpinState:
    """A spin state of an ion: quantum numbers, Landé factor and effective moment (Bohr magnetons)."""

    name: str
    S: float
    L: float
    J: float
    g: float
    moment: float

    @property
    def label(self):
        return f"{self.name} ({self.moment:.2f} μB)"


def parse_ion(ion):
    """Element and charge of an ion such as ``"Fe3+"`` (charge 0 for neutral atoms)."""
    m = re.fullmatch(r"([A-Z][a-z]?)(\d*)([+-]?)", ion.strip())
    if not m:
        raise ValueError(f"not an ion: {ion!r}")
    element, n, sign = m.groups()
    charge = int(n or (1 if sign else 0)) * (-1 if sign == "-" else 1)
    return element, charge


def magnetic_ions(element):
    """Ions of an element with tabulated magnetic form factors, cations by increasing charge."""
    ions = [k for k in _table("j0.csv") if parse_ion(k)[0] == element]
    return sorted(ions, key=lambda k: parse_ion(k)[1])


def shell(ion):
    """Open shell (``"3d"``, ``"4f"``, ...) and its electron count, or None."""
    element, charge = parse_ion(ion)
    z = Z.get(element)
    if z is None:
        return None
    for name, first, last, core in _SHELLS:
        if first <= z <= last:
            n = z - core - charge
            size = 10 if name.endswith("d") else 14
            return (name, n) if 0 <= n <= size else None
    return None


def _spin_only(name, unpaired):
    S = unpaired / 2
    return SpinState(name, S, 0.0, S, 2.0, 2.0 * np.sqrt(S * (S + 1)))


def _hund(n):
    """S, L, J of the f^n ground multiplet."""
    ml = [3, 2, 1, 0, -1, -2, -3]
    up, down = min(n, 7), max(n - 7, 0)
    S = (up - down) / 2
    L = abs(sum(ml[:up]) + sum(ml[:down]))
    J = abs(L - S) if n < 7 else L + S
    return S, L, J


def spin_states(ion):
    """
    Spin states of an ion, the default first: low spin (where it differs
    from high spin and carries a moment) for d ions, the Hund's-rule
    multiplet for f ions.

    Returns
    -------
    states : list of SpinState
        Empty for closed shells or ions outside the d and f series.
    """
    found = shell(ion)
    if found is None:
        return []
    name, n = found
    if name.endswith("d"):
        high = n if n <= 5 else 10 - n
        low = n if n <= 3 else (6 - n if n <= 6 else n - 6 if n <= 8 else 10 - n)
        states = [_spin_only(f"high spin, S = {_half(high)}", high)]
        if low != high:
            states.insert(0, _spin_only(f"low spin, S = {_half(low)}", low))  # low spin is the default
        return [s for s in states if s.S > 0]
    S, L, J = _hund(n)
    if J == 0:
        return []
    g = 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
    return [SpinState(f"free ion, J = {_half(2 * J)}", S, L, J, g, g * np.sqrt(J * (J + 1)))]


def _half(twice):
    twice = int(round(twice))
    return f"{twice // 2}" if twice % 2 == 0 else f"{twice}/2"
