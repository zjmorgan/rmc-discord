"""Average crystal structure built on Mantid's crystallography classes.

Conventions
-----------
- Miller indices ``h`` and fractional coordinates ``x`` are row vectors in
  arrays of shape ``(..., 3)``.
- ``B`` is Mantid's Busing-Levy matrix, so ``|B h| = 1/d``. Cartesian
  scattering vectors are ``Q = 2 pi B h`` (1/A).
- ``A = inv(B).T`` has the direct lattice vectors as columns, so Cartesian
  positions are ``r = A x`` (A) and ``Q . r = 2 pi h . x``.
"""

from dataclasses import dataclass, replace

import numpy as np

from rmc_discord.crystal.cif import read_cif


def _space_group(symbol, number):
    from mantid.geometry import SpaceGroupFactory

    candidates = []
    if symbol:
        candidates += [symbol, " ".join(symbol.split())]
    if number:
        candidates += list(
            SpaceGroupFactory.subscribedSpaceGroupSymbols(int(number))
        )
    for candidate in candidates:
        if SpaceGroupFactory.isSubscribedSymbol(candidate):
            return SpaceGroupFactory.createSpaceGroup(candidate)
    raise ValueError(f"unknown space group {symbol!r} ({number})")


def _operator(op):
    """Affine parts (W, w) of a Mantid symmetry operation."""

    origin = np.array(op.transformCoordinates([0, 0, 0]))
    W = np.column_stack(
        [np.array(op.transformCoordinates(e)) - origin for e in np.eye(3)]
    )
    return np.round(W).astype(int), origin


@dataclass
class Site:
    """
    Position in the unit cell with one or more possible occupants.

    Attributes
    ----------
    xyz : ndarray, (3,)
        Fractional coordinates in [0, 1).
    labels : list of str
        CIF labels of the occupants.
    symbols : list of str
        Type symbols of the occupants.
    occupancies : ndarray
        Occupancy of each occupant.
    U : ndarray, (3, 3)
        Cartesian displacement tensor (A^2), occupancy-weighted over
        occupants.
    orbit : int
        Index of the asymmetric-unit position the site belongs to.

    """

    xyz: np.ndarray
    labels: list
    symbols: list
    occupancies: np.ndarray
    U: np.ndarray
    orbit: int

    @property
    def vacancy(self):
        """Vacant fraction of the site."""
        return max(0.0, 1.0 - float(np.sum(self.occupancies)))


class Crystal:
    """
    Average structure: lattice, space group, and the full unit cell.

    Parameters
    ----------
    cell : sequence of float
        a, b, c (A), alpha, beta, gamma (degrees).
    space_group : str
        Hermann-Mauguin symbol understood by Mantid.
    asymmetric : list of AsymmetricSite
        Atom sites of the asymmetric unit.
    operators : list of str, optional
        Symmetry operators defining the setting. Mantid's operators for the
        space group are used if omitted.
    number : int, optional
        Space group number, used if the symbol is not recognized.
    tol : float, optional
        Fractional tolerance for merging equivalent positions.

    """

    def __init__(
        self,
        cell,
        space_group,
        asymmetric,
        operators=None,
        number=None,
        tol=1e-4,
    ):
        from mantid.geometry import SymmetryOperationFactory, UnitCell

        self.cell = tuple(float(v) for v in cell)
        self.unit_cell = UnitCell(*self.cell)
        self.B = np.array(self.unit_cell.getB())
        self.A = np.linalg.inv(self.B).T

        self.space_group = _space_group(space_group, number)
        self.point_group = self.space_group.getPointGroup()
        self.laue = self.point_group.getLauePointGroupSymbol()

        if operators:
            ops = [SymmetryOperationFactory.createSymOp(op) for op in operators]
        else:
            ops = list(self.space_group.getSymmetryOperations())
        self.operators = [_operator(op) for op in ops]

        self.asymmetric = list(asymmetric)
        self.tol = tol

        self._expand()

    @classmethod
    def from_cif(cls, filename, block=None):
        """
        Average structure from a CIF file.

        Parameters
        ----------
        filename : str or path
            CIF file.
        block : str, optional
            Data block name.

        Returns
        -------
        crystal : Crystal
            Average structure.

        """

        cif = read_cif(filename, block)
        return cls(
            cif.cell,
            cif.space_group,
            cif.sites,
            operators=cif.operators,
            number=cif.number,
        )

    def with_occupancies(self, occupancies):
        """
        Same structure with some occupancies changed.

        Parameters
        ----------
        occupancies : dict
            New occupancy of each CIF label.

        Returns
        -------
        crystal : Crystal
            New average structure.

        """

        unknown = set(occupancies) - {s.label for s in self.asymmetric}
        if unknown:
            raise KeyError(f"unknown labels {sorted(unknown)}")
        asymmetric = [
            replace(s, occupancy=float(occupancies[s.label])) if s.label in occupancies else s
            for s in self.asymmetric
        ]
        crystal = Crystal.__new__(Crystal)
        crystal.__dict__.update(self.__dict__)
        crystal.asymmetric = asymmetric
        crystal._expand()
        return crystal

    def _U_cartesian(self, site):
        if site.U is None:
            return site.U_iso * np.eye(3)
        N = np.diag(np.linalg.norm(self.B, axis=0))
        return self.A @ N @ site.U @ N @ self.A.T

    def _expand(self):
        A_inv = np.linalg.inv(self.A)

        xyz, U, index = [], [], []
        for k, site in enumerate(self.asymmetric):
            U0 = self._U_cartesian(site)
            found = []
            for W, w in self.operators:
                x = np.mod(W @ site.xyz + w, 1.0)
                x[np.isclose(x, 1.0, atol=self.tol)] = 0.0
                d = np.array(found) - x if found else np.empty((0, 3))
                d -= np.round(d)
                if found and np.any(np.all(np.abs(d) < self.tol, axis=1)):
                    continue
                found.append(x)
                R = self.A @ W @ A_inv
                xyz.append(x)
                U.append(R @ U0 @ R.T)
                index.append(k)

        self.atom_xyz = np.array(xyz)
        self.atom_U = np.array(U)
        self.atom_index = np.array(index)

        sites = []
        assigned = np.full(len(xyz), -1)
        for i in range(len(xyz)):
            if assigned[i] >= 0:
                continue
            d = self.atom_xyz - self.atom_xyz[i]
            d -= np.round(d)
            members = np.flatnonzero(np.all(np.abs(d) < self.tol, axis=1))
            members = members[assigned[members] < 0]
            assigned[members] = len(sites)
            occ = np.array([self.asymmetric[index[m]].occupancy for m in members])
            weights = occ / occ.sum() if occ.sum() > 0 else occ
            sites.append(
                Site(
                    xyz=self.atom_xyz[i].copy(),
                    labels=[self.asymmetric[index[m]].label for m in members],
                    symbols=[self.asymmetric[index[m]].symbol for m in members],
                    occupancies=occ,
                    U=np.einsum("m,mij->ij", weights, self.atom_U[members]),
                    orbit=int(min(index[m] for m in members)),
                )
            )
        self.sites = sites
        self.atom_site = assigned

    @property
    def n_atoms(self):
        """Number of atom positions (per occupant) in the unit cell."""
        return len(self.atom_xyz)

    def site_indices(self, labels):
        """
        Unit-cell sites occupied by any of the given CIF labels.

        Parameters
        ----------
        labels : iterable of str
            CIF site labels.

        Returns
        -------
        indices : list of int
            Indices into :attr:`sites`.

        """

        labels = set(labels)
        return [i for i, s in enumerate(self.sites) if labels & set(s.labels)]

    def Q(self, hkl):
        """
        Cartesian scattering vectors.

        Parameters
        ----------
        hkl : array_like, (..., 3)
            Miller indices.

        Returns
        -------
        Q : ndarray, (..., 3)
            Scattering vectors (1/A).

        """

        return 2 * np.pi * np.asarray(hkl, dtype=float) @ self.B.T

    def cartesian(self, xyz):
        """
        Cartesian positions of fractional coordinates.

        Parameters
        ----------
        xyz : array_like, (..., 3)
            Fractional coordinates.

        Returns
        -------
        r : ndarray, (..., 3)
            Positions (A).

        """

        return np.asarray(xyz, dtype=float) @ self.A.T

    def is_allowed(self, hkl):
        """
        Reflection conditions of the space group.

        Parameters
        ----------
        hkl : array_like, (n, 3), int
            Miller indices.

        Returns
        -------
        allowed : ndarray of bool, (n,)
            True where the reflection is not systematically absent.

        """

        hkl = np.atleast_2d(np.asarray(hkl, dtype=int))
        unique, inverse = np.unique(hkl, axis=0, return_inverse=True)
        allowed = np.array(
            [self.space_group.isAllowedReflection([int(v) for v in h]) for h in unique]
        )
        return allowed[inverse.ravel()]
