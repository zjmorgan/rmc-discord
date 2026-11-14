"""Disordered supercell: atoms on the average sites of an N1 x N2 x N3 box.

Only the disordered sites are represented. Atoms that are perfectly ordered
contribute only at integer hkl, where the diffuse intensity of a periodic
supercell vanishes, so they can be left out.

Each supercell atom has a *kind*: one occupant (or the vacancy) of one active
unit-cell site. Occupational disorder changes kinds, displacive disorder
changes displacements ``u``, and magnetic disorder changes unit spins ``S``.
Moment sizes are properties of kinds, not of atoms.

A rigid group (e.g. a carbonate ion) is a kind with several member atoms at
fixed offsets from its center. It is occupied or vacant as a whole and its
displacement is a translation of the whole group.
"""

from dataclasses import dataclass

import numpy as np

from rmc_discord.crystal import tables

VACANCY = "vacancy"


@dataclass
class Member:
    """
    Atom of a rigid group.

    Attributes
    ----------
    label : str
        CIF label.
    symbol : str
        Type symbol.
    b : complex
        Coherent neutron scattering length (fm).
    d : ndarray, (3,)
        Cartesian offset from the group center (A).
    U : ndarray, (3, 3)
        Cartesian displacement tensor (A^2).

    """

    label: str
    symbol: str
    b: complex
    d: np.ndarray
    U: np.ndarray


@dataclass
class Kind:
    """
    Possible occupant of an active site.

    Attributes
    ----------
    site : int
        Index into :attr:`Supercell.sites` (active sites).
    label : str
        CIF label, or ``"vacancy"``.
    symbol : str
        Type symbol (element, ion, or isotope).
    fraction : float
        Average fraction of the site occupied by this kind.
    b : complex
        Coherent neutron scattering length (fm).
    ion : str or None
        Magnetic ion for the form factor, None if nonmagnetic.
    moment : float
        Moment size (Bohr magnetons) used in the magnetic prefactor.
    g : float
        Lande g-factor for the form factor.
    U : ndarray, (3, 3)
        Cartesian displacement tensor (A^2) of the occupant; for a rigid
        group, the mean over its members (translation only).
    members : list of Member or None
        Atoms of a rigid group, None for a single atom.

    """

    site: int
    label: str
    symbol: str
    fraction: float
    b: complex
    ion: str = None
    moment: float = 0.0
    g: float = 2.0
    U: np.ndarray = None
    members: list = None

    @property
    def magnetic(self):
        return self.ion is not None and self.moment != 0

    @property
    def vacancy(self):
        return self.label == VACANCY

    @property
    def group(self):
        return self.members is not None


def _formula(symbols):
    counts = {}
    for symbol in symbols:
        element = "".join(c for c in symbol if c.isalpha())
        counts[element] = counts.get(element, 0) + 1
    return "".join(e + (str(n) if n > 1 else "") for e, n in counts.items())


class Supercell:
    """
    Supercell of the disordered sites of a crystal.

    Parameters
    ----------
    crystal : Crystal
        Average structure.
    N : sequence of int
        Number of unit cells along a, b, c.
    labels : iterable of str
        CIF labels of the sites to include. Co-located occupants of the same
        position are included together.
    ions : dict, optional
        Magnetic ion for each CIF label, e.g. ``{"Fe1": "Fe3+"}``.
    moments : dict, optional
        Moment size (Bohr magnetons) for each ion or label.
    g : dict, optional
        Lande g-factor for each ion or label (default 2).
    isotopes : dict, optional
        Isotope for the scattering length of each label, e.g.
        ``{"Fe1": "57Fe"}``.
    groups : dict, optional
        Rigid groups as ``{center label: [member labels]}``, e.g.
        ``{"C1": ["O7"]}``. Each member atom joins the nearest center of its
        group; the center label must be in ``labels``.
    cutoff : float, optional
        Largest center-member distance (A) of a rigid group.
    rng : numpy.random.Generator or int, optional
        Random source for the initial configuration.

    Attributes
    ----------
    sites : list of Site
        Active unit-cell sites.
    kinds : list of Kind
        Occupants of the active sites.
    site_kinds : list of ndarray
        Kind indices available on each active site.
    r : ndarray, (n, 3)
        Average Cartesian positions (A).
    site : ndarray, (n,), int
        Active-site index of each atom.
    cell : ndarray, (n, 3), int
        Unit-cell coordinates of each atom.
    kind : ndarray, (n,), int
        Current kind of each atom.
    u : ndarray, (n, 3)
        Displacements (A).
    S : ndarray, (n, 3)
        Unit spins.

    """

    def __init__(
        self,
        crystal,
        N,
        labels,
        ions=None,
        moments=None,
        g=None,
        isotopes=None,
        groups=None,
        cutoff=1.6,
        rng=None,
    ):
        self.crystal = crystal
        self.N = tuple(int(n) for n in N)
        self.rng = np.random.default_rng(rng)

        ions = ions or {}
        moments = moments or {}
        g = g or {}
        isotopes = isotopes or {}
        groups = groups or {}

        def length(label, symbol):
            return tables.scattering_length(isotopes.get(label, symbol))

        indices = crystal.site_indices(labels)
        if not indices:
            raise ValueError(f"no sites with labels {sorted(labels)}")
        self.site_indices = indices
        self.sites = [crystal.sites[i] for i in indices]

        for center, member_labels in groups.items():
            if center in ions:
                raise ValueError(f"rigid group {center} cannot be magnetic")
            if set(member_labels) & set(labels):
                raise ValueError(f"members of {center} are also active sites")
        group_members = _assign_members(crystal, groups, cutoff, length)

        self.kinds, self.site_kinds = [], []
        for j, (index, site) in enumerate(zip(indices, self.sites)):
            members = []
            for label, symbol, occ in zip(site.labels, site.symbols, site.occupancies):
                ion = ions.get(label)
                moment = moments.get(label, moments.get(ion, 0.0)) if ion else 0.0
                U = crystal.atom_U[_atom(crystal, index, label)]
                kind = Kind(
                    site=j,
                    label=label,
                    symbol=symbol,
                    fraction=float(occ),
                    b=length(label, symbol),
                    ion=ion,
                    moment=float(moment),
                    g=float(g.get(label, g.get(ion, 2.0))) if ion else 2.0,
                    U=U,
                )
                if label in groups:
                    group = [Member(label, symbol, kind.b, np.zeros(3), U)]
                    group += group_members[label, index]
                    kind.members = group
                    kind.symbol = _formula(m.symbol for m in group)
                    kind.b = sum(m.b for m in group)
                    kind.U = np.mean([m.U for m in group], axis=0)
                members.append(kind)
            if site.vacancy > 1e-6:
                members.append(
                    Kind(
                        site=j,
                        label=VACANCY,
                        symbol=VACANCY,
                        fraction=site.vacancy,
                        b=0j,
                        U=site.U,
                    )
                )
            total = sum(k.fraction for k in members)
            for k in members:
                k.fraction /= total
            self.site_kinds.append(
                np.arange(len(self.kinds), len(self.kinds) + len(members))
            )
            self.kinds += members

        n_cells = int(np.prod(self.N))
        n_sites = len(self.sites)

        cells = np.stack(
            np.meshgrid(*[np.arange(n) for n in self.N], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self.cell = np.repeat(cells, n_sites, axis=0)
        self.site = np.tile(np.arange(n_sites), n_cells)

        xyz = np.array([s.xyz for s in self.sites])
        self.r = crystal.cartesian(self.cell + xyz[self.site])

        self.kind = np.zeros(len(self.site), dtype=np.int64)
        self.u = np.zeros((len(self.site), 3))
        self.S = np.zeros((len(self.site), 3))

        self.randomize_occupancy()
        self.randomize_spins()

    def __len__(self):
        return len(self.site)

    @property
    def n_cells(self):
        return int(np.prod(self.N))

    @property
    def n_sites(self):
        """Number of active sites per unit cell."""
        return len(self.sites)

    def atoms_on_site(self, j):
        """Indices of the supercell atoms on active site ``j``."""
        return np.arange(j, len(self), self.n_sites)

    @property
    def magnetic_sites(self):
        """Active sites with at least one magnetic kind."""
        return [
            j
            for j, ks in enumerate(self.site_kinds)
            if any(self.kinds[k].magnetic for k in ks)
        ]

    @property
    def mixed_sites(self):
        """Active sites with more than one kind (occupational disorder)."""
        return [j for j, ks in enumerate(self.site_kinds) if len(ks) > 1]

    def randomize_occupancy(self):
        """Assign kinds at random with the exact average composition."""

        for j, ks in enumerate(self.site_kinds):
            atoms = self.atoms_on_site(j)
            fractions = np.array([self.kinds[k].fraction for k in ks])
            counts = np.floor(fractions * len(atoms)).astype(int)
            remainder = len(atoms) - counts.sum()
            order = np.argsort(-(fractions * len(atoms) - counts))
            counts[order[:remainder]] += 1
            values = np.repeat(ks, counts)
            self.kind[atoms] = self.rng.permutation(values)

    def randomize_spins(self):
        """Uniformly random unit spins."""

        v = self.rng.normal(size=(len(self), 3))
        self.S = v / np.linalg.norm(v, axis=1, keepdims=True)

    def randomize_displacements(self, scale=1.0):
        """
        Gaussian displacements with the site ADPs.

        Parameters
        ----------
        scale : float, optional
            Multiple of the ADP covariance.

        """

        L = self.cholesky_U()
        z = self.rng.normal(size=(len(self), 3))
        self.u = np.sqrt(scale) * np.einsum("nij,nj->ni", L[self.kind], z)

    def cholesky_U(self, floor=1e-8):
        """
        Lower Cholesky factors of the Cartesian ADPs of the kinds.

        Parameters
        ----------
        floor : float, optional
            Added to the diagonal (A^2) so zero ADPs stay factorizable.

        Returns
        -------
        L : ndarray, (n_kinds, 3, 3)
            ``U = L L^T`` for each kind.

        """

        return np.array([np.linalg.cholesky(k.U + floor * np.eye(3)) for k in self.kinds])

    def composition(self):
        """
        Current fraction of each kind on its site.

        Returns
        -------
        fractions : ndarray, (n_kinds,)
            Fraction of the site's atoms that have each kind.

        """

        counts = np.bincount(self.kind, minlength=len(self.kinds))
        return counts / self.n_cells

    def positions(self):
        """Current Cartesian positions ``r + u`` (A)."""
        return self.r + self.u

    def atoms(self):
        """
        Occupied atoms with rigid groups expanded into their members.

        Returns
        -------
        r : ndarray, (m, 3)
            Cartesian positions including displacements (A).
        labels : ndarray of str, (m,)
            CIF labels.
        b : ndarray, complex, (m,)
            Neutron scattering lengths (fm).
        owner : ndarray, (m,), int
            Supercell atom (or group) each atom belongs to.

        """

        r, labels, b, owner = [], [], [], []
        x = self.positions()
        for k, kind in enumerate(self.kinds):
            if kind.vacancy:
                continue
            atoms = np.flatnonzero(self.kind == k)
            members = kind.members or [Member(kind.label, kind.symbol, kind.b, np.zeros(3), kind.U)]
            for m in members:
                r.append(x[atoms] + m.d)
                labels.append(np.full(len(atoms), m.label, dtype=object))
                b.append(np.full(len(atoms), m.b, dtype=complex))
                owner.append(atoms)
        order = np.argsort(np.concatenate(owner), kind="stable")
        return (
            np.concatenate(r)[order],
            np.concatenate(labels)[order],
            np.concatenate(b)[order],
            np.concatenate(owner)[order],
        )


def _atom(crystal, site, label):
    """Unit-cell atom of a CIF label at a unit-cell site."""
    for a in np.flatnonzero(crystal.atom_site == site):
        if crystal.asymmetric[crystal.atom_index[a]].label == label:
            return a
    raise KeyError(f"no {label} at site {site}")


def _assign_members(crystal, groups, cutoff, length):
    """
    Member atoms of each rigid group center.

    Returns
    -------
    members : dict
        ``{(center label, unit-cell site): [Member, ...]}`` with Cartesian
        offsets from the nearest center (minimum image).

    """

    labels = np.array([crystal.asymmetric[k].label for k in crystal.atom_index])
    members = {}
    for center, member_labels in groups.items():
        centers = crystal.site_indices([center])
        if not centers:
            raise ValueError(f"no rigid-group center {center}")
        for i in centers:
            members[center, i] = []
        x_c = np.array([crystal.sites[i].xyz for i in centers])
        for a in np.flatnonzero(np.isin(labels, list(member_labels))):
            d = crystal.atom_xyz[a] - x_c
            d -= np.round(d)
            r = crystal.cartesian(d)
            dist = np.linalg.norm(r, axis=1)
            nearest = int(np.argmin(dist))
            if dist[nearest] > cutoff:
                raise ValueError(
                    f"{labels[a]} at {crystal.atom_xyz[a]} is {dist[nearest]:.2f} A"
                    f" from the nearest {center} (cutoff {cutoff} A)"
                )
            asym = crystal.asymmetric[crystal.atom_index[a]]
            members[center, centers[nearest]].append(
                Member(asym.label, asym.symbol, length(asym.label, asym.symbol),
                       r[nearest], crystal.atom_U[a])
            )
    return members
