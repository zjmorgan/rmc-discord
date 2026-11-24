"""Local occupation rules derived from the average structure geometry.

Positions closer than a grouping distance belong to one cluster, of one
of two kinds decided by their occupancies:

* exclusive (split-site) clusters, occupancies summing to at most one:
  alternatives of which at most one is occupied (an atom and its split
  partners, or two species sharing a site);
* co-occupied units, equal occupancies summing to more than one: atoms
  present or absent together with one shared occupation variable (a
  partially occupied molecule such as a carbonate group).

Forbidden contacts are pairs of positions in different clusters closer
than a physically allowed distance, which cannot both be occupied.
"""

import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class Contacts:
    r"""
    Directed contacts ``i -> j`` at cell translation ``R``.

    Attributes
    ----------
    i, j : ndarray of int, (m,)
        Position indices.
    R : ndarray of int, (m, 3)
        Cell translation of ``j`` relative to ``i``.
    distance : ndarray, (m,)
        Separation :math:`|A (\mathbf{R} + \mathbf{x}_j - \mathbf{x}_i)|`
        in :math:`\mathring{\mathrm{A}}`.

    """

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.i)

    def subset(self, mask):
        return Contacts(
            self.i[mask], self.j[mask], self.R[mask], self.distance[mask]
        )

    def summary(self, sites, decimals=3):
        """
        Count contacts by label pair and distance, per position.

        Parameters
        ----------
        sites : Sites
            Positions the indices refer to.
        decimals : int, optional
            Rounding of distances used for grouping.

        Returns
        -------
        table : list of tuple
            ``(label_i, label_j, distance, contacts per position i)``.

        """

        per_label = collections.Counter(sites.label)
        keys = zip(
            sites.label[self.i],
            sites.label[self.j],
            np.round(self.distance, decimals),
        )
        counts = collections.Counter(keys)
        return sorted(
            (li, lj, d, n / per_label[li]) for (li, lj, d), n in counts.items()
        )


def find_contacts(structure, sites, max_distance):
    """
    All directed contacts shorter than a distance.

    Parameters
    ----------
    structure : Structure
        Supplies the lattice.
    sites : Sites
        Positions to consider.
    max_distance : float
        Largest separation in Angstrom.

    Returns
    -------
    contacts : Contacts
        Only the self pair (i = j, R = 0) is excluded, so coincident
        positions of different labels (substitution) are contacts.

    """

    n = np.ceil(max_distance * np.linalg.norm(structure.B, axis=0)) + 1
    ranges = [np.arange(-k, k + 1) for k in n.astype(int)]
    R = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
    dx = sites.x[None, :, None] - sites.x[:, None, None] + R
    d = structure.distance(dx)
    self_pair = (
        np.all(R == 0, axis=1)[None, None]
        & np.eye(len(sites), dtype=bool)[..., None]
    )
    i, j, k = np.nonzero((d < max_distance) & ~self_pair)
    return Contacts(i, j, R[k], d[i, j, k])


@dataclass
class Cluster:
    """
    Group of nearby positions with a common occupation rule.

    Attributes
    ----------
    members : ndarray of int, (m,)
        Position indices.
    offsets : ndarray of int, (m, 3)
        Cell offset of each member relative to the first.
    vacancy : float
        Probability that no member is occupied.
    kind : str
        ``"exclusive"`` (at most one member occupied) or
        ``"co-occupied"`` (all members occupied together).

    """

    members: np.ndarray
    offsets: np.ndarray
    vacancy: float
    kind: str = "exclusive"

    def __len__(self):
        return len(self.members)


def find_clusters(
    structure,
    sites,
    cluster_distance,
    tol=1e-3,
    occupancy_tol=0.02,
    min_bond=0.9,
):
    """
    Group nearby positions into exclusive clusters or co-occupied units.

    Parameters
    ----------
    structure : Structure
        Supplies the lattice.
    sites : Sites
        Participating positions.
    cluster_distance : float
        Positions closer than this (Angstrom) are grouped.
    tol : float, optional
        Tolerance on the occupancy sum of an exclusive cluster.
    occupancy_tol : float, optional
        Largest occupancy difference within a co-occupied unit.
    min_bond : float, optional
        Shortest distance (Angstrom) at which two atoms of a co-occupied
        unit can both be present.

    Returns
    -------
    clusters : list of Cluster
        Every position belongs to exactly one cluster (possibly alone).

    Raises
    ------
    ValueError
        If an exclusive group is not a clique of alternatives, or a
        group with occupancies summing to more than one has unequal
        occupancies (neither alternatives nor one rigid unit).

    """

    contacts = find_contacts(structure, sites, cluster_distance)
    neighbors = [[] for _ in range(len(sites))]
    for i, j, R in zip(contacts.i, contacts.j, contacts.R):
        neighbors[i].append((j, R))

    offset = [None] * len(sites)
    clusters = []
    for root in range(len(sites)):
        if offset[root] is not None:
            continue
        offset[root] = np.zeros(3, int)
        members, queue = [root], collections.deque([root])
        while queue:
            i = queue.popleft()
            for j, R in neighbors[i]:
                if offset[j] is None:
                    offset[j] = offset[i] + R
                    members.append(j)
                    queue.append(j)
                elif not np.array_equal(offset[j], offset[i] + R):
                    raise ValueError("Cluster percolates through the cell")

        members = np.array(members)
        offsets = np.array([offset[m] for m in members])
        dx = (sites.x[members] + offsets)[None] - (sites.x[members] + offsets)[
            :, None
        ]
        d = structure.distance(dx)
        x = sites.occupancy[members]
        if x.sum() <= 1 + tol:
            off_diagonal = ~np.eye(len(members), dtype=bool)
            if np.any(d[off_diagonal] >= cluster_distance):
                raise ValueError(f"Cluster {members} is not a clique")
            vacancy = max(0.0, 1 - x.sum())
            clusters.append(Cluster(members, offsets, vacancy))
        elif np.ptp(x) <= occupancy_tol:
            off_diagonal = ~np.eye(len(members), dtype=bool)
            if np.any(d[off_diagonal] < min_bond):
                raise ValueError(
                    f"Group {members} sums to occupancy {x.sum():.4f} > 1"
                    f" but has atoms closer than {min_bond} A"
                )
            vacancy = 1 - x.mean()
            clusters.append(Cluster(members, offsets, vacancy, "co-occupied"))
        else:
            raise ValueError(
                f"Group {members} has occupancies {np.round(x, 4)}: neither"
                " alternatives (sum <= 1) nor one co-occupied unit"
            )
    return clusters


def forbidden_contacts(structure, sites, clusters, min_distance):
    """
    Contacts between different clusters that are too short to coexist.

    Parameters
    ----------
    structure : Structure
        Supplies the lattice.
    sites : Sites
        Participating positions.
    clusters : list of Cluster
        Split-site clusters from :func:`find_clusters`.
    min_distance : float or dict
        Smallest allowed separation in Angstrom, either one value or a
        mapping from sorted species pairs, e.g. ``{("Sn", "Sn"): 2.6}``.

    Returns
    -------
    contacts : Contacts
        Directed pairs that cannot both be occupied.

    """

    if isinstance(min_distance, dict):
        limit = max(min_distance.values())
    else:
        limit = min_distance
    contacts = find_contacts(structure, sites, limit)

    owner = np.empty(len(sites), int)
    rel = np.empty((len(sites), 3), int)
    for k, cluster in enumerate(clusters):
        owner[cluster.members] = k
        rel[cluster.members] = cluster.offsets
    same = (owner[contacts.i] == owner[contacts.j]) & np.all(
        contacts.R == rel[contacts.j] - rel[contacts.i], axis=1
    )

    if isinstance(min_distance, dict):
        pairs = zip(sites.species[contacts.i], sites.species[contacts.j])
        cut = np.array(
            [min_distance.get(tuple(sorted(p)), 0.0) for p in pairs]
        )
    else:
        cut = np.full(len(contacts), min_distance)
    return contacts.subset(~same & (contacts.distance < cut))
