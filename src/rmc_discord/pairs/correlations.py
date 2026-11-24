r"""Occupation covariances.

.. math::

    C_{ij}(\mathbf{R}) = \langle \delta n_i(\mathbf{0})\,
    \delta n_j(\mathbf{R}) \rangle, \qquad \delta n = n - \langle n \rangle.


Covariances may come from an analytic model or be estimated from
occupation configurations of a periodic lattice of cells. Only the
participating positions are represented; this is not an atomistic
supercell of the full structure.
"""

import numpy as np


def translations(max_offset):
    """
    Cell translations with ``|R_a| <= max_offset`` along each axis.

    Parameters
    ----------
    max_offset : int or sequence of int
        Cell-index cutoff per axis.

    Returns
    -------
    R : ndarray of int, (m, 3)

    """

    n = np.broadcast_to(max_offset, 3)
    ranges = [np.arange(-k, k + 1) for k in n]
    return np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)


def covariance_from_configurations(occupation, max_offset, mean=None):
    r"""
    Estimate occupation covariances from periodic configurations.

    Parameters
    ----------
    occupation : ndarray, (m, L1, L2, L3, n)
        Occupation (0 or 1) of each position in each cell for ``m``
        independent configurations.
    max_offset : int or sequence of int
        Cell-index cutoff; ``2 * max_offset + 1`` must not exceed the
        lattice size so that no translation is counted twice.
    mean : array_like, (n,), optional
        Average occupation. Estimated from the configurations by
        default.

    Returns
    -------
    R : ndarray of int, (k, 3)
        Cell translations.
    C : ndarray, (k, n, n)
        ``C[k, i, j]`` :math:`= \langle \delta n_i(\mathbf{0})\,
        \delta n_j(\mathbf{R}_k) \rangle` averaged over cells and
        configurations.

    """

    occupation = np.asarray(occupation, dtype=float)
    shape = np.array(occupation.shape[1:4])
    R = translations(max_offset)
    if np.any(2 * np.abs(R).max(axis=0) + 1 > shape):
        raise ValueError("max_offset too large for the lattice size")

    if mean is None:
        mean = occupation.mean(axis=(0, 1, 2, 3))
    delta = occupation - mean
    F = np.fft.fftn(delta, axes=(1, 2, 3))
    index = tuple((R % shape).T)

    n = occupation.shape[-1]
    C = np.empty((len(R), n, n))
    for i in range(n):
        corr = np.fft.ifftn(F[..., i, None].conj() * F, axes=(1, 2, 3))
        C[:, i, :] = corr.real.mean(axis=0)[index] / np.prod(shape)
    return R, C


def cluster_covariance(sites, cluster):
    r"""
    Covariance among the members of one cluster.

    In an exclusive cluster at most one member is occupied, so
    :math:`C_{ab} = x_a \delta_{ab} - x_a x_b`. In a co-occupied unit
    all members share one occupation variable with mean :math:`x`, so
    :math:`C_{ab} = x(1 - x)` for every pair.

    Parameters
    ----------
    sites : Sites
        Participating positions.
    cluster : Cluster

    Returns
    -------
    C : ndarray, (m, m)
        Covariance between members in the order of ``cluster.members``.

    """

    x = sites.occupancy[cluster.members]
    if cluster.kind == "co-occupied":
        xm = x.mean()
        return np.full((len(x), len(x)), xm * (1 - xm))
    return np.diag(x) - np.outer(x, x)


def independent_cluster_covariance(sites, clusters):
    r"""
    Covariance of independent clusters (no local order).

    Members of one cluster follow :func:`cluster_covariance`; different
    clusters are uncorrelated.

    Parameters
    ----------
    sites : Sites
        Participating positions.
    clusters : list of Cluster
        Split-site clusters covering all positions.

    Returns
    -------
    R : ndarray of int, (k, 3)
        Cell translations present.
    C : ndarray, (k, n, n)
        Occupation covariance.

    """

    entries = {}
    for cluster in clusters:
        cov = cluster_covariance(sites, cluster)
        for a, (i, oi) in enumerate(zip(cluster.members, cluster.offsets)):
            for c, (j, oj) in enumerate(zip(cluster.members, cluster.offsets)):
                entries.setdefault(tuple(oj - oi), []).append(
                    (i, j, cov[a, c])
                )

    R = np.array(sorted(entries))
    n = len(sites)
    C = np.zeros((len(R), n, n))
    for k, key in enumerate(map(tuple, R)):
        for i, j, value in entries[key]:
            C[k, i, j] += value
    return R, C


def fixed_cluster_orbits(orbits, sites, clusters):
    r"""
    Orbit coefficients fixed exactly by the average structure.

    Pairs inside one cluster (including self pairs) have the covariance
    of :func:`cluster_covariance`, independent of any local order.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions.
    clusters : list of Cluster
        Split-site clusters covering all positions.

    Returns
    -------
    index : ndarray of int
        Fixed orbit indices.
    value : ndarray
        Their covariance values.

    """

    owner = np.empty(len(sites), int)
    rel = np.empty((len(sites), 3), int)
    rank = np.empty(len(sites), int)
    blocks = []
    for k, cluster in enumerate(clusters):
        owner[cluster.members] = k
        rel[cluster.members] = cluster.offsets
        rank[cluster.members] = np.arange(len(cluster))
        blocks.append(cluster_covariance(sites, cluster))
    inside = (owner[orbits.i] == owner[orbits.j]) & np.all(
        orbits.R == rel[orbits.j] - rel[orbits.i], axis=1
    )
    value = np.zeros(len(orbits))
    for p in np.flatnonzero(inside):
        block = blocks[owner[orbits.i[p]]]
        value[p] = block[rank[orbits.i[p]], rank[orbits.j[p]]]

    index = np.unique(orbits.orbit[inside])
    mixed = np.isin(orbits.orbit, index) & ~inside
    if np.any(mixed):
        raise ValueError("Orbit mixes intra- and inter-cluster pairs")
    first = np.zeros(orbits.n_orbits, int)
    first[orbits.orbit[::-1]] = np.arange(len(orbits))[::-1]
    fixed = value[first[index]]
    spread = np.abs(
        value[inside] - fixed[np.searchsorted(index, orbits.orbit[inside])]
    )
    if np.any(spread > 1e-8):
        raise ValueError("Inconsistent fixed values within an orbit")
    return index, fixed


def realizability_bounds(orbits, sites, clusters):
    r"""
    Bounds on covariances of 0/1 occupation variables.

    For occupation variables with means :math:`x_i, x_j`,

    .. math::

        \max(0, x_i + x_j - 1) - x_i x_j \le C_{ij}
        \le \min(x_i, x_j) - x_i x_j .

    A refined coefficient outside these bounds cannot come from any
    arrangement of atoms with the given average occupancies, which
    usually means the model is compensating for something it lacks
    (wrong occupancies, background, or missing contributions).

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions.
    clusters : list of Cluster
        Clusters covering all positions; members of a co-occupied unit
        share its mean occupancy.

    Returns
    -------
    lower, upper : ndarray, (n_orbits,)
        Bounds of each orbit coefficient.

    """

    x = sites.occupancy.astype(float).copy()
    for cluster in clusters:
        if cluster.kind == "co-occupied":
            x[cluster.members] = 1 - cluster.vacancy
    reps = orbits.representatives()
    xi, xj = x[orbits.i[reps]], x[orbits.j[reps]]
    lower = np.maximum(0, xi + xj - 1) - xi * xj
    upper = np.minimum(xi, xj) - xi * xj
    return lower, upper
