r"""Symmetry-independent pair correlations.

A pair :math:`(i, j, \mathbf{R})` joins position :math:`i` in the
origin cell to position :math:`j` in cell :math:`\mathbf{R}`. A
space-group operator :math:`(W, \mathbf{t})` maps it to
:math:`(i', j', \mathbf{R}')` with

.. math::

    W \mathbf{x}_i + \mathbf{t} = \mathbf{x}_{i'} + \mathbf{L}_i, \quad
    W \mathbf{x}_j + \mathbf{t} = \mathbf{x}_{j'} + \mathbf{L}_j, \quad
    \mathbf{R}' = W \mathbf{R} + \mathbf{L}_j - \mathbf{L}_i;

pair reversal maps it to :math:`(j, i, -\mathbf{R})`. Correlations with
the full symmetry of the average structure are equal on each orbit, so
one coefficient per orbit
describes them. Pairs are selected with a distance cutoff, which is
invariant under the group.
"""

from dataclasses import dataclass

import numpy as np


def position_permutations(structure, sites, tol=1e-4):
    r"""
    Action of each operator on the positions of one cell.

    Parameters
    ----------
    structure : Structure
        Supplies the symmetry operators.
    sites : Sites
        Positions closed under the operators.
    tol : float, optional
        Fractional matching tolerance.

    Returns
    -------
    perm : ndarray of int, (g, n)
        ``perm[g, i]`` is the image position of ``i``.
    shift : ndarray of int, (g, n, 3)
        Lattice vector with :math:`W \mathbf{x}_i + \mathbf{t} =
        \mathbf{x}_{\mathrm{perm}} + \mathbf{L}`.

    """

    n = len(sites)
    perm = np.empty((len(structure.operators), n), int)
    shift = np.empty((len(structure.operators), n, 3), int)
    for g, (W, t) in enumerate(structure.operators):
        image = sites.x @ W.T + t
        diff = image[:, None] - sites.x[None]
        frac = (diff + 0.5) % 1 - 0.5
        match = np.all(np.abs(frac) < tol, axis=2)
        match &= sites.label[None] == sites.label[:, None]
        if not np.all(match.sum(axis=1) == 1):
            raise ValueError("Positions are not closed under the operators")
        perm[g] = np.argmax(match, axis=1)
        shift[g] = np.rint(image - sites.x[perm[g]]).astype(int)
    return perm, shift


@dataclass
class PairOrbits:
    r"""
    Pairs within a cutoff, grouped into symmetry orbits.

    Attributes
    ----------
    i, j : ndarray of int, (m,)
        Position indices.
    R : ndarray of int, (m, 3)
        Cell translation of ``j``.
    d : ndarray, (m, 3)
        Pair vectors :math:`\mathbf{R} + \mathbf{x}_j - \mathbf{x}_i`
        (fractional).
    orbit : ndarray of int, (m,)
        Orbit index of each pair, ``0 .. n_orbits - 1``, ordered by
        pair distance.
    distance : ndarray, (n_orbits,)
        Pair distance of each orbit in Angstrom.

    """

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    d: np.ndarray
    orbit: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.i)

    @property
    def n_orbits(self):
        return len(self.distance)

    def representatives(self):
        """Index of the first pair of each orbit."""

        _, first = np.unique(self.orbit, return_index=True)
        return first


def pair_orbits(structure, sites, max_distance, cylinder=None, tol=1e-6):
    r"""
    All pairs up to a distance, grouped into symmetry orbits.

    Parameters
    ----------
    structure : Structure
        Supplies lattice and operators.
    sites : Sites
        Participating positions, closed under the operators.
    max_distance : float
        Pair distance cutoff in Angstrom (self pairs included).
    cylinder : tuple, optional
        ``(axis, radius, half_length)`` adding pairs whose vector lies
        within ``radius`` (Angstrom) of the direction ``axis``
        (fractional, e.g. ``(0, 0, 1)``) and within ``half_length`` along
        it. Use it for correlations that are long-ranged along chains or
        columns but short across them; the cylinder must be invariant
        under the point group (e.g. along a unique axis).
    tol : float, optional
        Distance tolerance at the cutoff.

    Returns
    -------
    orbits : PairOrbits

    Raises
    ------
    ValueError
        If the selected pair set is not closed under the symmetry, e.g.
        a cylinder along a direction the point group does not preserve.

    """

    reach = (
        max_distance
        if cylinder is None
        else max(max_distance, cylinder[1], cylinder[2])
    )
    n = np.ceil(reach * np.linalg.norm(structure.B, axis=0)) + 1
    ranges = [np.arange(-k, k + 1) for k in n.astype(int)]
    T = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
    dx = sites.x[None, :, None] - sites.x[:, None, None] + T
    inside = structure.distance(dx) <= max_distance + tol
    if cylinder is not None:
        axis, radius, half_length = cylinder
        u = structure.A @ np.asarray(axis, dtype=float)
        u = u / np.linalg.norm(u)
        r = dx @ structure.A.T
        along = r @ u
        across = np.linalg.norm(r - along[..., None] * u, axis=-1)
        inside |= (across <= radius + tol) & (
            np.abs(along) <= half_length + tol
        )
    i, j, k = np.nonzero(inside)
    R = T[k]

    span = 2 * int(np.abs(R).max()) + 2 * int(np.max(n)) + 3
    offset = span // 2

    def encode(i, j, R):
        Rc = R + offset
        if np.any((Rc < 0) | (Rc >= span)):
            raise ValueError("Pair image outside the encoding range")
        cell = (Rc[:, 0] * span + Rc[:, 1]) * span + Rc[:, 2]
        return (i * len(sites) + j) * span**3 + cell

    key = encode(i, j, R)
    order = np.argsort(key)
    sorted_key = key[order]

    perm, shift = position_permutations(structure, sites)
    label = key.copy()
    for g, (W, _) in enumerate(structure.operators):
        ip, jp = perm[g, i], perm[g, j]
        Rp = R @ np.rint(W).astype(int).T + shift[g, j] - shift[g, i]
        for a, b, Rq in ((ip, jp, Rp), (jp, ip, -Rp)):
            image = encode(a, b, Rq)
            pos = np.searchsorted(sorted_key, image)
            found = (pos < len(key)) & (
                sorted_key[np.minimum(pos, len(key) - 1)] == image
            )
            if not np.all(found):
                raise ValueError("Pair set not closed under symmetry")
            label = np.minimum(label, image)

    d = R + sites.x[j] - sites.x[i]
    dist = structure.distance(d)
    _, first, inverse = np.unique(
        label, return_index=True, return_inverse=True
    )
    rank = np.lexsort((first, np.round(dist[first], 6)))
    order_of = np.empty_like(rank)
    order_of[rank] = np.arange(len(rank))
    orbit = order_of[inverse.ravel()]
    return PairOrbits(i, j, R, d, orbit, dist[first][rank])


def orbit_average(orbits, R, C):
    r"""
    Project a covariance onto orbit coefficients (symmetrize).

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    R : ndarray of int, (k, 3)
        Translations of the covariance array.
    C : ndarray, (k, n, n)
        Covariance ``C[k, i, j]`` :math:`= \langle \delta n_i(\mathbf{0})\,
        \delta n_j(\mathbf{R}_k) \rangle`.

    Returns
    -------
    c : ndarray, (n_orbits,)
        Mean of ``C`` over the pairs of each orbit.

    """

    index = {tuple(r): k for k, r in enumerate(R)}
    try:
        k = np.array([index[tuple(r)] for r in orbits.R])
    except KeyError as error:
        raise ValueError(
            "Covariance lacks a translation of the pairs"
        ) from error
    values = C[k, orbits.i, orbits.j]
    total = np.bincount(
        orbits.orbit, weights=values, minlength=orbits.n_orbits
    )
    count = np.bincount(orbits.orbit, minlength=orbits.n_orbits)
    return total / count


def orbit_pair_terms(orbits, sites, c, thermal=True, b=None):
    """
    Pair terms of one covariance coefficient per orbit.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions.
    c : array_like, (n_orbits,)
        Occupation covariance of each orbit.
    thermal : bool, optional
        Smear by :math:`U_i + U_j`.
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.

    Returns
    -------
    terms : PairTerms

    """

    from rmc_discord.pairs.pairterms import PairTerms

    b = sites.b if b is None else np.asarray(b)
    w = b[orbits.i] * b[orbits.j] * np.asarray(c)[orbits.orbit]
    U = sites.U[orbits.i] + sites.U[orbits.j] if thermal else None
    return PairTerms(orbits.d, w, U)


def _null_space(M, tol=1e-9):
    """Null space with an absolute tolerance.

    Symmetry constraint rows have entries of order one, so singular
    values below ``tol`` are round-off. A tolerance relative to the
    largest singular value (as in ``scipy.linalg.null_space``) fails when
    every row is round-off, e.g. for the identity operator.
    """

    if M.size == 0:
        return np.eye(M.shape[1])
    _, S, Vt = np.linalg.svd(M)
    rank = int(np.sum(S > tol))
    return Vt[rank:].T


def pair_images(orbits, structure, sites):
    """
    Index of the image of every pair under each operator, and its reverse.

    Parameters
    ----------
    orbits : PairOrbits
        A pair set closed under the operators and reversal.
    structure : Structure
    sites : Sites

    Returns
    -------
    image : ndarray of int, (g, m)
        ``image[g, p]`` is the pair that operator ``g`` maps ``p`` to.
    reverse : ndarray of int, (m,)
        Index of the reversed pair ``(j, i, -R)``.

    """

    n = len(sites)
    span = 2 * int(np.abs(orbits.R).max()) + 1
    offset = span // 2

    def encode(i, j, R):
        Rc = R + offset
        cell = (Rc[:, 0] * span + Rc[:, 1]) * span + Rc[:, 2]
        return (i * n + j) * span**3 + cell

    key = encode(orbits.i, orbits.j, orbits.R)
    order = np.argsort(key)

    def find(k):
        pos = np.searchsorted(key[order], k)
        pos = np.minimum(pos, len(key) - 1)
        if not np.all(key[order][pos] == k):
            raise ValueError("Pair set not closed under symmetry")
        return order[pos]

    perm, shift = position_permutations(structure, sites)
    image = np.empty((len(structure.operators), len(orbits)), int)
    for g, (W, _) in enumerate(structure.operators):
        Wi = np.rint(W).astype(int)
        Rp = orbits.R @ Wi.T + shift[g, orbits.j] - shift[g, orbits.i]
        image[g] = find(encode(perm[g, orbits.i], perm[g, orbits.j], Rp))
    reverse = find(encode(orbits.j, orbits.i, -orbits.R))
    return image, reverse


@dataclass
class VectorOrbits:
    r"""
    Symmetry-adapted antisymmetric vector coefficients on pair orbits.

    Each refined coefficient :math:`\theta_k` multiplies a basis vector
    field: pair :math:`p` carries the Cartesian vector
    ``vectors[p, k]`` (zero outside the coefficient's orbit), with
    :math:`\boldsymbol{\sigma}_{g p} = R_g \boldsymbol{\sigma}_p` under
    symmetry and :math:`\boldsymbol{\sigma}_{\bar p} =
    -\boldsymbol{\sigma}_p` under pair reversal.

    Attributes
    ----------
    orbit : ndarray of int, (k,)
        Orbit of each coefficient.
    pairs : list of ndarray
        Pair indices of each coefficient's orbit.
    vectors : list of ndarray
        ``(len(pairs[k]), 3)`` Cartesian basis vectors.

    """

    orbit: np.ndarray
    pairs: list
    vectors: list

    def __len__(self):
        return len(self.orbit)


def vector_orbits(orbits, structure, sites, exclude=(), axis=None):
    """
    Basis of antisymmetric pair vectors allowed by symmetry.

    Used for size-effect coefficients
    ``<dn_i(0) v_j(R)> - <v_i(0) dn_j(R)>``, which change sign under
    pair reversal and rotate with the operators.

    Parameters
    ----------
    orbits : PairOrbits
    structure : Structure
    sites : Sites
    exclude : sequence of int, optional
        Orbits without vector coefficients (e.g. within one cluster).
    axis : array_like, (3,), optional
        Restrict displacements to this fractional direction (e.g.
        ``(0, 0, 1)`` for motion along a chain); the restriction must be
        compatible with the symmetry.

    Returns
    -------
    vectors : VectorOrbits

    """

    image, reverse = pair_images(orbits, structure, sites)
    Ainv = np.linalg.inv(structure.A)
    rot = [structure.A @ W @ Ainv for W, _ in structure.operators]
    if axis is not None:
        u = structure.A @ np.asarray(axis, float)
        u = u / np.linalg.norm(u)
    reps = orbits.representatives()
    excluded = set(np.atleast_1d(exclude).tolist())

    orbit_of, pairs_of, vectors_of = [], [], []
    for o, p0 in enumerate(reps):
        if o in excluded:
            continue
        rows, mapping = [], {}
        for g, R in enumerate(rot):
            for sign, q in ((1, image[g, p0]), (-1, reverse[image[g, p0]])):
                if q == p0:
                    rows.append(R - sign * np.eye(3))
                mapping.setdefault(q, sign * R)
        if axis is not None:
            rows.append(np.eye(3) - np.outer(u, u))
        basis = _null_space(np.vstack(rows)) if rows else np.eye(3)
        members = np.array(sorted(mapping))
        for k in range(basis.shape[1]):
            v = np.array([mapping[q] @ basis[:, k] for q in members])
            orbit_of.append(o)
            pairs_of.append(members)
            vectors_of.append(v)
    return VectorOrbits(np.array(orbit_of, int), pairs_of, vectors_of)


def vector_slots(vectors, n_pairs):
    """
    Compact per-pair view of vector coefficients.

    Returns
    -------
    index : ndarray of int, (m, s)
        Coefficient indices carried by each pair (``-1`` padding).
    value : ndarray, (m, s, 3)
        The corresponding Cartesian basis vectors.

    """

    slots = np.zeros(n_pairs, int)
    for p in vectors.pairs:
        slots[p] += 1
    s = max(1, int(slots.max()))
    index = np.full((n_pairs, s), -1)
    value = np.zeros((n_pairs, s, 3))
    fill = np.zeros(n_pairs, int)
    for k, (p, v) in enumerate(zip(vectors.pairs, vectors.vectors)):
        index[p, fill[p]] = k
        value[p, fill[p]] = v
        fill[p] += 1
    return index, value


def rigid_unit_vector_rows(orbits, vectors, clusters):
    """
    Ties making co-occupied units move and fluctuate as rigid wholes.

    In a co-occupied unit the members share one occupation variable and,
    for a rigid molecule, one displacement, so the size-effect vector of
    every member pair between two units is the same. Ties are imposed at
    the representative pair of each orbit (the vector fields are
    covariant, so this suffices).

    Parameters
    ----------
    orbits : PairOrbits
    vectors : VectorOrbits
    clusters : list of Cluster

    Returns
    -------
    rows : ndarray, (r, len(vectors))
        Each row ``M`` states ``M @ theta = 0``.

    """

    index, value = vector_slots(vectors, len(orbits))
    lookup = {
        (i, j, tuple(R)): p
        for p, (i, j, R) in enumerate(zip(orbits.i, orbits.j, orbits.R))
    }
    unit = {}
    for c in clusters:
        if c.kind == "co-occupied":
            for m, o in zip(c.members, c.offsets):
                unit[m] = (c.members[0], c.offsets[0], o)

    def sigma(p):
        M = np.zeros((3, len(vectors)))
        for k, v in zip(index[p], value[p]):
            if k >= 0:
                M[:, k] += v
        return M

    rows = []
    for p in orbits.representatives():
        i, j, R = orbits.i[p], orbits.j[p], orbits.R[p]
        a0, oa0, oa = unit.get(i, (i, 0, 0))
        b0, ob0, ob = unit.get(j, (j, 0, 0))
        if (a0, b0) == (i, j):
            continue
        ref = lookup.get((a0, b0, tuple(R + oa - oa0 - ob + ob0)))
        if ref is None or ref == p:
            continue
        rows.append(sigma(p) - sigma(ref))
    if not rows:
        return np.zeros((0, len(vectors)))
    M = np.vstack(rows)
    return M[np.any(np.abs(M) > 1e-12, axis=1)]


_SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _sym_basis():
    """Orthonormal basis of symmetric 3x3 matrices (Frobenius)."""

    E = np.zeros((6, 3, 3))
    for k, (a, b) in enumerate(_SYM):
        if a == b:
            E[k, a, a] = 1
        else:
            E[k, a, b] = E[k, b, a] = 1 / np.sqrt(2)
    return E


def _sym_coordinates(T):
    E = _sym_basis()
    return np.einsum("kab,...ab->...k", E, T)


@dataclass
class TensorOrbits:
    r"""
    Symmetry-adapted symmetric tensor coefficients on pair orbits.

    Pair :math:`p` carries ``tensors[k][q]`` for coefficient :math:`k`
    on the ``q``-th pair of ``pairs[k]``, with
    :math:`V_{gp} = R_g V_p R_g^{\mathsf{T}}` under symmetry and
    :math:`V_{\bar p} = V_p` (symmetric part) under pair reversal.

    Attributes
    ----------
    orbit : ndarray of int, (k,)
    pairs : list of ndarray
    tensors : list of ndarray
        ``(len(pairs[k]), 3, 3)`` Cartesian symmetric tensors.

    """

    orbit: np.ndarray
    pairs: list
    tensors: list

    def __len__(self):
        return len(self.orbit)


def tensor_orbits(orbits, structure, sites, exclude=(), axis=None):
    r"""
    Basis of symmetric pair tensors allowed by symmetry.

    Used for displacement-displacement correlations
    :math:`V_{ij}(\mathbf{R}) = \langle \mathbf{v}_i(\mathbf{0})\,
    \mathbf{v}_j(\mathbf{R})^{\mathsf{T}} \rangle`, of which only the
    symmetric part contributes to :math:`\mathbf{Q}^{\mathsf{T}} V
    \mathbf{Q}`.

    Parameters
    ----------
    orbits : PairOrbits
    structure : Structure
    sites : Sites
    exclude : sequence of int, optional
        Orbits without tensor coefficients.
    axis : array_like, (3,), optional
        Restrict displacements to this fractional direction, so that
        :math:`V \propto \hat{\mathbf{u}}\hat{\mathbf{u}}^{\mathsf{T}}`.

    Returns
    -------
    tensors : TensorOrbits

    """

    E = _sym_basis()
    image, reverse = pair_images(orbits, structure, sites)
    Ainv = np.linalg.inv(structure.A)
    rot = [structure.A @ W @ Ainv for W, _ in structure.operators]
    act = [
        _sym_coordinates(np.einsum("ab,kbc,dc->kad", R, E, R)).T for R in rot
    ]
    if axis is not None:
        u = structure.A @ np.asarray(axis, float)
        u = u / np.linalg.norm(u)
        P = np.outer(u, u)
        uu = _sym_coordinates(P)
        # keep only the component along u u^T
        project = np.eye(6) - np.outer(uu, uu) / (uu @ uu)
    reps = orbits.representatives()
    excluded = set(np.atleast_1d(exclude).tolist())

    orbit_of, pairs_of, tensors_of = [], [], []
    for o, p0 in enumerate(reps):
        if o in excluded:
            continue
        rows, mapping = [], {}
        for g, R in enumerate(rot):
            for q in (image[g, p0], reverse[image[g, p0]]):
                if q == p0:
                    rows.append(act[g] - np.eye(6))
                mapping.setdefault(q, R)
        if axis is not None:
            rows.append(project)
        basis = _null_space(np.vstack(rows)) if rows else np.eye(6)
        members = np.array(sorted(mapping))
        for k in range(basis.shape[1]):
            T = np.einsum("k,kab->ab", basis[:, k], E)
            t = np.array([mapping[q] @ T @ mapping[q].T for q in members])
            orbit_of.append(o)
            pairs_of.append(members)
            tensors_of.append(t)
    return TensorOrbits(np.array(orbit_of, int), pairs_of, tensors_of)


def coefficient_slots(fields, n_pairs, shape):
    """
    Compact per-pair view of vector or tensor coefficients.

    Parameters
    ----------
    fields : VectorOrbits or TensorOrbits
    n_pairs : int
    shape : tuple
        ``(3,)`` for vectors or ``(3, 3)`` for tensors.

    Returns
    -------
    index : ndarray of int, (m, s)
        Coefficient indices carried by each pair (``-1`` padding).
    value : ndarray, (m, s) + shape
        The corresponding Cartesian basis values.

    """

    values = fields.vectors if hasattr(fields, "vectors") else fields.tensors
    slots = np.zeros(n_pairs, int)
    for p in fields.pairs:
        slots[p] += 1
    s = max(1, int(slots.max()))
    index = np.full((n_pairs, s), -1)
    value = np.zeros((n_pairs, s) + tuple(shape))
    fill = np.zeros(n_pairs, int)
    for k, (p, v) in enumerate(zip(fields.pairs, values)):
        index[p, fill[p]] = k
        value[p, fill[p]] = v
        fill[p] += 1
    return index, value


def rigid_unit_tensor_rows(orbits, tensors, clusters):
    """
    Ties making co-occupied units move as rigid wholes (tensor version).

    The displacement correlation of every member pair between two rigid
    units is the same tensor. See :func:`rigid_unit_vector_rows`.

    Returns
    -------
    rows : ndarray, (r, len(tensors))

    """

    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    lookup = {
        (i, j, tuple(R)): p
        for p, (i, j, R) in enumerate(zip(orbits.i, orbits.j, orbits.R))
    }
    unit = {}
    for c in clusters:
        if c.kind == "co-occupied":
            for m, o in zip(c.members, c.offsets):
                unit[m] = (c.members[0], c.offsets[0], o)

    def field(p):
        M = np.zeros((9, len(tensors)))
        for k, t in zip(index[p], value[p]):
            if k >= 0:
                M[:, k] += t.ravel()
        return M

    rows = []
    for p in orbits.representatives():
        i, j, R = orbits.i[p], orbits.j[p], orbits.R[p]
        a0, oa0, oa = unit.get(i, (i, 0, 0))
        b0, ob0, ob = unit.get(j, (j, 0, 0))
        if (a0, b0) == (i, j):
            continue
        ref = lookup.get((a0, b0, tuple(R + oa - oa0 - ob + ob0)))
        if ref is None or ref == p:
            continue
        rows.append(field(p) - field(ref))
    if not rows:
        return np.zeros((0, len(tensors)))
    M = np.vstack(rows)
    return M[np.any(np.abs(M) > 1e-12, axis=1)]
