r"""Exact linear constraints on orbit coefficients and their solution.

Two kinds of relation are imposed, both independent of any model of the
local order:

* unit rules: members of a co-occupied unit (e.g. a carbonate group)
  share one occupation variable, so their covariances with any position
  are equal;
* sum rules: a split-site cluster without vacancies always holds exactly
  one atom, so the fluctuations of its members sum to zero and their
  covariances with any position sum to zero,
  :math:`\sum_a C(a \,@\, \mathbf{o}_a,\, j \,@\, \mathbf{R}) = 0`;
* homometric ties: orbits whose pair vectors, scattering-length products
  and displacement tensors coincide give identical intensities, so only
  their sum is measurable; they are tied equal (each then carries the
  average of the group).

Together with coefficients fixed by the average structure they define
:math:`\mathbf{c} = \mathbf{c}_0 + N \boldsymbol{\theta}` with free
parameters :math:`\boldsymbol{\theta}`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


def _pair_lookup(orbits):
    return {
        (i, j, tuple(R)): o
        for i, j, R, o in zip(orbits.i, orbits.j, orbits.R, orbits.orbit)
    }


def sum_rule_rows(orbits, clusters, vacancy_tol=1e-3):
    """
    Sum-rule constraints of clusters without vacancies.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    clusters : list of Cluster
        Split-site clusters covering all positions.
    vacancy_tol : float, optional
        Clusters with ``vacancy`` above this get no rule.

    Returns
    -------
    rows : ndarray, (r, n_orbits)
        Each row ``M`` states ``M @ c = 0``. Rows that would involve
        pairs beyond the cutoff are omitted.

    """

    lookup = _pair_lookup(orbits)
    rows = set()
    for cluster in clusters:
        if (
            cluster.kind != "exclusive"
            or cluster.vacancy > vacancy_tol
            or len(cluster) < 2
        ):
            continue
        a0, o0 = cluster.members[0], cluster.offsets[0]
        for j, R0 in zip(orbits.j[orbits.i == a0], orbits.R[orbits.i == a0]):
            R = R0 + o0
            terms = {}
            for a, oa in zip(cluster.members, cluster.offsets):
                o = lookup.get((a, j, tuple(R - oa)))
                if o is None:
                    break
                terms[o] = terms.get(o, 0) + 1
            else:
                rows.add(tuple(sorted(terms.items())))
    M = np.zeros((len(rows), orbits.n_orbits))
    for r, row in enumerate(rows):
        for o, v in row:
            M[r, o] = v
    return M


def unit_rows(orbits, clusters):
    """
    Equality constraints of co-occupied units.

    Members of a co-occupied unit share one occupation variable, so for
    every member ``a`` and position ``j`` in cell ``R``,
    ``C(a @ o_a, j @ R) = C(a0 @ o_a0, j @ R)``.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    clusters : list of Cluster
        Clusters covering all positions.

    Returns
    -------
    rows : ndarray, (r, n_orbits)
        Each row ``M`` states ``M @ c = 0``. Rows that would involve
        pairs beyond the cutoff are omitted.

    """

    lookup = _pair_lookup(orbits)
    rows = set()
    for cluster in clusters:
        if cluster.kind != "co-occupied" or len(cluster) < 2:
            continue
        a0, o0 = cluster.members[0], cluster.offsets[0]
        for j, R0 in zip(orbits.j[orbits.i == a0], orbits.R[orbits.i == a0]):
            R = R0 + o0
            ref = lookup[(a0, j, tuple(R0))]
            for a, oa in zip(cluster.members[1:], cluster.offsets[1:]):
                o = lookup.get((a, j, tuple(R - oa)))
                if o is not None and o != ref:
                    rows.add((min(o, ref), max(o, ref)))
    M = np.zeros((len(rows), orbits.n_orbits))
    for r, (o, ref) in enumerate(sorted(rows)):
        M[r, o], M[r, ref] = 1, -1
    return M


def homometric_groups(orbits, sites, decimals=6, b=None, adp=True):
    """
    Groups of orbits that give identical intensities.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions (scattering lengths and ADPs).
    decimals : int, optional
        Rounding used to compare pair vectors and tensors.
    b : array_like, optional
        Amplitudes compared instead of the neutron scattering lengths;
        ones group orbits by pair vectors and ADPs alone (used for
        magnetic pairs, whose mixed-site amplitudes differ only slightly,
        so that such orbits are nearly collinear and not identifiable).
    adp : bool, optional
        Also compare the ADPs; without, orbits differing only by small
        Debye-Waller differences are grouped too.

    Returns
    -------
    group : ndarray of int, (n_orbits,)
        Group label of each orbit; singletons label themselves.

    """

    b = sites.b if b is None else np.asarray(b, float)
    U = sites.U[orbits.i] + sites.U[orbits.j]
    iu = np.triu_indices(3)
    rows = np.hstack(
        [
            orbits.d,
            (b[orbits.i] * b[orbits.j])[:, None],
            U[:, iu[0], iu[1]] if adp else np.zeros((len(U), 0)),
        ]
    )
    rows = np.round(rows, decimals)
    signature = {}
    group = np.arange(orbits.n_orbits)
    for o in range(orbits.n_orbits):
        r = rows[orbits.orbit == o]
        key = r[np.lexsort(r.T[::-1])].tobytes()
        group[o] = signature.setdefault(key, o)
    return group


def tie_rows(group, exclude=()):
    """Rows ``c_o - c_g = 0`` tying each orbit to its group label."""

    exclude = set(np.atleast_1d(exclude).tolist())
    pairs = [
        (o, g)
        for o, g in enumerate(group)
        if o != g and o not in exclude and g not in exclude
    ]
    M = np.zeros((len(pairs), len(group)))
    for r, (o, g) in enumerate(pairs):
        M[r, o], M[r, g] = 1, -1
    return M


@dataclass
class Parameterization:
    r"""
    Coefficients :math:`\mathbf{c} = \mathbf{c}_0 + N \boldsymbol{\theta}`
    satisfying all constraints.

    Attributes
    ----------
    c0 : ndarray, (n_orbits,)
        Fixed values plus the particular solution of the constraints.
    N : ndarray, (n_orbits, k)
        Orthonormal basis of free directions (zero on fixed orbits).
    residual : float
        Largest constraint violation of ``c0`` (from rounding of the
        average-structure occupancies).

    """

    c0: np.ndarray
    N: np.ndarray
    residual: float

    @property
    def n_free(self):
        return self.N.shape[1]

    def coefficients(self, theta):
        return self.c0 + self.N @ theta


def parameterize(n_orbits, fixed=None, rows=None, tol=1e-3):
    """
    Solve fixed values and linear constraints for free directions.

    Parameters
    ----------
    n_orbits : int
        Number of orbit coefficients.
    fixed : tuple of ndarray, optional
        ``(index, value)`` of fixed coefficients.
    rows : ndarray, (r, n_orbits), optional
        Constraint rows ``M @ c = 0``.
    tol : float, optional
        Largest tolerated violation of the constraints by the fixed
        values.

    Returns
    -------
    parameterization : Parameterization

    """

    free = np.ones(n_orbits, bool)
    c0 = np.zeros(n_orbits)
    if fixed is not None and len(fixed[0]):
        free[fixed[0]] = False
        c0[fixed[0]] = fixed[1]
    N = np.zeros((n_orbits, free.sum()))
    N[free] = np.eye(free.sum())
    residual = 0.0
    if rows is not None and len(rows):
        Mf, Mx = rows[:, free], rows[:, ~free]
        rhs = -Mx @ c0[~free]
        particular = np.linalg.lstsq(Mf, rhs, rcond=None)[0]
        residual = float(np.max(np.abs(Mf @ particular - rhs), initial=0))
        if residual > tol:
            raise ValueError(f"Fixed values violate constraints ({residual})")
        c0[free] = particular
        basis = null_space(Mf)
        N = np.zeros((n_orbits, basis.shape[1]))
        N[free] = basis
    return Parameterization(c0, N, residual)


def affine_parameterization(n, rows, rhs, tol=1e-9):
    r"""
    Coefficients satisfying inhomogeneous constraints :math:`M\mathbf{c} = \mathbf{r}`.

    Parameters
    ----------
    n : int
        Number of coefficients.
    rows : ndarray, (r, n)
        Constraint rows.
    rhs : ndarray, (r,)
        Right-hand sides.

    Returns
    -------
    parameterization : Parameterization
        Minimum-norm particular solution and the null space of ``rows``.

    """

    rows = np.atleast_2d(np.asarray(rows, float))
    rhs = np.atleast_1d(np.asarray(rhs, float))
    c0 = np.linalg.lstsq(rows, rhs, rcond=None)[0]
    residual = float(np.max(np.abs(rows @ c0 - rhs), initial=0))
    if residual > tol:
        raise ValueError(f"inconsistent constraints ({residual})")
    return Parameterization(c0, null_space(rows), residual)


def self_trace_rows(orbits, tensors, value=1.0):
    r"""
    Unit-spin normalization of symmetric spin tensors.

    Unit spins have :math:`\operatorname{tr}\langle \hat{\mathbf{S}}_i
    \hat{\mathbf{S}}_i^\mathsf{T}\rangle = 1` on every magnetic position;
    fixing the trace of each self tensor (its anisotropic part stays
    free) sets the absolute scale, as the self orbit does for isotropic
    correlations.

    Parameters
    ----------
    orbits : PairOrbits
    tensors : TensorOrbits
        From :func:`~rmc_discord.pairs.symmetry.tensor_orbits`.
    value : float, optional
        Trace of each self tensor.

    Returns
    -------
    rows : ndarray, (n_self, len(tensors))
    rhs : ndarray, (n_self,)

    """

    from rmc_discord.pairs.symmetry import coefficient_slots

    index, basis = coefficient_slots(tensors, len(orbits), (3, 3))
    reps = orbits.representatives()
    self_orbits = [o for o in range(orbits.n_orbits) if orbits.distance[o] < 1e-8]
    rows = np.zeros((len(self_orbits), len(tensors)))
    for r, o in enumerate(self_orbits):
        p = reps[o]
        for slot, k in enumerate(index[p]):
            if k >= 0:
                rows[r, k] = np.trace(basis[p, slot])
    return rows, np.full(len(self_orbits), float(value))


def vector_groups(orbits, sites, exclude=(), columns=None, decimals=6):
    """
    Orbits with the same set of pair vectors.

    Their intensities differ only by the amplitude prefactor (scattering
    lengths, Debye-Waller factors, form factors) and by how often each
    vector occurs, so the data determine only a weighted sum of their
    coefficients. Orbits are compared by their *distinct* vectors (e.g.
    C2-C2 and O7-O7 between carbonate units stacked along c share the
    vectors but not the number of atom pairs).

    Parameters
    ----------
    orbits : PairOrbits
    sites : Sites
    exclude : sequence of int, optional
        Orbits kept as their own group (e.g. fixed ones).
    columns : ndarray of int, optional
        Orbit of each coefficient, if not all orbits are coefficients.

    Returns
    -------
    group : ndarray of int, (n_columns,)
        Group label of each coefficient (the smallest member's position).
    multiplicity : ndarray, (n_columns,)
        Pairs per distinct vector of each orbit, the factor by which its
        column exceeds the column of one pair per vector.
    """

    columns = np.arange(orbits.n_orbits) if columns is None else np.asarray(columns)
    exclude = set(np.asarray(exclude, int).tolist())
    d = np.round(np.asarray(orbits.d, float), decimals)
    order = np.argsort(orbits.orbit, kind="stable")
    bounds = np.searchsorted(orbits.orbit[order], np.arange(orbits.n_orbits + 1))
    group = np.arange(len(columns))
    multiplicity = np.ones(len(columns))
    first = {}
    for k, o in enumerate(columns):
        v = d[order[bounds[o] : bounds[o + 1]]]
        distinct = np.unique(v, axis=0)
        multiplicity[k] = len(v) / len(distinct)
        if int(o) in exclude:
            continue
        group[k] = first.setdefault(distinct.tobytes(), k)
    return group, multiplicity


def collapse_parameterization(par, group, weight, tol=1e-9):
    r"""
    Keep only the free directions that change a group's weighted sum.

    With :math:`\mathbf{c} = \mathbf{c}_0 + N\boldsymbol{\theta}` and
    group sums :math:`\phi_g = \sum_{k \in g} w_k c_k = (M\mathbf{c})_g`,
    directions of :math:`\boldsymbol{\theta}` in the null space of
    :math:`MN` leave every :math:`\phi_g` unchanged and are dropped:
    :math:`N \to N V` with :math:`V` an orthonormal basis of the row space
    of :math:`MN`. Constraints on individual coefficients are kept, and a
    group whose members the constraints already separate stays resolved.

    Parameters
    ----------
    par : Parameterization
    group : ndarray of int, (n,)
        Group label of each coefficient.
    weight : ndarray, (n,)
        Weight of each coefficient in its group sum (e.g. :math:`b_i b_j`).

    Returns
    -------
    Parameterization
    """

    if par.n_free == 0:
        return par
    labels, inverse = np.unique(group, return_inverse=True)
    M = np.zeros((len(labels), len(group)))
    M[inverse.ravel(), np.arange(len(group))] = weight
    _, S, Vt = np.linalg.svd(M @ par.N, full_matrices=False)
    keep = S > tol * max(S.max(), 1e-300)
    return Parameterization(par.c0, par.N @ Vt[keep].T, par.residual)


def stack_parameterizations(*parts):
    """
    Combine independent coefficient blocks into one parameterization.

    Parameters
    ----------
    *parts : Parameterization
        Blocks in column order, e.g. occupational orbit coefficients
        then size-effect vector coefficients.

    Returns
    -------
    parameterization : Parameterization
        ``c = c0 + N theta`` over the concatenated coefficients, with a
        block-diagonal ``N``.

    """

    from scipy.linalg import block_diag

    c0 = np.concatenate([p.c0 for p in parts])
    N = block_diag(*[p.N for p in parts])
    residual = max(p.residual for p in parts)
    return Parameterization(c0, N, residual)
