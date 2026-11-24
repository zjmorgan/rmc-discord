r"""Crystallographic selection of reciprocal-space positions.

Measured volumes are reduced to the positions that carry independent
diffuse information:

* Bragg regions are excluded around reflections allowed by the lattice
  centering;
* symmetrized data are reduced to one representative per Laue orbit;
* diffuse features are targeted like satellites, at points
  :math:`\mathbf{G} + \mathbf{k}` or along lines
  :math:`\mathbf{G} + \mathbf{k} + t \mathbf{v}` for allowed
  :math:`\mathbf{G}` and the Laue star of :math:`\mathbf{k}` and
  :math:`\mathbf{v}`;
* a reproducible random fraction of the remaining positions samples the
  background between features.
"""

from dataclasses import dataclass, field

import numpy as np


def centering_vectors(structure, tol=1e-6):
    """
    Pure translations among the operators (lattice centering).

    Returns
    -------
    t : ndarray, (m, 3)
        Fractional centering vectors including the origin.

    """

    t = [
        T % 1
        for W, T in structure.operators
        if np.allclose(W, np.eye(3)) and np.all(np.isfinite(T))
    ]
    t = np.unique(np.round(np.array(t), 8) % 1, axis=0)
    return t


def is_allowed(h, structure, tol=1e-6):
    """
    Integer reflections allowed by the lattice centering.

    Parameters
    ----------
    h : array_like, (..., 3)
        Miller indices; non-integer positions are never allowed.
    structure : Structure

    Returns
    -------
    allowed : ndarray of bool, (...)

    """

    h = np.asarray(h, dtype=float)
    integer = np.all(np.abs(h - np.rint(h)) < tol, axis=-1)
    phase = h @ centering_vectors(structure).T
    centered = np.all(np.abs(phase - np.rint(phase)) < tol, axis=-1)
    return integer & centered


def _rotation(op):
    if isinstance(op, str):
        from rmc_discord.pairs.structure import parse_symop

        op = parse_symop(op)[0]
    return np.rint(np.asarray(op)).astype(int)


def laue_operators(structure, extra=()):
    """
    Laue group rotations acting on Miller indices as ``h @ W``.

    Parameters
    ----------
    structure : Structure
        Supplies the point-group rotations of the space group.
    extra : sequence, optional
        Additional rotations (symmetry-operator strings such as
        ``"y, x, z"`` or 3x3 matrices) when the data were symmetrized to
        a higher Laue group than that of the structure, e.g. by
        merohedral twinning. The group is closed under products.

    Returns
    -------
    W : ndarray of int, (g, 3, 3)
        Distinct rotations of the Laue group, inversion included.

    """

    W = [np.rint(W).astype(int) for W, _ in structure.operators]
    W += [_rotation(op) for op in extra]
    group = {w.tobytes(): w for w in W + [-w for w in W]}
    while True:
        new = {}
        for a in group.values():
            for b in group.values():
                c = a @ b
                if c.tobytes() not in group:
                    new[c.tobytes()] = c
        if not new:
            break
        group.update(new)
        if len(group) > 48:
            raise ValueError(
                "Extra operators do not form a crystallographic group"
            )
    return np.unique(np.array(list(group.values())), axis=0)


def coset_operators(structure, extra=()):
    """
    Representatives of the data Laue group modulo that of the structure.

    A model with the symmetry of the structure is compared with data
    symmetrized to the larger group by averaging it over these
    rotations (``h @ W``).

    Parameters
    ----------
    structure : Structure
    extra : sequence, optional
        Additional rotations, as for :func:`laue_operators`.

    Returns
    -------
    W : ndarray of int, (n, 3, 3)
        Coset representatives, the identity first.

    """

    small = {w.tobytes() for w in laue_operators(structure)}
    covered, reps = set(), [np.eye(3, dtype=int)]
    covered |= small
    for w in laue_operators(structure, extra):
        if w.tobytes() in covered:
            continue
        reps.append(w)
        covered |= {
            (w @ np.frombuffer(g, int).reshape(3, 3)).tobytes() for g in small
        }
    return np.array(reps)


def _nearest_allowed(r, structure):
    """Nearest allowed reciprocal-lattice vector to each row of ``r``."""

    shifts = np.array(
        np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij")
    ).reshape(3, -1)
    candidates = np.rint(r)[:, None, :] + shifts.T[None]
    allowed = is_allowed(candidates, structure)
    dQ = structure.Q(r[:, None, :] - candidates)
    dist = np.linalg.norm(dQ, axis=-1)
    dist[~allowed] = np.inf
    k = np.argmin(dist, axis=1)
    return candidates[np.arange(len(r)), k], dist[np.arange(len(r)), k]


def bragg_mask(h, structure, radius):
    """
    Positions within ``radius`` (inverse Angstrom) of allowed Bragg peaks.

    Parameters
    ----------
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    radius : float
        Exclusion radius in inverse Angstrom.

    Returns
    -------
    mask : ndarray of bool, (m,)

    """

    from scipy.spatial import cKDTree

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    if not len(h):
        return np.zeros(0, bool)
    # allowed reflections around the data, searched in Cartesian Q (exact nearest peak)
    lo, hi = np.floor(h.min(0)) - 1, np.ceil(h.max(0)) + 1
    G = np.array(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij")).reshape(3, -1).T
    G = G[is_allowed(G, structure)]
    dist, _ = cKDTree(structure.Q(G)).query(structure.Q(h), distance_upper_bound=radius, workers=-1)
    return dist < radius


def _star(vectors, structure):
    W = laue_operators(structure)
    stars = np.einsum("ka,gab->gkb", np.atleast_2d(vectors), W)
    return np.unique(np.round(stars.reshape(-1, 3), 8), axis=0)


def satellite_mask(h, structure, k, radius):
    r"""
    Positions within ``radius`` of satellites :math:`\mathbf{G} + \mathbf{k}`
    (star of :math:`\mathbf{k}`).

    Parameters
    ----------
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    k : array_like, (3,)
        Propagation vector in r.l.u.
    radius : float
        Inverse Angstrom.

    Returns
    -------
    mask : ndarray of bool, (m,)

    """

    h = np.asarray(h, dtype=float)
    mask = np.zeros(len(h), bool)
    for kk in _star(k, structure):
        _, dist = _nearest_allowed(h - kk, structure)
        mask |= dist < radius
    return mask


def line_mask(h, structure, k, direction, radius):
    r"""
    Positions within ``radius`` of lines
    :math:`\mathbf{G} + \mathbf{k} + t \mathbf{v}`.

    The Laue group is applied to ``(k, v)`` jointly, so the star of the
    line is generated. Distance is measured perpendicular to the line.

    Parameters
    ----------
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    k : array_like, (3,)
        A point on the line in r.l.u.
    direction : array_like, (3,)
        Line direction in r.l.u.
    radius : float
        Inverse Angstrom.

    Returns
    -------
    mask : ndarray of bool, (m,)

    """

    h = np.asarray(h, dtype=float)
    W = laue_operators(structure)
    k = np.asarray(k, dtype=float) @ W
    v = np.asarray(direction, dtype=float) @ W
    lines = np.unique(np.round(np.hstack([k, v]), 8), axis=0)
    shifts = np.array(
        np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij")
    ).reshape(3, -1)
    mask = np.zeros(len(h), bool)
    for kk, vv in zip(lines[:, :3], lines[:, 3:]):
        # smallest allowed lattice step along the line
        step = next(
            m * vv
            for m in range(1, 13)
            if is_allowed(m * vv, structure) and np.any(m * vv)
        )
        u = structure.Q(vv)
        u = u / np.linalg.norm(u)
        r = h - kk
        # perpendicular distance is invariant to allowed steps along v
        t = (structure.Q(r) @ u) / np.linalg.norm(structure.Q(step))
        r = r - np.outer(np.rint(t), step)
        candidates = np.rint(r)[:, None, :] + shifts.T[None]
        dQ = structure.Q(r[:, None, :] - candidates)
        perp = dQ - (dQ @ u)[..., None] * u
        dist = np.linalg.norm(perp, axis=-1)
        dist[~is_allowed(candidates, structure)] = np.inf
        mask |= dist.min(axis=1) < radius
    return mask


def asymmetric_unit(index, shape, center, structure):
    """
    Representatives of Laue orbits on a centered regular grid.

    Parameters
    ----------
    index : tuple of ndarray
        Grid indices of candidate positions.
    shape : tuple of int
        Grid shape.
    center : tuple of int
        Index of h = 0 along each axis; the grid must be symmetric about
        it with equal steps along h, k, l so that Laue images of grid
        points are grid points.
    structure : Structure

    Returns
    -------
    keep : ndarray of bool
        True for the representative (largest encoded image) of each
        orbit among the candidates.

    """

    n = np.stack(index, -1) - np.asarray(center)
    size = np.asarray(shape)

    def encode(m):
        c = m + np.asarray(center)
        if np.any((c < 0) | (c >= size)):
            raise ValueError("Grid is not closed under the Laue group")
        return (c[:, 0] * size[1] + c[:, 1]) * size[2] + c[:, 2]

    own = encode(n)
    best = own.copy()
    for W in laue_operators(structure):
        best = np.maximum(best, encode(n @ W))
    return own == best


@dataclass
class QSelection:
    """
    Selected reciprocal-space positions.

    Attributes
    ----------
    index : tuple of ndarray
        Grid indices into the volume.
    h : ndarray, (m, 3)
        Miller indices.
    category : ndarray of str, (m,)
        ``"feature"`` or ``"background"``.
    seed : int
        Random seed of the background sampling.

    """

    index: tuple
    h: np.ndarray
    category: np.ndarray
    seed: int = 0
    info: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.h)

    def subset(self, mask):
        return QSelection(
            tuple(i[mask] for i in self.index),
            self.h[mask],
            self.category[mask],
            self.seed,
            dict(self.info),
        )


def select(
    volume,
    structure,
    bragg_radius,
    features=(),
    feature_radius=None,
    background_fraction=0.0,
    q_range=(0.0, np.inf),
    symmetrized=True,
    seed=0,
):
    r"""
    Select positions of a measured volume for refinement.

    Parameters
    ----------
    volume : Volume
        Measured data on a regular grid.
    structure : Structure
    bragg_radius : float
        Exclusion radius around allowed reflections (inverse Angstrom).
    features : sequence of dict, optional
        Feature definitions, each ``{"k": ...}`` for satellites or
        ``{"k": ..., "direction": ...}`` for lines.
    feature_radius : float, optional
        Capture radius of features (inverse Angstrom).
    background_fraction : float, optional
        Fraction of the remaining valid positions sampled at random.
    q_range : tuple of float, optional
        Allowed :math:`|\mathbf{Q}|` range in inverse Angstrom.
    symmetrized : bool, optional
        Keep one representative per Laue orbit.
    seed : int, optional
        Seed for background sampling.

    Returns
    -------
    selection : QSelection

    """

    valid = np.isfinite(volume.signal) & (volume.sigma > 0)
    index = np.nonzero(valid)
    if symmetrized:
        center = tuple(int(np.argmin(np.abs(c))) for c in volume.centers)
        keep = asymmetric_unit(index, volume.signal.shape, center, structure)
        index = tuple(i[keep] for i in index)

    h = volume.hkl(index)
    Q = np.linalg.norm(structure.Q(h), axis=1)
    keep = (Q >= q_range[0]) & (Q <= q_range[1])
    keep &= ~bragg_mask(h, structure, bragg_radius)
    index, h = tuple(i[keep] for i in index), h[keep]

    feature = np.zeros(len(h), bool)
    for f in features:
        if "direction" in f:
            feature |= line_mask(
                h, structure, f["k"], f["direction"], feature_radius
            )
        else:
            feature |= satellite_mask(h, structure, f["k"], feature_radius)

    rng = np.random.default_rng(seed)
    background = ~feature & (rng.random(len(h)) < background_fraction)
    chosen = feature | background
    category = np.where(feature, "feature", "background")[chosen]
    info = {
        "bragg_radius": bragg_radius,
        "features": list(features),
        "feature_radius": feature_radius,
        "background_fraction": background_fraction,
        "q_range": q_range,
        "symmetrized": symmetrized,
    }
    return QSelection(
        tuple(i[chosen] for i in index), h[chosen], category, seed, info
    )


def canonical(h, structure, decimals=6, extra=()):
    """
    Canonical Laue-equivalent of each position.

    The representative is the lexicographically largest image, so
    equivalent positions share one key regardless of grid or ordering.

    Parameters
    ----------
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    decimals : int, optional
        Rounding used to compare images.
    extra : sequence, optional
        Additional rotations of the data symmetry, as for
        :func:`laue_operators`.

    Returns
    -------
    rep : ndarray, (m, 3)

    """

    W = laue_operators(structure, extra).astype(float)
    h = np.asarray(h, dtype=float)
    best = None
    for w in W:  # running lexicographic maximum, one image at a time
        image = np.round(h @ w, decimals)
        if best is None:
            best = image
            continue
        x, b = image.T, best.T
        larger = (x[0] > b[0]) | ((x[0] == b[0]) & ((x[1] > b[1]) | ((x[1] == b[1]) & (x[2] > b[2]))))
        best[larger] = image[larger]
    return best


def in_fundamental_domain(h, structure, extra=(), decimals=6):
    """
    Positions lying in the Laue fundamental domain.

    A position is kept when it is its own canonical image. Unlike
    :func:`asymmetric_unit` this needs no grid closed under the Laue
    group (e.g. hexagonal data on an orthogonal grid), and keeps about
    one position per orbit of symmetrized data.

    Parameters
    ----------
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    extra : sequence, optional
        Additional rotations of the data symmetry.
    decimals : int, optional
        Rounding used to compare images.

    Returns
    -------
    keep : ndarray of bool, (m,)

    """

    h = np.asarray(h, dtype=float)
    rep = canonical(h, structure, decimals, extra)
    return np.all(np.isclose(rep, np.round(h, decimals)), axis=1)


def powder_ring_mask(
    Q, y, bin_width=0.01, baseline_width=0.5, threshold=4.0, dilate=1
):
    r"""
    Positions on powder rings, detected from the radial profile.

    The median intensity in thin :math:`|\mathbf{Q}|` shells is compared
    with a running median over ``baseline_width``. Shells exceeding the
    baseline by more than ``threshold`` robust standard deviations
    (median absolute deviation of the excess) are rings, and are
    widened by ``dilate`` shells on each side. Single-crystal Bragg and
    diffuse intensity is anisotropic and barely moves the shell median,
    so rings from polycrystalline contamination or the sample
    environment stand out.

    Parameters
    ----------
    Q : ndarray, (m,)
        :math:`|\mathbf{Q}|` in inverse Angstrom.
    y : ndarray, (m,)
        Intensities.
    bin_width : float, optional
        Shell width in inverse Angstrom.
    baseline_width : float, optional
        Width of the running-median baseline in inverse Angstrom.
    threshold : float, optional
        Detection threshold in robust standard deviations.
    dilate : int, optional
        Extra shells masked on each side of a detected ring.

    Returns
    -------
    mask : ndarray of bool, (m,)
        True on rings.
    rings : ndarray
        Centers of the masked shells in inverse Angstrom.

    """

    Q, y = np.asarray(Q, float), np.asarray(y, float)
    edges = np.arange(Q.min(), Q.max() + bin_width, bin_width)
    shell = np.clip(np.digitize(Q, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    median = np.full(n, np.nan)
    order = np.argsort(shell, kind="stable")
    bounds = np.searchsorted(shell[order], np.arange(n + 1))
    ys = y[order]
    for k in range(n):
        values = ys[bounds[k] : bounds[k + 1]]
        if len(values) >= 5:
            median[k] = np.median(values)
    half = max(1, int(round(baseline_width / bin_width / 2)))
    baseline = np.empty(n)
    for k in range(n):
        # symmetric window, shrinking at the ends so a sloped profile
        # does not produce a one-sided (biased) baseline
        w = min(half, k, n - 1 - k)
        window = median[k - w : k + w + 1]
        finite = window[np.isfinite(window)]
        baseline[k] = np.median(finite) if len(finite) else np.nan
    excess = median - baseline
    scale = 1.4826 * np.nanmedian(np.abs(excess - np.nanmedian(excess)))
    ring = np.nan_to_num(excess) > threshold * scale
    ring[[0, -1]] = False
    for _ in range(dilate):
        grown = ring.copy()
        grown[1:] |= ring[:-1]
        grown[:-1] |= ring[1:]
        ring = grown
    centers = 0.5 * (edges[1:] + edges[:-1])
    return ring[shell], centers[ring]
