r"""Sampling the measured volume at points commensurate with the supercell.

A periodic :math:`N_1 \times N_2 \times N_3` supercell has a structure factor
only at :math:`\mathbf{h} = \mathbf{m}/\mathbf{N}` (integer
:math:`\mathbf{m}`). At integer :math:`\mathbf{h}` the configuration's
structure factor equals its own average, so the diffuse intensity
:math:`|F - \langle F \rangle|^2` vanishes there identically; those points
carry no information and are dropped. At all other commensurate points
:math:`\langle F \rangle = 0`, so the diffuse intensity is :math:`|F|^2`.

Each valid data bin is assigned to its nearest commensurate point and the
bins are combined by inverse-variance weighting, so any data grid and
projection can be used without rebinning by integer factors.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class QSample:
    r"""
    Observed intensity at commensurate reciprocal-space points.

    Attributes
    ----------
    m : ndarray, (n, 3), int
        Commensurate indices, :math:`\mathbf{h} = \mathbf{m}/\mathbf{N}`.
    N : tuple of int
        Supercell size.
    hkl : ndarray, (n, 3)
        Miller indices.
    Q : ndarray, (n, 3)
        Cartesian scattering vectors (1/A).
    y : ndarray, (n,)
        Observed intensity.
    variance : ndarray, (n,)
        Squared uncertainty of ``y``.
    bins : ndarray, (n,), int
        Number of data bins combined into each point.

    """

    m: np.ndarray
    N: tuple
    hkl: np.ndarray
    Q: np.ndarray
    y: np.ndarray
    variance: np.ndarray
    bins: np.ndarray

    def __len__(self):
        return len(self.y)

    @property
    def weight(self):
        """Inverse variance."""
        return 1.0 / self.variance

    @property
    def Q_norm(self):
        """Magnitude of the scattering vectors (1/A)."""
        return np.linalg.norm(self.Q, axis=1)

    def select(self, keep):
        """
        Subset of the points.

        Parameters
        ----------
        keep : array_like of bool or int
            Mask or indices of the points to keep.

        Returns
        -------
        sample : QSample
            Selected points.

        """

        return QSample(
            m=self.m[keep],
            N=self.N,
            hkl=self.hkl[keep],
            Q=self.Q[keep],
            y=self.y[keep],
            variance=self.variance[keep],
            bins=self.bins[keep],
        )

    def random_subset(self, fraction, rng=None):
        """
        Random subset of the points.

        Parameters
        ----------
        fraction : float
            Fraction of points to keep.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        subset : QSample
            Selected points.
        indices : ndarray, int
            Indices of the selected points.

        """

        rng = np.random.default_rng(rng)
        n = max(1, int(round(fraction * len(self))))
        indices = np.sort(rng.choice(len(self), n, replace=False))
        return self.select(indices), indices


def commensurate(
    volume,
    crystal,
    N,
    hkl_range=None,
    mask=None,
    aggregate="mean",
    norm=None,
):
    """
    Sample a volume at points commensurate with a supercell.

    Parameters
    ----------
    volume : Volume
        Measured intensity, or counts if ``norm`` is given.
    crystal : Crystal
        Average structure (for Cartesian Q).
    N : sequence of int
        Supercell size along a, b, c.
    hkl_range : array_like, (3, 2), optional
        Region of interest (min, max) in h, k, l. The whole volume if
        omitted.
    mask : ndarray of bool, optional
        Bins of the volume to exclude, same shape as the signal.
    aggregate : {"mean", "nearest"}, optional
        ``"mean"`` combines every bin whose center is nearest a commensurate
        point by inverse-variance weighting; ``"nearest"`` takes only the bin
        containing the point.
    norm : Volume, optional
        Normalization of a counts volume (treated as exact). Bins are then
        combined as :math:`\\sum d / \\sum n` with variance
        :math:`\\sum \\sigma_d^2 / (\\sum n)^2`, which keeps zero-count bins
        and avoids the low bias of inverse-variance weights for Poisson
        counts. Point variances are floored at the smallest positive bin
        variance (one count).

    Returns
    -------
    sample : QSample
        Observed intensity at commensurate, non-integer points.

    """

    N = np.asarray(N, dtype=int)

    valid = np.isfinite(volume.signal) & np.isfinite(volume.variance)
    if norm is None:
        valid &= volume.variance > 0
    else:
        if norm.shape != volume.shape or not np.allclose(norm.W, volume.W):
            raise ValueError("counts and normalization are on different grids")
        valid &= np.isfinite(norm.signal) & (norm.signal > 0)
    if mask is not None:
        valid &= ~mask

    index = np.argwhere(valid)
    y = volume.signal[valid]
    var = volume.variance[valid]
    n = norm.signal[valid] if norm is not None else np.ones(len(y))
    hkl_bins = volume.hkl(index)

    if aggregate == "mean":
        m = np.round(hkl_bins * N).astype(int)
    elif aggregate == "nearest":
        m_all = np.round(hkl_bins * N).astype(int)
        h_m = m_all / N
        own, inside = volume.index(h_m)
        same = inside & np.all(own == index, axis=1)
        m, y, var, n = m_all[same], y[same], var[same], n[same]
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    integer = np.all(np.mod(m, N) == 0, axis=1)
    m, y, var, n = m[~integer], y[~integer], var[~integer], n[~integer]

    points, inverse, counts = np.unique(
        m, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    if norm is None:
        w = 1.0 / var
        sw = np.bincount(inverse, weights=w)
        y_point = np.bincount(inverse, weights=w * y) / sw
        var_point = 1.0 / sw
    else:
        sn = np.bincount(inverse, weights=n)
        floor = var[var > 0].min() if np.any(var > 0) else 1.0
        y_point = np.bincount(inverse, weights=y) / sn
        var_point = np.maximum(np.bincount(inverse, weights=var), floor) / sn**2

    hkl = points / N
    sample = QSample(
        m=points,
        N=tuple(int(v) for v in N),
        hkl=hkl,
        Q=crystal.Q(hkl),
        y=y_point,
        variance=var_point,
        bins=counts,
    )

    if hkl_range is not None:
        r = np.asarray(hkl_range, dtype=float)
        keep = np.all((hkl >= r[:, 0] - 1e-9) & (hkl <= r[:, 1] + 1e-9), axis=1)
        sample = sample.select(keep)

    return sample


def bragg_mask(sample, crystal, radius=0.1, shape="ellipsoid", allowed_only=True):
    """
    Points near Bragg reflections.

    Parameters
    ----------
    sample : QSample
        Sample points.
    crystal : Crystal
        Average structure (reflection conditions).
    radius : float or sequence of 3 floats, optional
        Half-width along h, k, l (rlu).
    shape : {"ellipsoid", "box"}, optional
        Shape of the excluded region around each reflection.
    allowed_only : bool, optional
        Exclude only reflections allowed by the space group.

    Returns
    -------
    near : ndarray of bool, (n,)
        True for points to exclude.

    """

    radius = np.broadcast_to(np.asarray(radius, dtype=float), (3,))
    n = np.round(sample.hkl).astype(int)
    d = (sample.hkl - n) / radius

    if shape == "ellipsoid":
        near = np.sum(d**2, axis=1) <= 1
    elif shape == "box":
        near = np.all(np.abs(d) <= 1, axis=1)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    if allowed_only and near.any():
        near[near] = crystal.is_allowed(n[near])

    return near


def outlier_mask(sample, factor=1.5, radius=0.5):
    """
    Points far outside the local interquartile range.

    Points are grouped by nearest integer ``hkl`` within ``radius`` and
    compared to the quartiles of their group, as in the original punch.

    Parameters
    ----------
    sample : QSample
        Sample points.
    factor : float, optional
        Multiple of the interquartile range beyond which a point is an
        outlier.
    radius : float, optional
        Half-width (rlu) of the group around each integer ``hkl``.

    Returns
    -------
    outlier : ndarray of bool, (n,)
        True for points to exclude.

    """

    n = np.round(sample.hkl).astype(int)
    inside = np.all(np.abs(sample.hkl - n) <= radius, axis=1)
    groups, inverse = np.unique(n, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    outlier = np.zeros(len(sample), dtype=bool)
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    for members in np.split(order, splits):
        members = members[inside[members]]
        if len(members) < 4:
            continue
        q1, q3 = np.percentile(sample.y[members], [25, 75])
        iqr = q3 - q1
        bad = (sample.y[members] < q1 - factor * iqr) | (
            sample.y[members] > q3 + factor * iqr
        )
        outlier[members[bad]] = True

    return outlier


def powder_ring_mask(
    sample, bin_width=0.01, baseline_width=0.5, threshold=4.0, dilate=1, exclude=None
):
    """
    Points on powder rings, detected from the radial profile.

    The median intensity in thin :math:`|\\mathbf{Q}|` shells is compared
    with a running median over ``baseline_width``. Shells exceeding the
    baseline by more than ``threshold`` robust standard deviations are
    rings, widened by ``dilate`` shells on each side. Single-crystal
    intensity is anisotropic and barely moves the shell median, so rings
    from polycrystalline contamination or the sample environment stand
    out.

    Parameters
    ----------
    sample : QSample
        Sample points.
    bin_width : float, optional
        Shell width (1/A).
    baseline_width : float, optional
        Width of the running-median baseline (1/A).
    threshold : float, optional
        Detection threshold in robust standard deviations.
    dilate : int, optional
        Extra shells masked on each side of a ring.
    exclude : ndarray of bool, optional
        Points left out of the profile (e.g. near Bragg peaks).

    Returns
    -------
    mask : ndarray of bool, (n_points,)
        True on rings.
    rings : ndarray
        Centers of the masked shells (1/A).

    """

    Q, y = sample.Q_norm, sample.y
    use = np.ones(len(Q), bool) if exclude is None else ~np.asarray(exclude)
    edges = np.arange(Q.min(), Q.max() + bin_width, bin_width)
    shell = np.clip(np.digitize(Q, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    median = np.full(n, np.nan)
    order = np.argsort(shell[use], kind="stable")
    groups = np.split(y[use][order], np.cumsum(np.bincount(shell[use], minlength=n))[:-1])
    for k, values in enumerate(groups):
        if len(values) >= 5:
            median[k] = np.median(values)
    half = max(1, int(round(baseline_width / bin_width / 2)))
    baseline = np.full(n, np.nan)
    for k in range(n):
        w = min(half, k, n - 1 - k)
        window = median[k - w : k + w + 1]
        finite = window[np.isfinite(window)]
        if len(finite):
            baseline[k] = np.median(finite)
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
