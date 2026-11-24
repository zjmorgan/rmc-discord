r"""
Diffuse-scattering observations from a measured volume.

Bins are used as measured (no snapping to a supercell grid): centers
:math:`\mathbf{h}`, intensities, uncertainties, the bin widths and the
projection matrix for exact bin averaging. Bragg regions, a
:math:`|\mathbf{Q}|` range and powder rings can be masked, and
Laue-equivalent bins can be merged into one observation per orbit.
"""

from dataclasses import dataclass, field

import numpy as np

from rmc_discord.pairs.selection import bragg_mask, canonical, powder_ring_mask


@dataclass
class Observations:
    """
    Intensities at measured bins.

    Attributes
    ----------
    h : ndarray, (m, 3)
        Miller indices of the bin centers (one measured member per orbit
        when merged).
    y, sigma : ndarray, (m,)
        Intensities and uncertainties.
    index : ndarray of int, (m, 3)
        Volume bin of each observation.
    count : ndarray of int, (m,)
        Bins merged into each observation.
    widths : ndarray, (3,)
        Bin widths along the volume axes.
    projection : ndarray, (3, 3)
        Projection matrix of the volume (``h = W u``).
    held : ndarray of bool, (m,)
        Held-out observations for validation.
    info : dict
        Masking summary.

    """

    h: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    index: np.ndarray
    count: np.ndarray
    widths: np.ndarray
    projection: np.ndarray
    held: np.ndarray
    info: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)

    def select(self, keep):
        keep = np.asarray(keep)
        return Observations(self.h[keep], self.y[keep], self.sigma[keep], self.index[keep], self.count[keep],
                            self.widths, self.projection, self.held[keep], dict(self.info))

    def split(self, fraction, seed=0):
        """Same observations with a new random held-out set."""
        held = np.random.default_rng(seed).random(len(self)) < fraction
        return Observations(self.h, self.y, self.sigma, self.index, self.count, self.widths,
                            self.projection, held, dict(self.info))


def from_volume(
    volume,
    structure,
    bragg_radius=0.15,
    q_range=(0.0, np.inf),
    h_max=None,
    rings=False,
    merge="mean",
    extra=(),
    error_floor=0.0,
    held_out=0.1,
    seed=0,
):
    r"""
    Observations from a volume.

    Parameters
    ----------
    volume : rmc_discord.data.volume.Volume
        Measured (diffuse) intensity.
    structure : Structure
        Average structure (Bragg positions, Laue group, metric).
    bragg_radius : float, optional
        Exclusion radius around allowed Bragg peaks (inverse Angstrom);
        zero keeps them.
    q_range : tuple of float, optional
        Kept :math:`|\mathbf{Q}|` range (inverse Angstrom).
    h_max : float, optional
        Keep :math:`\max_a |h_a| \le` ``h_max``.
    rings : bool, optional
        Mask powder rings found in the radial profile.
    merge : {"mean", "first", None}, optional
        Merge Laue-equivalent bins: inverse-variance mean (for data that
        are not symmetrized), one member (symmetrized data), or keep all.
    extra : sequence, optional
        Extra Laue operators when the data symmetry exceeds the
        structure's (see :func:`~rmc_discord.pairs.selection.laue_operators`).
    error_floor : float, optional
        Relative uncertainty added in quadrature, :math:`\sigma^2 \to
        \sigma^2 + (f y)^2`.
    held_out : float, optional
        Fraction of observations held out for validation.
    seed : int, optional
        Seed of the held-out split.

    Returns
    -------
    obs : Observations

    """

    valid = np.isfinite(volume.signal) & np.isfinite(volume.variance) & (volume.variance > 0)
    index = np.argwhere(valid)
    h = volume.hkl(index)
    y = volume.signal[valid]
    var = volume.variance[valid]
    info = {"bins": int(valid.sum()), "q_range": tuple(float(q) for q in q_range), "bragg_radius": float(bragg_radius),
            "h_max": h_max}

    keep = np.ones(len(y), bool)
    Q = np.linalg.norm(structure.Q(h), axis=1)
    keep &= (Q >= q_range[0]) & (Q <= q_range[1])
    if h_max is not None:
        keep &= np.all(np.abs(h) <= h_max, axis=1)
    masked = {"bragg": h[:0], "rings": h[:0]}
    if bragg_radius > 0:
        bragg = np.zeros(len(h), bool)
        bragg[keep] = bragg_mask(h[keep], structure, bragg_radius)
        info["bragg"] = int((bragg & keep).sum())
        masked["bragg"] = h[bragg & keep]
        keep &= ~bragg
    if rings:
        ring, centers = powder_ring_mask(Q[keep], y[keep])
        info["ring_centers"] = centers
        sub = np.flatnonzero(keep)
        info["rings"] = int(ring.sum())
        masked["rings"] = h[sub[ring]]
        keep[sub[ring]] = False
    info["masked"] = masked
    index, h, y, var = index[keep], h[keep], y[keep], var[keep]
    var = var + (error_floor * y) ** 2

    count = np.ones(len(y), dtype=np.int64)
    if merge is not None and len(y):
        key = canonical(h, structure, extra=extra)
        _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        count = np.bincount(inverse)
        if merge == "mean":
            w = 1.0 / var
            sw = np.bincount(inverse, w)
            y = np.bincount(inverse, w * y) / sw
            var = 1.0 / sw
        else:
            y, var = y[first], var[first]
        index, h = index[first], h[first]
    info["observations"] = int(len(y))
    info["extra"] = tuple(extra)

    held = np.random.default_rng(seed).random(len(y)) < held_out
    return Observations(h, y, np.sqrt(var), index, count, np.asarray(volume.widths, float),
                        np.asarray(volume.W, float), held, info)
