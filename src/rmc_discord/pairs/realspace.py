r"""Real-space view of pair terms: the model 3D-:math:`\Delta`\ PDF.

The pair terms :math:`(\mathbf{d}_p, w_p, U_p)` define

.. math::

    \Delta P(\mathbf{r}) = \sum_p w_p\, G(\mathbf{r} - \mathbf{d}_p;\, U_p),

whose Fourier transform is the diffuse intensity evaluated by
:func:`rmc_discord.pairs.design.evaluate`.
"""

import numpy as np
from scipy.special import erf


def delta_pdf_slab(terms, structure, x, y, z_range, window=5.0):
    r"""
    Model 3D-delta-PDF averaged over a slab of constant ``z``.

    Each Gaussian is integrated exactly over ``z`` in ``z_range`` using
    its conditional distribution given ``(x, y)``, and divided by the
    slab thickness, matching a slab-averaged experimental map.

    Parameters
    ----------
    terms : PairTerms
        Smeared pair terms (``U`` positive definite). Merged terms are
        not supported; pass the full reversal-closed set.
    structure : Structure
        Supplies the lattice.
    x, y : ndarray
        Regular grids of fractional coordinates along the first two
        axes.
    z_range : tuple of float
        Slab limits in fractional coordinates along the third axis.
    window : float, optional
        Gaussians are evaluated within this many standard deviations.

    Returns
    -------
    pdf : ndarray, (len(y), len(x))
        Density in :math:`\mathrm{fm}^2` per unit cell per
        :math:`\mathring{\mathrm{A}}^3`.

    """

    if terms.U is None:
        raise ValueError("Point terms have no real-space density")

    Ainv = np.linalg.inv(structure.A)
    V = Ainv @ terms.U @ Ainv.T  # fractional covariance
    Vxy = V[:, :2, :2]
    Vxy_inv = np.linalg.inv(Vxy)
    gain = np.einsum("pa,pab->pb", V[:, 2, :2], Vxy_inv)
    s = np.sqrt(V[:, 2, 2] - np.einsum("pa,pa->p", gain, V[:, 2, :2]))

    z1, z2 = z_range
    sz = np.sqrt(V[:, 2, 2])
    dx, dy = x[1] - x[0], y[1] - y[0]
    reach = window * np.sqrt(np.max(V[:, :2, :2].diagonal(0, 1, 2), axis=1))
    keep = (
        (terms.d[:, 2] > z1 - window * sz)
        & (terms.d[:, 2] < z2 + window * sz)
        & (terms.d[:, 0] > x[0] - reach)
        & (terms.d[:, 0] < x[-1] + reach)
        & (terms.d[:, 1] > y[0] - reach)
        & (terms.d[:, 1] < y[-1] + reach)
    )

    pdf = np.zeros((len(y), len(x)))
    K = int(np.ceil(reach[keep].max() / min(dx, dy))) if keep.any() else 0
    k = np.arange(-K, K + 1)
    for chunk in np.array_split(np.flatnonzero(keep), 1 + keep.sum() // 4096):
        d = terms.d[chunk]
        ix = np.rint((d[:, 0] - x[0]) / dx).astype(int)[:, None, None] + k
        iy = np.rint((d[:, 1] - y[0]) / dy).astype(int)[:, None, None]
        iy = iy + k[:, None]
        ix, iy = np.broadcast_arrays(ix, iy)
        inside = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        rx = x[np.clip(ix, 0, len(x) - 1)] - d[:, 0, None, None]
        ry = y[np.clip(iy, 0, len(y) - 1)] - d[:, 1, None, None]
        r = np.stack([rx, ry], -1)
        Vi = Vxy_inv[chunk]
        quad = np.einsum("pija,pab,pijb->pij", r, Vi, r)
        norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Vxy[chunk])))
        mu = d[:, 2, None, None] + np.einsum("pa,pija->pij", gain[chunk], r)
        sc = np.sqrt(2) * s[chunk, None, None]
        frac = 0.5 * (erf((z2 - mu) / sc) - erf((z1 - mu) / sc))
        value = terms.w[chunk, None, None] * norm[:, None, None]
        value = value * np.exp(-0.5 * quad) * frac / (z2 - z1)
        np.add.at(pdf, (iy[inside], ix[inside]), value[inside])

    return pdf / abs(np.linalg.det(structure.A))
