r"""Bragg scattering of the average structure, on the diffuse scale.

The diffuse model is an intensity per unit cell expressed as a density
in reciprocal lattice units. In the same units a Bragg reflection of the
average structure integrates to :math:`|F(\mathbf{G})|^2`, because the
lattice sum :math:`N^{-1}|\sum_\mathbf{R} e^{2\pi i \mathbf{h}\cdot
\mathbf{R}}|^2` tends to :math:`\delta(\mathbf{h} - \mathbf{G})`. With a
normalized peak profile :math:`P`,

.. math::

    I_\mathrm{Bragg}(\mathbf{h}) = \sum_\mathbf{G} |F(\mathbf{G})|^2\,
    P(\mathbf{h} - \mathbf{G}), \qquad \int P\, d^3h = 1,

so Bragg and diffuse intensities share one scale factor: including the
Bragg peaks anchors the absolute scale of the refined correlations
without punching them out. The profile is a Gaussian in
:math:`\mathbf{Q}` whose covariance has a constant part and radial and
transverse parts growing with :math:`|\mathbf{Q}|` (resolution and
mosaic), averaged numerically over each measured bin.
"""

from dataclasses import dataclass

import numpy as np


def structure_factor(h, structure, sites=None, b=None, thermal=True):
    r"""
    Structure factor of the average structure.

    .. math::

        F(\mathbf{h}) = \sum_a b_a x_a e^{-W_a}
        e^{2\pi i \mathbf{h}\cdot\mathbf{x}_a}

    Parameters
    ----------
    h : array_like, (m, 3)
        Miller indices.
    structure : Structure
    sites : Sites, optional
        Positions to include; all positions of the cell by default.
    b : array_like, optional
        Scattering lengths in fm; neutron values by default.
    thermal : bool, optional
        Include Debye-Waller factors.

    Returns
    -------
    F : ndarray of complex, (m,)
        In fm per unit cell.

    """

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    sites = structure.sites() if sites is None else sites
    b = sites.b if b is None else np.asarray(b)
    weight = b * sites.occupancy
    phase = np.exp(2j * np.pi * h @ sites.x.T)
    if thermal:
        Q = structure.Q(h)
        W = 0.5 * np.einsum("qa,pab,qb->qp", Q, sites.U, Q)
        return (weight * np.exp(-W) * phase).sum(axis=1)
    return (weight * phase).sum(axis=1)


@dataclass
class Profile:
    r"""
    Gaussian Bragg peak profile in :math:`\mathbf{Q}`.

    The covariance at reflection :math:`\mathbf{G}` (unit vector
    :math:`\hat{\mathbf{q}}`) is

    .. math::

        \Sigma = \sigma_0^2 I + (\rho |\mathbf{Q}|)^2
        \hat{\mathbf{q}}\hat{\mathbf{q}}^{\mathsf{T}}
        + (\tau |\mathbf{Q}|)^2 (I - \hat{\mathbf{q}}
        \hat{\mathbf{q}}^{\mathsf{T}}).

    Attributes
    ----------
    sigma0 : float
        Constant width in inverse Angstrom.
    radial : float
        Relative radial width (e.g. :math:`\Delta d / d`).
    transverse : float
        Transverse width per :math:`|\mathbf{Q}|` (mosaic, radians).

    """

    sigma0: float
    radial: float = 0.0
    transverse: float = 0.0

    def covariance(self, QG):
        """Covariance matrices (n, 3, 3) at Cartesian vectors ``QG``."""

        QG = np.asarray(QG, dtype=float).reshape(-1, 3)
        q = np.linalg.norm(QG, axis=1)
        qhat = QG / np.where(q > 0, q, 1)[:, None]
        outer = qhat[:, :, None] * qhat[:, None, :]
        eye = np.eye(3)[None]
        return (
            self.sigma0**2 * eye
            + ((self.radial * q) ** 2)[:, None, None] * outer
            + ((self.transverse * q) ** 2)[:, None, None] * (eye - outer)
        )


def _bin_samples(widths, projection, n):
    """Midpoint sample offsets (in h) spanning one measured bin."""

    t = (np.arange(n) + 0.5) / n - 0.5
    box = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    u = box * np.asarray(widths, dtype=float)
    W = np.eye(3) if projection is None else np.asarray(projection, float)
    return u @ W.T


def bragg_design(
    h,
    structure,
    profile,
    widths=None,
    projection=None,
    operators=None,
    samples=3,
    reach=4.0,
    sites=None,
    thermal=True,
):
    r"""
    Bragg intensity of the average structure at measured positions.

    Each position is the mean over ``samples**3`` midpoints of its bin
    (bin widths ``widths`` along the volume axes, projection ``W``) of
    :math:`\sum_\mathbf{G} |F(\mathbf{G})|^2 P(\mathbf{h} - \mathbf{G})`,
    averaged over ``operators`` (``h @ W``) when the data symmetry is
    higher than that of the structure. All integer reflections allowed
    by the lattice centering within ``reach`` standard deviations are
    included.

    Parameters
    ----------
    h : array_like, (m, 3)
        Miller indices of bin centers.
    structure : Structure
    profile : Profile
    widths : array_like, (3,), optional
        Bin widths along the volume axes; point evaluation if omitted.
    projection : ndarray, (3, 3), optional
        Projection matrix of the volume.
    operators : ndarray of int, (n, 3, 3), optional
        Rotations to average over.
    samples : int, optional
        Midpoints per axis within a bin.
    reach : float, optional
        Profile truncation in standard deviations.
    sites : Sites, optional
        Positions in the structure factor; all by default.
    thermal : bool, optional
        Include Debye-Waller factors.

    Returns
    -------
    I : ndarray, (m,)
        Intensity per unit cell, on the scale of the diffuse model.

    """

    from rmc_discord.pairs.selection import is_allowed

    h = np.asarray(h, dtype=float).reshape(-1, 3)
    if operators is None or len(operators) == 0:
        operators = [np.eye(3)]
    offsets = (
        np.zeros((1, 3))
        if widths is None
        else _bin_samples(widths, projection, samples)
    )
    jacobian = (2 * np.pi) ** 3 * abs(np.linalg.det(structure.B))
    # renormalize the truncated Gaussian so each peak carries |F|^2
    from scipy.special import gammainc

    jacobian /= gammainc(1.5, reach**2 / 2)
    shifts = np.stack(
        np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)

    cache = {}

    def F2(G):
        key = [tuple(g) for g in G.astype(int)]
        missing = [k for k in dict.fromkeys(key) if k not in cache]
        if missing:
            values = (
                np.abs(
                    structure_factor(
                        np.array(missing), structure, sites, thermal=thermal
                    )
                )
                ** 2
            )
            cache.update(zip(missing, values))
        return np.array([cache[k] for k in key])

    out = np.zeros(len(h))
    for W in operators:
        W = np.asarray(W, dtype=float)
        for off in offsets:
            hs = (h + off) @ W
            G = np.rint(hs)[:, None, :] + shifts[None]
            G = G.reshape(-1, 3)
            point = np.repeat(np.arange(len(h)), len(shifts))
            keep = is_allowed(G, structure)
            G, point = G[keep], point[keep]
            unique, which = np.unique(G, axis=0, return_inverse=True)
            which = which.ravel()
            QU = structure.Q(unique)
            S = profile.covariance(QU)
            Sinv = np.linalg.inv(S)
            norm = 1 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(S))
            dQ = structure.Q(hs[point]) - QU[which]
            m2 = np.einsum("pa,pab,pb->p", dQ, Sinv[which], dQ)
            near = m2 < reach**2
            density = np.exp(-0.5 * m2[near]) * norm[which[near]]
            weight = F2(unique)[which[near]]
            np.add.at(out, point[near], weight * density * jacobian)
    return out / (len(offsets) * len(operators))


def fit_profile(
    y,
    sigma,
    h,
    structure,
    initial,
    others=None,
    background=None,
    fit=("sigma0", "radial", "transverse"),
    **kwargs,
):
    """
    Refine the Bragg profile by variable projection.

    For each trial profile the model ``y = s * bragg + others @ a +
    background @ beta`` is linear in ``s``, ``a`` and ``beta`` and is
    solved exactly; only the profile widths are optimized nonlinearly.

    Parameters
    ----------
    y, sigma : ndarray, (m,)
        Observed intensities and uncertainties.
    h : ndarray, (m, 3)
        Miller indices.
    structure : Structure
    initial : Profile
        Starting widths.
    others : ndarray, (m, k), optional
        Further fixed-shape columns fitted alongside (e.g. diffuse).
    background : ndarray, (m, nb), optional
        Background basis.
    fit : sequence of str, optional
        Profile widths to refine.
    **kwargs
        Passed to :func:`bragg_design` (bin widths, projection,
        operators, samples).

    Returns
    -------
    profile : Profile
        Refined widths.
    scale : float
        Scale factor of the Bragg term.
    chi2 : float
        Reduced chi-squared at the solution.

    """

    from scipy.optimize import least_squares

    names = list(fit)
    start = np.log([getattr(initial, n) for n in names])
    extra = [a for a in (others, background) if a is not None]

    def solve(logp):
        params = dict(vars(initial))
        params.update(zip(names, np.exp(logp)))
        profile = Profile(**params)
        X = np.column_stack(
            [bragg_design(h, structure, profile, **kwargs)] + extra
        )
        Xw, yw = X / sigma[:, None], y / sigma
        p, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return profile, p, yw - Xw @ p

    result = least_squares(lambda lp: solve(lp)[2], start, x_scale=1.0)
    profile, p, r = solve(result.x)
    return (
        profile,
        float(p[0]),
        float(r @ r / max(1, len(y) - len(p) - len(names))),
    )
