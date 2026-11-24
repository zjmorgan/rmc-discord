r"""Linear least-squares refinement of pair-correlation coefficients.

The model for observed intensities :math:`y` with uncertainties
:math:`\sigma` is

.. math::

    y = s\,(A_\mathrm{fixed} \mathbf{c}_\mathrm{fixed}
    + A_\mathrm{free} \mathbf{c}_\mathrm{free}) + B \boldsymbol{\beta},

where :math:`A` holds design columns of the pair orbits,
:math:`\mathbf{c}_\mathrm{fixed}` are coefficients known exactly from
the average structure (they set the absolute scale), :math:`s` is the
scale factor and :math:`B` a smooth background basis. With
:math:`\mathbf{u} = s\,\mathbf{c}_\mathrm{free}` the problem is linear in
:math:`(s, \mathbf{u}, \boldsymbol{\beta})` and is solved in one
weighted least-squares step.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Refinement:
    """
    Refined coefficients and statistics.

    Attributes
    ----------
    c : ndarray, (n_orbits,)
        All orbit coefficients (fixed values included).
    esd : ndarray, (n_orbits,)
        Standard uncertainties (zero for fixed coefficients).
    free : ndarray of bool, (n_orbits,)
        Refined coefficients.
    scale, scale_esd : float
        Scale factor (1 and 0 when no coefficient is fixed).
    background, background_esd : ndarray
        Background coefficients.
    chi2 : float
        Weighted sum of squared residuals.
    n_obs, n_params, rank : int
        Observations, parameters and numerical rank of the problem.
    singular_values : ndarray
        Of the weighted design matrix, for conditioning checks.
    cov : ndarray, (n_orbits, n_orbits) or None
        Covariance of ``c`` when available.

    """

    c: np.ndarray
    esd: np.ndarray
    free: np.ndarray
    scale: float
    scale_esd: float
    background: np.ndarray
    background_esd: np.ndarray
    chi2: float
    n_obs: int
    n_params: int
    rank: int
    singular_values: np.ndarray
    cov: np.ndarray = None  # covariance of c (streaming solver)

    @property
    def reduced_chi2(self):
        return self.chi2 / max(1, self.n_obs - self.n_params)

    def predict(self, A, B=None):
        """Model intensities for design columns ``A`` and basis ``B``."""

        y = self.scale * (A @ self.c)
        if B is not None and len(self.background):
            y = y + B @ self.background
        return y


def r_factors(y, sigma, model):
    r"""
    Agreement factors.

    Returns
    -------
    R : float
        :math:`\sum |y - m| / \sum |y|`.
    wR : float
        :math:`\sqrt{\sum w (y - m)^2 / \sum w y^2}` with
        :math:`w = 1 / \sigma^2`.

    """

    w = 1 / sigma**2
    R = np.sum(np.abs(y - model)) / np.sum(np.abs(y))
    wR = np.sqrt(np.sum(w * (y - model) ** 2) / np.sum(w * y**2))
    return R, wR


def background_basis(Q, degree=1):
    r"""
    Polynomial background in :math:`|\mathbf{Q}|^2`.

    Parameters
    ----------
    Q : ndarray, (m,)
        ``|Q|`` in inverse Angstrom.
    degree : int, optional
        Highest power of :math:`|\mathbf{Q}|^2`; ``-1`` for no background.

    Returns
    -------
    B : ndarray, (m, degree + 1)

    """

    q2 = np.asarray(Q) ** 2 / np.max(np.asarray(Q) ** 2)
    return np.stack([q2**n for n in range(degree + 1)], 1)


def radial_background_basis(Q, knots):
    r"""
    Piecewise-linear isotropic background in :math:`|\mathbf{Q}|`.

    Hat functions on the given knots form a partition of unity, so any
    background that depends only on :math:`|\mathbf{Q}|`, including
    powder rings from the sample environment, is represented to the
    resolution of the knot spacing. Use it when no background
    measurement was subtracted.

    Parameters
    ----------
    Q : ndarray, (m,)
        :math:`|\mathbf{Q}|` in inverse Angstrom.
    knots : ndarray, (k,)
        Increasing knot positions spanning the range of ``Q``.

    Returns
    -------
    B : ndarray, (m, k)

    """

    Q = np.asarray(Q, dtype=float)
    knots = np.asarray(knots, dtype=float)
    B = np.zeros((len(Q), len(knots)))
    for k in range(len(knots)):
        e = np.zeros(len(knots))
        e[k] = 1
        B[:, k] = np.interp(Q, knots, e)
    return B


def refine(
    y,
    sigma,
    A,
    fixed=None,
    B=None,
    parameterization=None,
    ridge=0.0,
    rcond=1e-10,
):
    r"""
    Refine orbit coefficients against observed intensities.

    Coefficients are :math:`\mathbf{c} = \mathbf{c}_0 + N
    \boldsymbol{\theta}` (see
    :func:`rmc_discord.pairs.constraints.parameterize`). When ``c0``
    is nonzero (coefficients fixed by the average structure) it sets the
    absolute scale and a scale factor ``s`` is refined, with
    ``y = s A c0 + (A N) u + B beta`` and ``theta = u / s``. Otherwise
    the coefficients are on the scale of the data (``s = 1``).

    Parameters
    ----------
    y, sigma : ndarray, (m,)
        Observed intensities and uncertainties.
    A : ndarray, (m, n_orbits)
        Design columns from :func:`design_matrix`.
    fixed : tuple of ndarray, optional
        ``(index, value)`` of fixed coefficients; ignored when a
        ``parameterization`` is given.
    B : ndarray, (m, nb), optional
        Background basis.
    parameterization : Parameterization, optional
        Constrained coefficients.
    ridge : float, optional
        Tikhonov damping of ``u`` (in units of the weighted normal
        matrix), zero by default.
    rcond : float, optional
        Relative singular-value cutoff defining the numerical rank.

    Returns
    -------
    result : Refinement

    """

    from rmc_discord.pairs.constraints import parameterize

    y, sigma = np.asarray(y, float), np.asarray(sigma, float)
    n = A.shape[1]
    if parameterization is None:
        parameterization = parameterize(n, fixed)
    c0, N = parameterization.c0, parameterization.N
    k = N.shape[1]
    has_scale = bool(np.any(c0 != 0))
    B = np.zeros((len(y), 0)) if B is None else np.asarray(B, float)

    blocks = [A @ N, B]
    if has_scale:
        blocks.insert(0, (A @ c0)[:, None])
    X = np.hstack(blocks)
    Xw, yw = X / sigma[:, None], y / sigma

    offset = 1 if has_scale else 0
    if ridge > 0:
        damp = np.zeros((k, X.shape[1]))
        damp[np.arange(k), offset + np.arange(k)] = np.sqrt(ridge)
        Xa, ya = np.vstack([Xw, damp]), np.concatenate([yw, np.zeros(k)])
    else:
        Xa, ya = Xw, yw

    U, S, Vt = np.linalg.svd(Xa, full_matrices=False)
    keep = S > rcond * S[0]
    p = Vt[keep].T @ ((U[:, keep].T @ ya) / S[keep])
    cov = (Vt[keep].T / S[keep] ** 2) @ Vt[keep]

    residual = yw - Xw @ p
    chi2 = float(residual @ residual)

    s = p[0] if has_scale else 1.0
    u = p[offset : offset + k]
    c = c0 + N @ u / s

    # delta-method propagation from (s, u) to c = c0 + N u / s
    J = N / s
    if has_scale:
        J = np.hstack([(-N @ u / s**2)[:, None], J])
        block = cov[: offset + k, : offset + k]
    else:
        block = cov[offset : offset + k, offset : offset + k]
    esd = np.sqrt(np.maximum(np.einsum("na,ab,nb->n", J, block, J), 0))

    free = np.any(np.abs(N) > 1e-10, axis=1)
    beta = p[offset + k :]
    beta_esd = np.sqrt(np.diag(cov)[offset + k :])
    return Refinement(
        c,
        esd,
        free,
        float(s),
        float(np.sqrt(cov[0, 0])) if has_scale else 0.0,
        beta,
        beta_esd,
        chi2,
        len(y),
        X.shape[1],
        int(keep.sum()),
        S,
    )
